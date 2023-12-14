import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

############# GENERAL HELPER FUNCTIONS FROM LUCIDRAINS LINFORMER ######################
def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def layer_drop(layers, prob):
    to_drop = torch.empty(len(layers)).uniform_(0, 1) < prob
    blocks = [block for block, drop in zip(layers, to_drop) if not drop]
    blocks = layers[:1] if len(blocks) == 0 else blocks
    return blocks

def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x
# helper classes

#Residual connections
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

#Module for applying layer normalization before layer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

#GeLU activation
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}, layer_dropout = 0.):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return 


##################SPARSE ATTENTION************************
def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = torch.transpose(x, 0, 2, 1, 3)
    x = torch.reshape(x, [n, t, embd])
    return x

################# STANDARD MULTIHEADED SELF ATTENTION #######################
class SparseSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, heads = 8, dropout = 0., blocksize=32):
        super().__init__()

        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len

        self.heads = heads #Number of heads

        self.blocksize = blocksize

        dim_head = dim // heads #Set the dimension of each attention head
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False) #Multiply Q by its weights

        self.to_k = nn.Linear(dim, dim_head * heads, bias = False) #Multiply K by its weights

        self.to_v = nn.Linear(dim, dim_head * heads, bias = False) #Multiply V by its weights

        self.dropout = nn.Dropout(dropout) #Dropout layer
        self.to_out = nn.Linear(dim_head * heads, dim) #Linear layer for output
    
    def forward(self, x, context = None, **kwargs):
        b, n, d, d_h, h = *x.shape, self.dim_head, self.heads

        kv_len = n if context is None else context.shape[1] #Keys and values must be the same length as input
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x) #Multiply queries by weights

        kv_input = x if context is None else context #Input keys and values

        keys = self.to_k(kv_input) #Multiply keys by weights
        values = self.to_v(kv_input) #Multiply values by weights

        n_ctx = queries.size()[1]
        local_attn_ctx = 32

        queries = strided_transpose(queries, n_ctx, local_attn_ctx, self.blocksize)
        keys = strided_transpose(keys, n_ctx, local_attn_ctx, self.blocksize)
        values = strided_transpose(values, n_ctx, local_attn_ctx, self.blocksize)

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, n, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        #From linformer: bhnd,bhkd->bhnk
        dots = torch.einsum('bhnd,bhnd->bhn', queries, keys) * (d_h ** -0.5) #Dot product the queries and keys, normalize by dimension
        print("Got past this einsum!")
        attn = dots.softmax(dim=-1) #Compute softmax
        attn = self.dropout(attn) #Dropout layer
        #From linformer: bhnk,bhkd->bhnd
        out = torch.einsum('bhn,bhnd->bhnd', attn, values ) #Multiply result of attention by the values

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class SparseEncoderLayer(nn.Module):
    def __init__(self, dim, seq_len, heads = 8, dropout = 0.):
        super().__init__()
        layers = nn.ModuleList([])
        attn = SparseSelfAttention(dim=dim, seq_len=seq_len, heads=heads, dropout=dropout)
        ff = FeedForward(dim, dropout = dropout)

        layers.append(nn.ModuleList([
            PreNorm(dim, attn),
            PreNorm(dim, ff)
        ]))

        # execute_type = ReversibleSequence if reversible else SequentialSequence
        execute_type = SequentialSequence
        self.net = execute_type(layers)

    def forward(self, x):
        return self.net(x)




if __name__ == "__main__":
    sparse_self_attention = SparseSelfAttention(dim=16, seq_len=256).cuda()
    sparse_layer = SparseEncoderLayer(dim=16, seq_len=256).cuda()

    with torch.no_grad():
        torch_input = torch.rand(1, 256, 16).cuda()

    out = sparse_self_attention(torch_input)
    out = sparse_layer(torch_input)
    print("complete!")