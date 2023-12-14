import math
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


##################### LINFORMER SELF ATTENTION ####################################
class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()

        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k #The projection dimension, a constant you can set

        self.heads = heads #Number of heads

        dim_head = dim // heads #Set the dimension of each attention head
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False) #Multiply Q by its weights

        kv_dim = dim_head if one_kv_head else (dim_head * heads) #Can optimize by sharing parameters across linear projection matrices for each head
        self.to_k = nn.Linear(dim, kv_dim, bias = False) #Multiply K by its weights
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k))) #Project k

        self.share_kv = share_kv #Can optimize by sharing the same projection matrix across keys and values
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout) #Dropout layer
        self.to_out = nn.Linear(dim_head * heads, dim) #Linear layer for output

    def forward(self, x, context = None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1] #Keys and values must be the same length as input
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x) #Multiply queries by weights

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context #Input keys and values

        keys = self.to_k(kv_input) #Multiply keys by weights
        values = self.to_v(kv_input) if not self.share_kv else keys  

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k) #If stated, share the key and value projections

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        # attention
        print("Queries:")
        print(queries.shape)
        print("Keys:")
        print(keys.shape)

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))


        # attention
        print("Queries:")
        print(queries.shape)
        print("Keys:")
        print(keys.shape)

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5) #Dot product the queries and keys, normalize by dimension
        attn = dots.softmax(dim=-1) #Compute softmax
        attn = self.dropout(attn) #Dropout layer
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values ) #Multiply result of attention by the values

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class LinformerEncoderLayer(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dropout = 0.):
        super().__init__()
        layers = nn.ModuleList([])
        attn = LinformerSelfAttention(dim=dim, seq_len=seq_len, k=k, heads=heads, one_kv_head=True, share_kv=True, dropout=0)
        # attn = MultiheadSelfAttention(dim=dim, seq_len=seq_len, k=k, heads=heads, dropout=dropout)
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





if __name__ == '__main__':
    linformer_self_attention = LinformerSelfAttention(dim=16, seq_len=256).cuda()
    linformer_layer = LinformerEncoderLayer(dim=16, seq_len=256).cuda()

    embed_dim = 16
    num_heads = 8
    torch_attn = torch.nn.MultiheadAttention(embed_dim, num_heads).cuda()

    with torch.no_grad():
        torch_input = torch.rand(1, 256, 16).cuda()

    out = linformer_self_attention(torch_input)
    out = linformer_layer(torch_input)
    print("complete!")