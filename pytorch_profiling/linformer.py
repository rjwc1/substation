import math
import torch
from torch import nn
import torch.nn.functional as F
from daceml.torch import dace_module

# from linformer.reversible import ReversibleSequence, SequentialSequence

# helper functions

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

# helper classes
# @dace_module
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

# @dace_module
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# @dace_module
class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# @dace_module
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
        return x

# @dace_module
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

@dace_module(cuda=True, backward=False)
class LinformerSelfAttentionDaCe(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context = None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        # assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context = None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

# @dace_module
class Linformer(nn.Module):
    def __init__(self, dim, seq_len, depth, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, reversible = False, dropout = 0.):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn = LinformerSelfAttention(dim, seq_len, k = k, heads = heads, dim_head = dim_head, one_kv_head = one_kv_head, share_kv = share_kv, dropout = dropout)
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

# @dace_module(cuda=True, backward=False)
class LinformerDaCe(nn.Module):
    def __init__(self, dim, seq_len, depth, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, reversible = False, dropout = 0.):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn = LinformerSelfAttention(dim, seq_len, k = k, heads = heads, dim_head = dim_head, one_kv_head = one_kv_head, share_kv = share_kv, dropout = dropout)
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

# @dace_module
class LinformerLM(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, reversible = False, dropout = 0.):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.linformer = Linformer(dim, seq_len, depth, k = k, heads = heads, dim_head = dim_head,
                one_kv_head = one_kv_head, share_kv = share_kv, reversible = reversible, dropout = dropout)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = self.linformer(x)
        out = self.to_logits(x)
        return out

if __name__ == '__main__':
    l = LinformerSelfAttention(16,16).cuda()
    ld = LinformerSelfAttentionDaCe(16,16)

    with torch.no_grad():
        dace_input = torch.rand(16,16,16).cuda()
        torch_input = torch.clone(dace_input)

    # dace_input.requires_grad = True
    # torch_input.requires_grad = True

    dace_output = ld(dace_input)
    torch_output = l(torch_input)

    # assert torch.allclose(dace_output, torch_output)

    from daceml.testing.profiling import time_funcs, print_time_statistics

    def run_dace():
        out = ld(dace_input)
        # out.backward(dace_dy)


    def run_torch():
        out = l(torch_input)
        # out.backward(torch_dy)


    times = time_funcs([run_dace, run_torch],
                    func_names=["dace", "torch"],
                    warmups=5,
                    num_iters=100)
    print_time_statistics(times, ["dace", "torch"])

    from daceml.transformation import TaskletFusion
    from dace.transformation.dataflow import Vectorization, TrivialMapRangeElimination
    from dace.transformation.subgraph import SubgraphFusion
    from daceml.util import utils
    from dace.library import change_default
    from daceml import onnx as donnx

    ld.reset_sdfg()
    def expand_and_simplify(module):
        # use the pure expansions of operators
        with change_default(donnx, "pure"):
            utils.auto_optimize(module.sdfg, cuda=True, simplify=True)

    ld.append_post_onnx_hook("auto_optimize", expand_and_simplify)

    # apply subgraph fusion
    # def fuse_sg(module):
    #     sdfg = module.sdfg
    #     sdfg.apply_transformations_repeated(TrivialMapRangeElimination)
    #     SubgraphFusion.apply_to(sdfg, *sdfg.node(0).nodes())


    # ld.append_post_onnx_hook("subgraph_fusion", fuse_sg)

    # apply tasklet fusion
    ld.append_post_onnx_hook("fuse_tasklets", lambda x:\
            x.dace_model.sdfg.apply_transformations_repeated(TaskletFusion))


    # apply vectorization
    def vectorize(fwd, bwd):
        fwd.apply_transformations(Vectorization)
        # bwd.apply_transformations(Vectorization)


    ld.append_post_autodiff_hook("vectorize", vectorize)

    times = time_funcs([run_dace, run_torch],
                   func_names=["dace", "torch"],
                   warmups=5,
                   num_iters=100)
    print_time_statistics(times, ["dace", "torch"])

    # l.forward(torch.ones(16,16,16))

    # l.forward_sdfg
    # PreNorm.to_sdfg().save('prenorm.sdfg')
    # GELU_.to_sdfg().save('gelu.sdfg')
    # FeedForward.to_sdfg().save('feedforward.sdfg')
    # LinformerSelfAttention.to_sdfg().save('linformerselfattention.sdfg')
    # Linformer.to_sdfg().save('linformer.sdfg')
    # LinformerLM.to_sdfg().save('linformerlm.sdfg')
