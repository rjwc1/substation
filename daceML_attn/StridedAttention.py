import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from daceml.torch import dace_module

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError('Not yet implemented')
    b = torch.reshape(b, [1, 1, n, n])
    return b

def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    # assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    # x = torch.transpose(x, 0, 2, 1, 3)
    x = torch.transpose(x, 0, 2)
    x = torch.transpose(x, 1, 3)
    x = torch.reshape(x, [n, t, embd])
    return x

def split_heads(x, n):
    return torch.transpose(split_states(x, n), 0, 2, 1, 3)

def merge_heads(x):
    return merge_states(torch.transpose(x, 0, 2, 1, 3))

def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return torch.reshape(x, new_x_shape)

def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = x.size()
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return torch.reshape(x, new_x_shape)

def attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None):
    q = split_heads(q, heads)
    k = split_heads(k, heads)
    v = split_heads(v, heads)
    n_timesteps = k.size()[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx).float()
    w = torch.matmul(q, k.transpose(-2, -1))
    scale_amount = 1.0 / np.sqrt(q.size()[-1])
    w = w * scale_amount
    w = w * mask + -1e9 * (1 - mask)
    w = F.softmax(w, dim=-1)
    a = torch.matmul(w, v)
    a = merge_heads(a)
    return a

def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=32, num_verts=None, vertsize=None):
    n_ctx = q.size()[1]
    if attn_mode == 'strided':
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = q.size()[-1] // heads
    scale_amount = 1.0 / np.sqrt(n_state)
    w = torch.matmul(q, k.transpose(-2, -1))
    w = F.softmax(w * scale_amount, dim=-1)
    a = torch.matmul(w, v)
    if attn_mode == 'strided':
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        # a = torch.transpose(a, 0, 2, 1, 3)
        a = torch.transpose(a, 0, 2)
        a = torch.transpose(a, 1, 3)
        a = torch.reshape(a, [n, t, embd])
    return a

@dace_module(cuda=True, backward=False, auto_optimize=True, simplify=True)
class SparseAttention(nn.Module):
    def __init__(self, heads, attn_mode, local_attn_ctx=None, blocksize=32):
        # super(SparseAttention, self).__init__()
        super().__init__()
        self.heads = heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize

    def forward(self, q, k, v):
        return blocksparse_attention_impl(q, k, v, self.heads, self.attn_mode, self.local_attn_ctx)


# Example usage:
if __name__ == "__main__":
    n_batch = 1
    n_ctx = 256
    n_embd = 256
    heads = 8
    attn_mode = "strided"
    local_attn_ctx = 32
    blocksize = 32

    q = torch.randn(n_batch, n_ctx, n_embd).cuda()
    k = torch.randn(n_batch, n_ctx, n_embd).cuda()
    v = torch.randn(n_batch, n_ctx, n_embd).cuda()

    model = SparseAttention(heads, attn_mode, local_attn_ctx, blocksize).cuda()
    # output = model(q, k, v)
    # print(output[0])

    from daceml.testing.profiling import time_funcs, print_time_statistics

    def run_dace():
        out = model(q, k, v)

    times = time_funcs([run_dace],
                    func_names=["dace"],
                    warmups=5,
                    num_iters=100)
    print_time_statistics(times, ["dace"])

    from daceml.transformation import TaskletFusion
    from dace.transformation.dataflow import Vectorization, TrivialMapRangeElimination
    from dace.transformation.subgraph import SubgraphFusion
    from dace.transformation.dataflow import MapReduceFusion
    from daceml.util import utils
    from dace.library import change_default
    from daceml import onnx as donnx

    model.reset_sdfg()
    def expand_and_simplify(module):
        # use the pure expansions of operators
        with change_default(donnx, "pure"):
            utils.auto_optimize(module.sdfg, cuda=True, simplify=True)

    model.append_post_onnx_hook("auto_optimize", expand_and_simplify)

    # apply tasklet fusion
    # model.append_post_onnx_hook("fuse_tasklets", lambda x:\
    #         x.dace_model.sdfg.apply_transformations_repeated(TaskletFusion))

    # model.append_post_onnx_hook("gpu_transformation", lambda x:\
    #         x.dace_model.sdfg.apply_gpu_transformations())

    # model.append_post_onnx_hook("map_fusion", lambda x:\
    #         x.dace_model.sdfg.apply_transformations(MapReduceFusion))
    
    # apply vectorization
    def vectorize(fwd, bwd):
        fwd.apply_transformations(Vectorization)
        # bwd.apply_transformations(Vectorization)
    
    # model.append_post_autodiff_hook("vectorize", vectorize)

    times = time_funcs([run_dace],
                   func_names=["dace"],
                   warmups=5,
                   num_iters=100)
    print_time_statistics(times, ["dace"])
