""" Wrapper for DaCe pytorch implementations. """

import torch
import torch.nn
from torch.nn.init import xavier_uniform_, ones_, zeros_, uniform_
from dace.codegen.compiled_sdfg import CompiledSDFG
import numpy as np

from substation.dtypes import torch_dtype

import dace

from substation.transformer_sdfg import encoder


class DaceEncoder(torch.nn.Module):
    def __init__(self, N: int, P: int, H: int, csdfg_fwd: CompiledSDFG,
                 csdfg_bwd: CompiledSDFG):
        """
        Initialize a DaCe-implemented encoder PyTorch module.
        :param P: Head dimension.
        :param H: Number of heads.
        :param csdfg_fwd: Compiled SDFG for forward pass.
        :param csdfg_bwd: Compiled SDFG for backpropagation.
        """
        super().__init__()

        class DaceEncoderFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, WQ, WK, WV, WO, BQ, BK, BV, BO, scaler,
                        norm1_scale, norm1_bias, norm2_scale, norm2_bias,
                        linear1_w, linear1_b, linear2_w, linear2_b,
                        attn_dropout, linear1_dropout, ff_dropout):
                B, SM, N = x.shape

                output = torch.empty([B, SM, N], device=x.device)
                params = dict(x=x.contiguous(),
                              attn_wq=WQ.contiguous(),
                              attn_wk=WK.contiguous(),
                              attn_wv=WV.contiguous(),
                              attn_wo=WO.contiguous(),
                              attn_bq=BQ.contiguous(),
                              attn_bk=BK.contiguous(),
                              attn_bv=BV.contiguous(),
                              attn_bo=BO.contiguous(),
                              attn_scale=scaler.item(),
                              norm1_scale=norm1_scale.contiguous(),
                              norm1_bias=norm1_bias.contiguous(),
                              norm2_scale=norm2_scale.contiguous(),
                              norm2_bias=norm2_bias.contiguous(),
                              linear1_w=linear1_w.contiguous(),
                              linear1_b=linear1_b.contiguous(),
                              linear2_w=linear2_w.contiguous(),
                              linear2_b=linear2_b.contiguous(),
                              attn_dropout=attn_dropout.contiguous(),
                              linear1_dropout=linear1_dropout.contiguous(),
                              ff_dropout=ff_dropout.contiguous(),
                              __return=output,
                              B=B,
                              P=P,
                              H=H,
                              N=H * P,
                              emb=4 * N,
                              SM=SM)
                csdfg_fwd(**params)
                ctx.save_for_backward(x, output)
                return output

            @staticmethod
            def backward(ctx, grads):
                # ctx.saved_tensors
                output = torch.empty_like(K, device='cuda')
                gK, gQ, gV, gWK, gWQ, gWV = [
                    torch.empty(grads.shape, grads.dtype) for _ in range(6)
                ]
                csdfg_bwd(output=output,
                          gK=gK,
                          gQ=gQ,
                          gV=gV,
                          gWK=gWK,
                          gWQ=gWQ,
                          gWV=gWV)
                return gK, gQ, gV, gWK, gWQ, gWV

        self.P = P
        self.H = H
        self.N = N
        emb = 4 * N

        self.op = DaceEncoderFunction
        self.WK = torch.nn.Parameter(torch.Tensor(P, H, P * H))
        self.WQ = torch.nn.Parameter(torch.Tensor(P, H, P * H))
        self.WV = torch.nn.Parameter(torch.Tensor(P, H, P * H))
        self.WO = torch.nn.Parameter(torch.Tensor(P, H, P * H))
        self.BK = torch.nn.Parameter(torch.Tensor(P, H))
        self.BQ = torch.nn.Parameter(torch.Tensor(P, H))
        self.BV = torch.nn.Parameter(torch.Tensor(P, H))
        self.BO = torch.nn.Parameter(torch.Tensor(N))
        self.norm1_scale = torch.nn.Parameter(torch.Tensor(N))
        self.norm1_bias = torch.nn.Parameter(torch.Tensor(N))
        self.norm2_scale = torch.nn.Parameter(torch.Tensor(N))
        self.norm2_bias = torch.nn.Parameter(torch.Tensor(N))
        self.linear1_w = torch.nn.Parameter(torch.Tensor(emb, N))
        self.linear1_b = torch.nn.Parameter(torch.Tensor(emb))
        self.linear2_w = torch.nn.Parameter(torch.Tensor(N, emb))
        self.linear2_b = torch.nn.Parameter(torch.Tensor(N))
        self.attn_dropout = torch.nn.Parameter(torch.Tensor(B, SM, N))
        self.linear1_dropout = torch.nn.Parameter(torch.Tensor(B, SM, emb))
        self.ff_dropout = torch.nn.Parameter(torch.Tensor(B, SM, N))
        self.scaler = torch.tensor([P**-0.5], dtype=torch_dtype)

        # Initialize parameters
        xavier_uniform_(self.WK)
        xavier_uniform_(self.WQ)
        xavier_uniform_(self.WV)
        xavier_uniform_(self.WO)
        xavier_uniform_(self.linear1_w)
        xavier_uniform_(self.linear2_w)

        # Initialized as uniform for testing
        uniform_(self.BK)
        uniform_(self.BQ)
        uniform_(self.BV)
        uniform_(self.BO)
        uniform_(self.linear1_b)
        uniform_(self.linear2_b)
        uniform_(self.norm1_scale)
        uniform_(self.norm1_bias)
        uniform_(self.norm2_scale)
        uniform_(self.norm2_bias)

        # No dropout (for reproducibility)
        ones_(self.attn_dropout)
        ones_(self.linear1_dropout)
        ones_(self.ff_dropout)

    def forward(self, x):
        result = self.op.apply(x.transpose(0, 1), self.WQ, self.WK, self.WV,
                               self.WO, self.BQ, self.BK, self.BV, self.BO,
                               self.scaler, self.norm1_scale, self.norm1_bias,
                               self.norm2_scale, self.norm2_bias,
                               self.linear1_w, self.linear1_b, self.linear2_w,
                               self.linear2_b, self.attn_dropout,
                               self.linear1_dropout, self.ff_dropout)

        # OUT [B, SN, N] -> [SN, B, N]

        return result.transpose(0, 1)


if __name__ == '__main__':
    B = 2
    H = 16
    P = 64
    N = P * H
    SM, SN = 512, 512
    X = torch.randn([SM, B, N], requires_grad=True)

    import dace.libraries.blas as blas

    # GPU
    # import dace.data
    # blas.default_implementation = 'cuBLAS'
    # sdfg = encoder.to_sdfg()
    # # Notify DaCe that all input/output arrays are allocated on the GPU
    # for arr in sdfg.arrays.values():
    #     if not arr.transient and isinstance(arr, dace.data.Array):
    #         arr.storage = dace.StorageType.GPU_Global
    # sdfg.apply_gpu_transformations()
    # sdfg.apply_strict_transformations()
    # csdfg = sdfg.compile()

    # CPU
    blas.default_implementation = 'MKL'
    csdfg = encoder.compile()

    op = DaceEncoder(N, P, H, csdfg, None)  #.cuda()
    res_dace = op.forward(X)

    t_encoder = torch.nn.TransformerEncoderLayer(N,
                                                 H,
                                                 dim_feedforward=4 * N,
                                                 dropout=0,
                                                 activation="relu")  #.cuda()
    t_encoder.self_attn.in_proj_weight.data = torch.cat(
        (op.WQ.transpose(0, 1).reshape(N, N), op.WK.transpose(0, 1).reshape(
            N, N), op.WV.transpose(0, 1).reshape(N, N)),
        dim=0)
    t_encoder.self_attn.in_proj_bias.data = torch.cat(
        (op.BQ.transpose(0, 1).reshape(N), op.BK.transpose(0, 1).reshape(N),
         op.BV.transpose(0, 1).reshape(N)))
    t_encoder.self_attn.out_proj.weight.data = op.WO.transpose(0, 2).reshape(
        N, N)
    t_encoder.self_attn.out_proj.bias.data = op.BO
    t_encoder.linear1.weight.data = op.linear1_w
    t_encoder.linear1.bias.data = op.linear1_b
    t_encoder.linear2.weight.data = op.linear2_w
    t_encoder.linear2.bias.data = op.linear2_b
    t_encoder.norm1.weight.data = op.norm1_scale
    t_encoder.norm1.bias.data = op.norm1_bias
    t_encoder.norm2.weight.data = op.norm2_scale
    t_encoder.norm2.bias.data = op.norm2_bias
    t_encoder.train()
    res_torch = t_encoder.forward(X)

    result = torch.allclose(res_dace, res_torch, rtol=1e-03, atol=1e-05)
    print('Result:', result)
    if not result:
        exit(1)
