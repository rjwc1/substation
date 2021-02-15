from dace.dtypes import typeclass
import dace
import numpy as np
import math
import os

# Imported to register transformation
from substation.xforms.merge_source_sink import MergeSourceSinkArrays

from substation.attention import attn_forward_sdfg
from substation.dtypes import *

B, H, N, P, SM, SN = (dace.symbol(s) for s in ['B', 'H', 'N', 'P', 'SM', 'SN'])
emb = dace.symbol('emb')
eps = 1e-5


@dace.program
def relu(x):
    return np.maximum(x, 0)


@dace.program
def gelu(x: dace_dtype[B, SM, N]):
    """Gaussian Error Linear Unit applied to x."""
    out = np.ndarray(x.shape, x.dtype)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            inp << x[i, j, k]
            outp >> out[i, j, k]
            outp = 0.5 * inp * (1 + math.tanh(
                math.sqrt(2.0 / math.pi) * (inp + 0.044715 * (inp**3))))
    return out


@dace.program
def linear_with_bias(x: dace_dtype[B, SM, N], w: dace_dtype[emb, N],
                     bias: dace_dtype[emb]):
    """Fully-connected layer with weights w and bias."""
    return np.einsum('bik,jk->bij', x, w) + bias


@dace.program
def meanstd(x: dace_dtype[B, SM, N]):
    mean = np.ndarray([B, SM], x.dtype)
    std = np.ndarray([B, SM], x.dtype)

    moment = dace.reduce(lambda a, b: a + b, x, axis=2, identity=0)
    second_moment = dace.reduce(lambda a, b: a + b * b, x, axis=2, identity=0)

    for i, j in dace.map[0:B, 0:SM]:
        with dace.tasklet:
            fmom << moment[i, j]
            smom << second_moment[i, j]
            mn >> mean[i, j]
            st >> std[i, j]
            mn = fmom / (SM * B)
            st = math.sqrt((smom / (SM*B)) - mn*mn)

    return mean, std


@dace.program
def layer_norm(x: dace_dtype[B, SM, N]):
    """ Apply layer normalization to x. """
    out = np.ndarray(x.shape, x.dtype)
    mean, std = meanstd(x)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            in_x << x[i, j, k]
            in_m << mean[i, j]
            in_s << std[i, j]
            o >> out[i, j, k]
            o = (in_x - in_m) / (in_s + eps)
    return out


@dace.program
def layer_norm_scaled(x: dace_dtype[B, SM, N], scale: dace_dtype[N],
                      bias: dace_dtype[N]):
    """Apply layer normalization to x, with scale and bias.

    scale and bias are the same shape as the final axis.

    """
    out = np.ndarray(x.shape, x.dtype)
    mean, std = meanstd(x)
    for i, j, k in dace.map[0:B, 0:SM, 0:N]:
        with dace.tasklet:
            in_scal << scale[k]
            in_bias << bias[k]
            in_x << x[i, j, k]
            in_m << mean[i, j]
            in_s << std[i, j]
            o >> out[i, j, k]
            o = in_scal * ((in_x - in_m) / (in_s + eps)) + in_bias
    return out


@dace.program
def dropout(x: dace_dtype[B, SM, N], mask: dace_dtype[B, SM, N]):
    """Apply dropout with pre-randomized dropout mask."""
    return x * mask


@dace.program
def softmax(x: dace_dtype[B, H, SM, SN]):
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True, initial=0)
    return tmp_out / tmp_sum


@dace.program
def mha_forward(
    q: dace_dtype[B, SN, N],
    k: dace_dtype[B, SM, N],
    v: dace_dtype[B, SM, N],
    wq: dace_dtype[P, H, N],
    wk: dace_dtype[P, H, N],
    wv: dace_dtype[P, H, N],
    wo: dace_dtype[P, H, N],
    bq: dace_dtype[P, H, 1, 1],
    bk: dace_dtype[P, H, 1, 1],
    bv: dace_dtype[P, H, 1, 1],
    bo: dace_dtype[N],
    scaler: dace_dtype,
):
    qq = np.einsum("phi,bji->phbj", wq, q) + bq
    kk = np.einsum("phi,bki->phbk", wk, k) + bk
    vv = np.einsum("phi,bki->phbk", wv, v) + bv
    beta = scaler * np.einsum("phbk,phbj->hbjk", kk, qq)
    alpha = softmax(beta)
    gamma = np.einsum("phbk,hbjk->phbj", vv, alpha)
    out = np.einsum("phi,phbj->bji", wo, gamma) + bo
    return out


activation = relu
# activation = gelu


@dace.program
def encoder(x: dace_dtype[B, SM,
                          N], attn_wq: dace_dtype[P, H,
                                                  N], attn_wk: dace_dtype[P, H,
                                                                          N],
            attn_wv: dace_dtype[P, H, N], attn_wo: dace_dtype[P, H, N],
            attn_bq: dace_dtype[P, H, 1, 1], attn_bk: dace_dtype[P, H, 1, 1],
            attn_bv: dace_dtype[P, H, 1, 1], attn_bo: dace_dtype[N],
            attn_scale: dace_dtype, norm1_scale: dace_dtype[N],
            norm1_bias: dace_dtype[N], norm2_scale: dace_dtype[N],
            norm2_bias: dace_dtype[N], linear1_w: dace_dtype[emb, N],
            linear1_b: dace_dtype[emb], linear2_w: dace_dtype[N, emb],
            linear2_b: dace_dtype[N], attn_dropout: dace_dtype[B, SM, N],
            linear1_dropout: dace_dtype[B, SM,
                                        emb], ff_dropout: dace_dtype[B, SM, N]):

    # Self-attention.
    attn = mha_forward(x, x, x, attn_wq, attn_wk, attn_wv, attn_wo, attn_bq,
                       attn_bk, attn_bv, attn_bo, attn_scale)

    # Residual connection.
    attn_resid = dropout(attn, attn_dropout) + x

    normed1 = layer_norm_scaled(attn_resid, norm1_scale, norm1_bias)

    # Feedforward network.
    ff = linear_with_bias(
        dropout(activation(linear_with_bias(normed1, linear1_w, linear1_b)),
                linear1_dropout), linear2_w, linear2_b)

    # Residual connection.
    ff_resid = dropout(ff, ff_dropout) + normed1
    normed2 = layer_norm_scaled(ff_resid, norm2_scale, norm2_bias)
    return normed2


@dace.program
def decoder(x: dace_dtype[B, SM, N], attn_wq: dace_dtype[P, H, N],
            attn_wk: dace_dtype[P, H, N], attn_wv: dace_dtype[P, H, N],
            attn_wo: dace_dtype[P, H, N], attn_scale: dace_dtype,
            attn_mask: dace_dtype[SM, SM], norm1_scale: dace_dtype[N],
            norm1_bias: dace_dtype[N], norm2_scale: dace_dtype[N],
            norm2_bias: dace_dtype[N], linear1_w: dace_dtype[emb, N],
            linear1_b: dace_dtype[emb], linear2_w: dace_dtype[N, emb],
            linear2_b: dace_dtype[N], attn_dropout: dace_dtype[B, SM, N],
            linear1_dropout: dace_dtype[B, SM,
                                        emb], ff_dropout: dace_dtype[B, SM, N]):
    # Masked self-attention.
    attn = np.ndarray(x.shape, x.dtype)
    attn_forward_mask(Q=x,
                      K=x,
                      V=x,
                      WQ=attn_wq,
                      WK=attn_wk,
                      WV=attn_wv,
                      WO=attn_wo,
                      scaler=attn_scale,
                      OUT=attn,
                      MASK=attn_mask,
                      B=B,
                      H=H,
                      N=N,
                      P=P,
                      SM=SM,
                      SN=SM)

    # Residual connection.
    attn_resid = dropout(attn, attn_dropout) + x
    normed1 = layer_norm(attn_resid, norm1_scale, norm1_bias)
    # Feedforward network.
    ff = linear_with_bias(
        dropout(gelu(linear_with_bias(normed1, linear1_w, linear1_b)),
                linear1_dropout), linear2_w, linear2_b)
    # Residual connection.
    ff_resid = dropout(ff, ff_dropout) + normed1
    normed2 = layer_norm(ff_resid, norm2_scale, norm2_bias)
    return normed2


@dace.program
def dec_with_enc_attn(
    x: dace_dtype[B, SN, N], encoder_out: dace_dtype[B, SM, N],
    sattn_wq: dace_dtype[P, H, N], sattn_wk: dace_dtype[P, H, N],
    sattn_wv: dace_dtype[P, H,
                         N], sattn_wo: dace_dtype[P, H,
                                                  N], sattn_scale: dace_dtype,
    sattn_mask: dace_dtype[SN, SN], edattn_wq: dace_dtype[P, H, N],
    edattn_wk: dace_dtype[P, H, N], edattn_wv: dace_dtype[P, H, N],
    edattn_wo: dace_dtype[P, H, N], edattn_scale: dace_dtype,
    norm1_scale: dace_dtype[N], norm1_bias: dace_dtype[N],
    norm2_scale: dace_dtype[N], norm2_bias: dace_dtype[N],
    norm3_scale: dace_dtype[N], norm3_bias: dace_dtype[N],
    linear1_w: dace_dtype[emb, N], linear1_b: dace_dtype[emb],
    linear2_w: dace_dtype[N, emb], linear2_b: dace_dtype[N],
    sattn_dropout: dace_dtype[B, SN, N], edattn_dropout: dace_dtype[B, SN, N],
    linear1_dropout: dace_dtype[B, SN, emb], ff_dropout: dace_dtype[B, SN, N]):
    # Masked self-attention.
    sattn = np.ndarray(x.shape, x.dtype)
    attn_forward_mask(Q=x,
                      K=x,
                      V=x,
                      WQ=sattn_wq,
                      WK=sattn_wk,
                      WV=sattn_wv,
                      WO=sattn_wo,
                      scaler=sattn_scale,
                      OUT=sattn,
                      MASK=sattn_mask,
                      B=B,
                      H=H,
                      N=N,
                      P=P,
                      SM=SN,
                      SN=SN)
    # Residual connection.
    sattn_resid = dropout(sattn, sattn_dropout) + x
    normed1 = layer_norm(sattn_resid, norm1_scale, norm1_bias)

    # Encoder-decoder attention.
    edattn = np.ndarray(normed1.shape, dace.float32)
    attn_forward(Q=normed1,
                 K=encoder_out,
                 V=encoder_out,
                 WQ=edattn_wq,
                 WK=edattn_wk,
                 WV=edattn_wv,
                 WO=edattn_wo,
                 scaler=edattn_scale,
                 OUT=edattn,
                 B=B,
                 H=H,
                 N=N,
                 P=P,
                 SM=SM,
                 SN=SN)

    # Residual connection.
    edattn_resid = dropout(edattn, edattn_dropout) + normed1
    normed2 = layer_norm(edattn_resid, norm2_scale, norm2_bias)
    # Feedforward network.
    ff = linear_with_bias(
        dropout(gelu(linear_with_bias(normed2, linear1_w, linear1_b)),
                linear1_dropout), linear2_w, linear2_b)
    # Residual connection.
    ff_resid = dropout(ff, ff_dropout) + normed2
    normed3 = layer_norm(ff_resid, norm3_scale, norm3_bias)
    return normed3


if __name__ == '__main__':
    B = 1  #2
    H = 8  #16
    P = 16  #64
    N = P * H
    SM, SN = 32, 32  #512, 512
    emb = 4 * N
    sizes = dict(B=B, H=H, P=P, N=N, SM=SM, emb=emb)
    from dace.transformation.dataflow import MapFusion
    from dace.transformation.interstate import StateFusion

    # softmax.to_sdfg().save('softmax.sdfg')

    # sdfg = mha_forward.to_sdfg()
    # #sdfg.apply_transformations_repeated([StateFusion])
    # sdfg.save('mha3.sdfg')

    from typing import Set

    def find_new_name(sdfg: dace.SDFG, name: str) -> str:
        """ Tries to find a new name by adding an underscore and a number. """
        index = 0
        taken = set(sdfg.arrays.keys() | sdfg.symbols.keys()
                    | sdfg.constants.keys())
        while (name + ('_%d' % index)) in taken:
            index += 1

        return name + ('_%d' % index)

    def unify_symbols(sdfg: dace.SDFG):
        """ Uses one set of symbols across all nested SDFGs. """
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    # First, get nested symbols and replace them if they match
                    # the names of the outer symbols
                    usedsyms: Set[str] = set()
                    for symvalue in node.symbol_mapping.values():
                        usedsyms |= set(
                            map(
                                str,
                                dace.symbolic.pystr_to_symbolic(
                                    symvalue).free_symbols))

                    # Replace clashing names
                    clashing = usedsyms & (node.sdfg.symbols.keys()
                                           | node.sdfg.arrays.keys())
                    for clash in clashing:
                        new_name = find_new_name(node.sdfg, clash)
                        node.sdfg.replace(clash, new_name)
                        if clash in node.symbol_mapping:
                            node.symbol_mapping[new_name] = node.symbol_mapping[
                                clash]
                            del node.symbol_mapping[clash]

                    # Two-step replacement (N -> __dacesym_N --> map[N]) to avoid clashes
                    for symname, symvalue in node.symbol_mapping.items():
                        if str(symname) != str(symvalue):
                            node.sdfg.replace(symname, '__dacesym_' + symname)
                    for symname, symvalue in node.symbol_mapping.items():
                        if str(symname) != str(symvalue):
                            if str(symvalue) in node.sdfg.symbols:
                                del node.sdfg.symbols[str(symvalue)]
                            node.sdfg.replace('__dacesym_' + symname,
                                              str(symvalue))

                    # Replace symbol mapping
                    node.symbol_mapping = {k: k for k in usedsyms}

                    # Recursively descend
                    unify_symbols(node.sdfg)

    esdfg = encoder.to_sdfg()  #strict=False)
    esdfg.save('encoder.sdfg')

    #esdfg = dace.SDFG.from_file('encoder.sdfg')
    #esdfg.expand_library_nodes()
    # unify_symbols(esdfg)
    # esdfg.save('encoder.sdfg')

    # Create sample data
    def deal_with_one(desc):
        if isinstance(desc, typeclass):
            return np.random.rand()
        shape = [dace.symbolic.evaluate(s, sizes) for s in desc.shape]
        return np.random.rand(*shape).astype(np.float32)

    arrays = {k: deal_with_one(v) for k, v in encoder.f.__annotations__.items()}
    esdfg(**arrays, **sizes)

    import torch

    #esdfg.apply_transformations_repeated([StateFusion, MergeSourceSinkArrays])
    #esdfg.apply_strict_transformations()
    #esdfg.apply_transformations_repeated(MapFusion)

    # dsdfg = decoder.to_sdfg()
    # dsdfg.apply_strict_transformations()
    # dsdfg.apply_transformations_repeated(MapFusion)
    # dsdfg.save('decoder-nonstrict.sdfg')

    # desdfg = dec_with_enc_attn.to_sdfg()
    # desdfg.apply_strict_transformations()
    # desdfg.apply_transformations_repeated(MapFusion)
    # desdfg.save('decoder_encattn.sdfg')

    # # Remove duplicate CUBLAS creation code. TODO: Use library nodes instead
    # cublas_found = False
    # for node, parent in desdfg.all_nodes_recursive():
    #     if isinstance(node, dace.nodes.Tasklet):
    #         if 'cublasHandle_t' in node.code_global:
    #             if cublas_found:
    #                 node.code_global = ''
    #                 node.code_init = ''
    #                 node.code_exit = ''
    #             cublas_found = True

    # dsdfg.compile(optimizer=False)
    # desdfg.compile(optimizer=False)
