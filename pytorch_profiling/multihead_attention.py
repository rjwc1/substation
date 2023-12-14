"""Profile multi-head attention."""

import sys
import os.path

import argparse
import torch
import torch.nn

import profiling
import math

parser = argparse.ArgumentParser(
    description='Profile multi-head attention')
parser.add_argument(
    '--batch-size', default=8, type=int,
    help='Mini-batch size (default: 8)')
parser.add_argument(
    '--max-seq-len', default=512, type=int,
    help='Maximum sequence length (default: 512)')
parser.add_argument(
    '--max-enc-seq-len', default=512, type=int,
    help='Maximum encoder sequence length (default: 512)')
parser.add_argument(
    '--embed-size', default=1024, type=int,
    help='Embedding size (default: 1024)')
parser.add_argument(
    '--num-heads', default=16, type=int,
    help='Number of attention heads (default: 16)')
parser.add_argument(
    '--attn-type', default='self', type=str,
    choices=['self', 'encdec', 'arb'],
    help='Type of attention to use (default: self)')
parser.add_argument(
    '--mask', default=False, action='store_true',
    help='Whether to use a causal mask (default: no)')
parser.add_argument(
    '--no-bias', default=False, action='store_true',
    help='Do not use bias on attention (default: use)')
parser.add_argument(
    '--no-backprop', default=False, action='store_true',
    help='Do not run backprop')
parser.add_argument(
    '--fp', default='fp32', type=str,
    choices=['fp16', 'fp32'],
    help='Precision to use for training (default: fp32)')
parser.add_argument(
    '--apex', default=False, action='store_true',
    help='Enable Nvidia Apex AMP')
parser.add_argument(
    '--num-iters', default=100, type=int,
    help='Number of benchmark iterations')
parser.add_argument(
    '--num-warmups', default=5, type=int,
    help='Number of warmup iterations')
parser.add_argument(
    '--plot-file', default=None, type=str,
    help='Save violin plots to file')


class LinformerSelfAttention(torch.nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, bias=False, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = self.default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = torch.nn.Linear(dim, dim_head * heads, bias = bias)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = torch.nn.Linear(dim, kv_dim, bias = bias)
        self.proj_k = torch.nn.Parameter(self.init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = torch.nn.Linear(dim, kv_dim, bias = bias)
            self.proj_v = torch.nn.Parameter(self.init_(torch.zeros(seq_len, k)))

        self.dropout = torch.nn.Dropout(dropout)
        self.to_out = torch.nn.Linear(dim_head * heads, dim)
    
    def default(self, val, default_val):
        return val if val is not None else default_val
    
    def init_(self, tensor):
        dim = tensor.shape[-1]
        std = 1 / math.sqrt(dim)
        tensor.uniform_(-std, std)
        return tensor

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

def time_multihead_attention(
        q, num_heads, seq_len, k=None, v=None, mask=False, mode='self',
        bias=True, do_backprop=True, fp='fp32', use_apex=False,
        num_iters=100, num_warmups=5):
    """Benchmark multi-head attention.

    q, k, v are input values in (sequence, batch, embedding) order,
    passed based on the mode.

    num_heads is the number of heads in multi-head attention.

    mask is True if doing masked multi-head attention.

    mode is one of 'self', 'encdec', or 'arb'. 'self' requires only
    q; 'encdec' needs q and k; and 'arb' needs q, k, and v.

    do_backprop is whether to benchmark backprop.

    fp is the precision to perform operations in.

    use_apex is whether to import and use Nvidia's Apex library.

    num_iters and num_warmups are the number of warmup and benchmarking
    iterations, respectively.

    Returns the runtimes for each iteration of each function.

    """
    if use_apex:
        from apex import amp
    embed_size = q.size(2)
    # attn = torch.nn.MultiheadAttention(
    #     embed_size, num_heads, bias=bias).to(profiling.cuda_device)
    attn = LinformerSelfAttention(dim=embed_size, seq_len=seq_len, heads=num_heads, bias=bias).to(profiling.cuda_device)
    attn.train()
    q = q.to(profiling.cuda_device)
    if k is not None:
        k = k.to(profiling.cuda_device)
        mask_shape = (q.size(0), k.size(0))
    else:
        mask_shape = (q.size(0), q.size(0))
    if v is not None:
        v = v.to(profiling.cuda_device)
    dy = profiling.generate_batch(
        q.size(1), q.size(0), embed_size).to(profiling.cuda_device)
    if mask:
        mask = profiling.gen_attention_mask(
            *mask_shape).to(profiling.cuda_device)
    else:
        mask = None
    if fp == 'fp16':
        if use_apex:
            attn = amp.initialize(attn)
        else:
            q = q.half()
            if k is not None:
                k = k.half()
            if v is not None:
                v = v.half()
            if mask is not None:
                mask = mask.half()
            attn = attn.half()
            dy = dy.half()
    result, backward_result = None, None
    def forward():
        nonlocal result
        if mode == 'self':
            # result = attn.forward(q, q, q, need_weights=False, attn_mask=mask)[0]
            result = attn.forward(q)[0]
        # elif mode == 'encdec':
        #     result = attn.forward(q, k, k, need_weights=False, attn_mask=mask)[0]
        # elif mode == 'arb':
        #     result = attn.forward(q, k, v, need_weights=False, attn_mask=mask)[0]
    def backward():
        nonlocal backward_result
        backward_result = result.backward(dy)
    def clear():
        attn.zero_grad()
    return profiling.time_funcs(
        [forward, backward, clear], name='MHA ' + mode,
        func_names=['forward', 'backward', 'clear'],
        num_iters=num_iters, warmups=num_warmups)


if __name__ == '__main__':
    args = parser.parse_args()
    # Check this here first.
    if args.plot_file and os.path.exists(args.plot_file):
        print(f'{args.plot_file} exists, aborting.')
        sys.exit(1)
    q = profiling.generate_batch(
        args.batch_size, args.max_seq_len, args.embed_size)
    if args.attn_type == 'self':
        k = None
        v = None
    elif args.attn_type == 'encdec':
        k = profiling.generate_batch(
            args.batch_size, args.max_enc_seq_len, args.embed_size)
        v = None
    elif args.attn_type == 'arb':
        k = profiling.generate_batch(
            args.batch_size, args.max_enc_seq_len, args.embed_size)
        v = profiling.generate_batch(
            args.batch_size, args.max_enc_seq_len, args.embed_size)
    times = time_multihead_attention(
        q, args.num_heads, seq_len = args.max_seq_len, k=k, v=v, mask=args.mask, mode=args.attn_type,
        bias=not args.no_bias, do_backprop=not args.no_backprop, fp=args.fp, use_apex=args.apex,
        num_iters=args.num_iters, num_warmups=args.num_warmups)
    print(f'batch={args.batch_size} seqlen={args.max_seq_len} enc-seqlen={args.max_enc_seq_len} embed={args.embed_size} heads={args.num_heads} mode={args.attn_type} mask={args.mask} fp={args.fp}')
    profiling.print_time_statistics(times, ['forward', 'backward', 'clear'])
    if args.plot_file:
        profiling.plot_violins(times, ['forward', 'backward', 'clear'],
                               args.plot_file)
