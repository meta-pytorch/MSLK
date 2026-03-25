# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Detailed benchmarks for NSA vs dense FA4 attention.

Reports wall-clock time, effective TFLOPS, and speedup ratio.
Separately times each NSA component: compression, selection, attention branches, gating.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch


def benchmark_fn(fn, warmup=5, repeat=20):
    """Benchmark a CUDA function using CUDA events. Returns median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = sorted(times)
    return times[len(times) // 2]


def compute_flops_dense(B, H, N, D):
    """Compute FLOPs for dense attention forward pass."""
    # QK^T: 2*B*H*N*N*D, PV: 2*B*H*N*N*D
    return 4 * B * H * N * N * D


def compute_flops_nsa(B, H, N, D, compress_block_size, num_selected, window_size):
    """Compute FLOPs for NSA forward pass."""
    l = compress_block_size
    k = num_selected
    w = window_size
    # Compressed: 4*B*H*N*(N/l)*D
    # Selected: 4*B*H*N*(k*l)*D
    # Sliding: 4*B*H*N*w*D
    flops_cmp = 4 * B * H * N * (N // l) * D
    flops_slc = 4 * B * H * N * (k * l) * D
    flops_sld = 4 * B * H * N * w * D
    return flops_cmp + flops_slc + flops_sld


def run_benchmark(
    N: int,
    B: int = 2,
    H: int = 32,
    H_kv: int = 8,
    D: int = 128,
    compress_block_size: int = 64,
    num_selected_blocks: int = 16,
    window_size: int = 512,
    dtype=torch.bfloat16,
    warmup: int = 5,
    repeat: int = 20,
):
    """Run NSA vs dense FA4 benchmark for a given sequence length."""
    from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func
    from mslk.attention.sparse_attn.compress import compress_kv
    from mslk.attention.sparse_attn.gating import compute_gates, gate_and_combine
    from mslk.attention.sparse_attn.nsa_forward import nsa_forward
    from mslk.attention.sparse_attn.select import score_and_select_blocks
    from mslk.attention.sparse_attn.sparsity_masks import build_fa4_block_sparse_tensors

    Q = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)

    # --- Dense FA4 baseline ---
    dense_ms = benchmark_fn(
        lambda: flash_attn_func(Q, K, V, causal=True),
        warmup=warmup, repeat=repeat,
    )
    dense_flops = compute_flops_dense(B, H, N, D)
    dense_tflops = dense_flops / (dense_ms * 1e-3) / 1e12

    # --- NSA end-to-end ---
    nsa_ms = benchmark_fn(
        lambda: nsa_forward(
            Q, K, V,
            compress_block_size=compress_block_size,
            num_selected_blocks=num_selected_blocks,
            window_size=window_size,
            causal=True,
        ),
        warmup=warmup, repeat=repeat,
    )
    nsa_flops = compute_flops_nsa(
        B, H, N, D, compress_block_size, num_selected_blocks, window_size
    )
    nsa_tflops = nsa_flops / (nsa_ms * 1e-3) / 1e12

    # --- Component timing ---
    # Compression
    cmp_ms = benchmark_fn(
        lambda: compress_kv(K, V, compress_block_size),
        warmup=warmup, repeat=repeat,
    )

    # Selection
    K_cmp, V_cmp = compress_kv(K, V, compress_block_size)
    sel_ms = benchmark_fn(
        lambda: score_and_select_blocks(
            Q, K_cmp, num_selected_blocks, compress_block_size,
            causal=True, q_tile_size=256,
        ),
        warmup=warmup, repeat=repeat,
    )

    # Block-sparse mask construction
    block_indices = score_and_select_blocks(
        Q, K_cmp, num_selected_blocks, compress_block_size,
        causal=True, q_tile_size=256,
    )
    mask_ms = benchmark_fn(
        lambda: build_fa4_block_sparse_tensors(
            block_indices, compress_block_size, n_block_size=128, seqlen_k=N,
        ),
        warmup=warmup, repeat=repeat,
    )

    # Compressed attention FA4
    fa4_cmp_ms = benchmark_fn(
        lambda: flash_attn_func(Q, K_cmp, V_cmp, causal=True),
        warmup=warmup, repeat=repeat,
    )

    # Selected attention FA4 (block-sparse)
    sparse_tensors = build_fa4_block_sparse_tensors(
        block_indices, compress_block_size, n_block_size=128, seqlen_k=N,
    )
    from mslk.fb.mslk.attention.flash_attn.interface import _flash_attn_fwd
    fa4_slc_ms = benchmark_fn(
        lambda: _flash_attn_fwd(
            Q, K, V, causal=True, block_sparse_tensors=sparse_tensors,
        ),
        warmup=warmup, repeat=repeat,
    )

    # Sliding window FA4
    fa4_sld_ms = benchmark_fn(
        lambda: flash_attn_func(Q, K, V, causal=True, window_size=(window_size, 0)),
        warmup=warmup, repeat=repeat,
    )

    # Gating
    O_cmp = torch.randn_like(Q)
    O_slc = torch.randn_like(Q)
    O_sld = torch.randn_like(Q)
    gates = torch.randn(B, N, H, 3, device="cuda", dtype=dtype)
    gate_ms = benchmark_fn(
        lambda: gate_and_combine(O_cmp, O_slc, O_sld, gates),
        warmup=warmup, repeat=repeat,
    )

    speedup = dense_ms / nsa_ms
    theoretical_speedup = N / (N / compress_block_size + num_selected_blocks * compress_block_size + window_size)

    return {
        "N": N,
        "dense_ms": dense_ms,
        "dense_tflops": dense_tflops,
        "nsa_ms": nsa_ms,
        "nsa_tflops": nsa_tflops,
        "speedup": speedup,
        "theoretical_speedup": theoretical_speedup,
        "compress_ms": cmp_ms,
        "select_ms": sel_ms,
        "mask_ms": mask_ms,
        "fa4_cmp_ms": fa4_cmp_ms,
        "fa4_slc_ms": fa4_slc_ms,
        "fa4_sld_ms": fa4_sld_ms,
        "gate_ms": gate_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="NSA vs FA4 benchmark")
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-heads-kv", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--compress-block-size", type=int, default=64)
    parser.add_argument("--num-selected-blocks", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    print(f"{'='*100}")
    print(f"NSA vs Dense FA4 Benchmark")
    print(f"Config: B={args.batch_size}, H={args.num_heads}, H_kv={args.num_heads_kv}, "
          f"D={args.head_dim}, l={args.compress_block_size}, k={args.num_selected_blocks}, "
          f"w={args.window_size}")
    print(f"{'='*100}")

    header = (
        f"{'N':>8} | {'Dense(ms)':>10} {'TFLOPS':>8} | "
        f"{'NSA(ms)':>10} {'TFLOPS':>8} | {'Speedup':>8} {'Theory':>8} | "
        f"{'Cmp':>6} {'Sel':>6} {'Mask':>6} "
        f"{'FA4c':>6} {'FA4s':>6} {'FA4w':>6} {'Gate':>6}"
    )
    print(header)
    print("-" * len(header))

    for N in args.seq_lens:
        result = run_benchmark(
            N=N,
            B=args.batch_size,
            H=args.num_heads,
            H_kv=args.num_heads_kv,
            D=args.head_dim,
            compress_block_size=args.compress_block_size,
            num_selected_blocks=args.num_selected_blocks,
            window_size=args.window_size,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print(
            f"{result['N']:>8} | "
            f"{result['dense_ms']:>10.2f} {result['dense_tflops']:>8.1f} | "
            f"{result['nsa_ms']:>10.2f} {result['nsa_tflops']:>8.1f} | "
            f"{result['speedup']:>8.2f}x {result['theoretical_speedup']:>7.1f}x | "
            f"{result['compress_ms']:>6.2f} {result['select_ms']:>6.2f} "
            f"{result['mask_ms']:>6.2f} {result['fa4_cmp_ms']:>6.2f} "
            f"{result['fa4_slc_ms']:>6.2f} {result['fa4_sld_ms']:>6.2f} "
            f"{result['gate_ms']:>6.2f}"
        )

    print(f"{'='*100}")


if __name__ == "__main__":
    main()
