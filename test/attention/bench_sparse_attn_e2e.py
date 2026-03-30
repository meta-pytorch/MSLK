# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Lean end-to-end benchmark for NSA vs dense FA4 at large sequence lengths.

Only measures end-to-end Dense FA4 and NSA times (no component breakdowns)
to minimize memory usage and allow benchmarking at N >= 2M.
"""

from __future__ import annotations

import argparse
import gc

import torch


def benchmark_fn(fn, warmup=3, repeat=7):
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


def run_e2e(
    N: int,
    compress_block_size: int,
    B: int = 1,
    H: int = 8,
    H_kv: int = 2,
    D: int = 128,
    num_selected_blocks: int = 16,
    window_size: int = 512,
    dtype=torch.bfloat16,
    warmup: int = 3,
    repeat: int = 7,
    skip_dense: bool = False,
):
    """Run end-to-end Dense FA4 + NSA benchmark for one (N, l) pair."""
    from mslk.attention.sparse_attn.nsa_forward import nsa_forward
    from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func

    torch.cuda.empty_cache()
    gc.collect()

    Q = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)

    dense_ms = -1.0
    if not skip_dense:
        try:
            dense_ms = benchmark_fn(
                lambda Q=Q, K=K, V=V: flash_attn_func(Q, K, V, causal=True),
                warmup=warmup,
                repeat=repeat,
            )
        except Exception as e:
            print(f"  Dense FA4 failed: {e}")

    torch.cuda.empty_cache()

    nsa_ms = -1.0
    try:
        nsa_ms = benchmark_fn(
            lambda Q=Q, K=K, V=V: nsa_forward(
                Q,
                K,
                V,
                compress_block_size=compress_block_size,
                num_selected_blocks=num_selected_blocks,
                window_size=window_size,
                causal=True,
            ),
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as e:
        print(f"  NSA l={compress_block_size} failed: {e}")

    del Q, K, V
    torch.cuda.empty_cache()
    gc.collect()

    return dense_ms, nsa_ms


def main():
    parser = argparse.ArgumentParser(description="E2E NSA vs FA4 benchmark (lean)")
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[2097152, 4194304, 8388608, 16777216],
    )
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-heads-kv", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-selected-blocks", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument(
        "--skip-dense-above",
        type=int,
        default=0,
        help="Skip dense FA4 for N above this value (0 = never skip)",
    )
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Config: B={args.batch_size}, H={args.num_heads}, H_kv={args.num_heads_kv}, "
        f"D={args.head_dim}, k={args.num_selected_blocks}, w={args.window_size}"
    )

    # Header
    l_cols = "  ".join(f"{'l=' + str(l):>10}" for l in args.block_sizes)
    print(f"\n{'N':>10} | {'Dense(ms)':>10} | {l_cols}")
    print("-" * (14 + 13 + 12 * len(args.block_sizes)))

    for N in args.seq_lens:
        skip_dense = args.skip_dense_above > 0 and N > args.skip_dense_above

        dense_ms = None
        nsa_times = {}

        for i, l in enumerate(args.block_sizes):
            d_ms, n_ms = run_e2e(
                N=N,
                compress_block_size=l,
                B=args.batch_size,
                H=args.num_heads,
                H_kv=args.num_heads_kv,
                D=args.head_dim,
                num_selected_blocks=args.num_selected_blocks,
                window_size=args.window_size,
                warmup=args.warmup,
                repeat=args.repeat,
                skip_dense=(skip_dense or i > 0),  # Only measure dense once per N
            )
            if i == 0:
                dense_ms = d_ms
            nsa_times[l] = n_ms

        dense_str = f"{dense_ms:>10.2f}" if dense_ms and dense_ms > 0 else "     skip"
        nsa_strs = "  ".join(
            f"{nsa_times[l]:>10.2f}" if nsa_times[l] > 0 else "    failed"
            for l in args.block_sizes
        )
        print(f"{N:>10} | {dense_str} | {nsa_strs}")


if __name__ == "__main__":
    main()
