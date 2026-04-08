# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""NSA vs Dense FA4 benchmark — forward and backward at various sequence lengths."""

import gc
import time

import torch


def _time_fn(fn, warmup=3, iters=5):
    """Time a function with warmup, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2]


def benchmark_dense_fa4(N, B=1, H=32, H_kv=8, D=128, backward=False):
    """Benchmark dense causal FA4 attention."""
    from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func

    Q = torch.randn(
        B, N, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )
    K = torch.randn(
        B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )
    V = torch.randn(
        B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )

    def run():
        out, _lse = flash_attn_func(Q, K, V, softmax_scale=1.0 / D**0.5, causal=True)
        if backward:
            loss = out.sum()
            loss.backward()
            Q.grad = K.grad = V.grad = None

    try:
        return _time_fn(run)
    except Exception as e:
        return f"FAILED: {e}"


def benchmark_nsa(N, B=1, H=32, H_kv=8, D=128, backward=False):
    """Benchmark NSA sparse attention."""
    from mslk.attention.sparse_attn import nsa

    Q = torch.randn(
        B, N, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )
    K = torch.randn(
        B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )
    V = torch.randn(
        B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )

    def run():
        out = nsa(
            Q,
            K,
            V,
            compress_block_size=64,
            num_selected_blocks=16,
            window_size=512,
        )
        if backward:
            loss = out.sum()
            loss.backward()
            Q.grad = K.grad = V.grad = None

    try:
        return _time_fn(run)
    except Exception as e:
        return f"FAILED: {e}"


def main():
    print("NSA vs Dense FA4 Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: B=1, H=32, H_kv=8, D=128")
    print("=" * 80)

    # Sequence lengths: 1K to 3M (powers of 2, plus some intermediate points)
    seq_lengths = [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
        3145728,
    ]

    print(
        f"\n{'N':>10} | {'Dense Fwd':>12} | {'NSA Fwd':>12} | {'Speedup':>8} | "
        f"{'Dense Bwd':>12} | {'NSA Bwd':>12} | {'Speedup':>8}"
    )
    print("-" * 90)

    for N in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        # Forward
        dense_fwd = benchmark_dense_fa4(N, backward=False)
        gc.collect()
        torch.cuda.empty_cache()

        nsa_fwd = benchmark_nsa(N, backward=False)
        gc.collect()
        torch.cuda.empty_cache()

        if isinstance(dense_fwd, str) or isinstance(nsa_fwd, str):
            fwd_speedup = "N/A"
            dense_fwd_s = (
                str(dense_fwd)
                if isinstance(dense_fwd, str)
                else f"{dense_fwd:>10.2f}ms"
            )
            nsa_fwd_s = (
                str(nsa_fwd) if isinstance(nsa_fwd, str) else f"{nsa_fwd:>10.2f}ms"
            )
        else:
            fwd_speedup = f"{dense_fwd / nsa_fwd:.2f}x"
            dense_fwd_s = f"{dense_fwd:>10.2f}ms"
            nsa_fwd_s = f"{nsa_fwd:>10.2f}ms"

        # Backward
        dense_bwd = benchmark_dense_fa4(N, backward=True)
        gc.collect()
        torch.cuda.empty_cache()

        nsa_bwd = benchmark_nsa(N, backward=True)
        gc.collect()
        torch.cuda.empty_cache()

        if isinstance(dense_bwd, str) or isinstance(nsa_bwd, str):
            bwd_speedup = "N/A"
            dense_bwd_s = (
                str(dense_bwd)[:30]
                if isinstance(dense_bwd, str)
                else f"{dense_bwd:>10.2f}ms"
            )
            nsa_bwd_s = (
                str(nsa_bwd)[:30] if isinstance(nsa_bwd, str) else f"{nsa_bwd:>10.2f}ms"
            )
        else:
            bwd_speedup = f"{dense_bwd / nsa_bwd:.2f}x"
            dense_bwd_s = f"{dense_bwd:>10.2f}ms"
            nsa_bwd_s = f"{nsa_bwd:>10.2f}ms"

        print(
            f"{N:>10} | {dense_fwd_s:>12} | {nsa_fwd_s:>12} | {fwd_speedup:>8} | "
            f"{dense_bwd_s:>12} | {nsa_bwd_s:>12} | {bwd_speedup:>8}"
        )


if __name__ == "__main__":
    main()
