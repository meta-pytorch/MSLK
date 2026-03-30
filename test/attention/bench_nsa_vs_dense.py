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


def benchmark_nsa_varlen(
    total_N,
    num_seqs=2,
    H=32,
    H_kv=8,
    D=128,
    backward=False,
):
    """Benchmark NSA with varlen (packed sequences)."""
    from mslk.attention.sparse_attn import nsa

    # Split total_N into num_seqs sequences with varying lengths
    # Use 60/40 split for 2 seqs, or equal for more
    if num_seqs == 2:
        seqlens = [int(total_N * 0.6), total_N - int(total_N * 0.6)]
    else:
        base = total_N // num_seqs
        seqlens = [base] * num_seqs
        seqlens[-1] = total_N - sum(seqlens[:-1])

    # Align each seqlen to q_tile_size (256)
    q_tile_size = 256
    seqlens = [max(q_tile_size, (s // q_tile_size) * q_tile_size) for s in seqlens]
    total = sum(seqlens)

    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seqlens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    Q = torch.randn(
        total, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )
    K = torch.randn(
        total, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )
    V = torch.randn(
        total, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )

    def run():
        out = nsa(
            Q,
            K,
            V,
            compress_block_size=64,
            num_selected_blocks=16,
            window_size=512,
            cu_seqlens=cu_seqlens,
        )
        if backward:
            loss = out.sum()
            loss.backward()
            Q.grad = K.grad = V.grad = None

    try:
        return _time_fn(run), seqlens
    except Exception as e:
        return f"FAILED: {e}", seqlens


def _fmt(val):
    """Format a timing value or error string."""
    if isinstance(val, str):
        return val[:30]
    return f"{val:>10.2f}ms"


def _speedup(dense, sparse):
    """Compute speedup string."""
    if isinstance(dense, str) or isinstance(sparse, str):
        return "N/A"
    return f"{dense / sparse:.2f}x"


def main():
    print("NSA vs Dense FA4 Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: B=1, H=32, H_kv=8, D=128")
    print("=" * 100)

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
    ]

    # --- Part 1: Regular (fixed-length) ---
    print("\n--- Regular (B=1, fixed-length) ---")
    print(
        f"{'N':>10} | {'Dense Fwd':>12} | {'NSA Fwd':>12} | {'Speedup':>8} | "
        f"{'Dense F+B':>12} | {'NSA F+B':>12} | {'Speedup':>8}"
    )
    print("-" * 90)

    fwd_data = {"dense": [], "nsa": []}
    fwdbwd_data = {"dense": [], "nsa": []}

    for N in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        dense_fwd = benchmark_dense_fa4(N, backward=False)
        gc.collect()
        torch.cuda.empty_cache()
        nsa_fwd = benchmark_nsa(N, backward=False)
        gc.collect()
        torch.cuda.empty_cache()

        dense_bwd = benchmark_dense_fa4(N, backward=True)
        gc.collect()
        torch.cuda.empty_cache()
        nsa_bwd = benchmark_nsa(N, backward=True)
        gc.collect()
        torch.cuda.empty_cache()

        fwd_data["dense"].append(dense_fwd)
        fwd_data["nsa"].append(nsa_fwd)
        fwdbwd_data["dense"].append(dense_bwd)
        fwdbwd_data["nsa"].append(nsa_bwd)

        print(
            f"{N:>10} | {_fmt(dense_fwd):>12} | {_fmt(nsa_fwd):>12} | "
            f"{_speedup(dense_fwd, nsa_fwd):>8} | "
            f"{_fmt(dense_bwd):>12} | {_fmt(nsa_bwd):>12} | "
            f"{_speedup(dense_bwd, nsa_bwd):>8}"
        )

    # --- Part 2: Varlen (packed sequences) ---
    print("\n--- Varlen (2 packed sequences, 60/40 split) ---")
    print(
        f"{'Total N':>10} | {'Seqlens':>20} | {'NSA Varlen F+B':>15} | "
        f"{'NSA Regular F+B':>15} | {'Ratio':>8}"
    )
    print("-" * 80)

    for N in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        varlen_result = benchmark_nsa_varlen(N, num_seqs=2, backward=True)
        if isinstance(varlen_result, tuple):
            varlen_bwd, seqlens = varlen_result
        else:
            varlen_bwd, seqlens = varlen_result, []
        gc.collect()
        torch.cuda.empty_cache()

        regular_bwd = benchmark_nsa(N, backward=True)
        gc.collect()
        torch.cuda.empty_cache()

        seqlens_s = f"{seqlens}" if seqlens else "N/A"
        ratio = (
            _speedup(regular_bwd, varlen_bwd)
            if not isinstance(varlen_bwd, str)
            else "N/A"
        )

        print(
            f"{N:>10} | {seqlens_s:>20} | {_fmt(varlen_bwd):>15} | "
            f"{_fmt(regular_bwd):>15} | {ratio:>8}"
        )

    # --- Part 3: Crossover analysis ---
    print("\n--- Crossover Analysis ---")
    for mode, data in [("Forward", fwd_data), ("Fwd+Bwd", fwdbwd_data)]:
        for i, N in enumerate(seq_lengths):
            d, s = data["dense"][i], data["nsa"][i]
            if isinstance(d, (int, float)) and isinstance(s, (int, float)):
                if d > s:
                    print(f"{mode}: NSA becomes faster at N={N} ({d / s:.2f}x speedup)")
                    break
        else:
            print(f"{mode}: NSA never becomes faster in tested range")


if __name__ == "__main__":
    main()
