# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Quick benchmark: test NSA and Dense FA4 at 2M and 3M context lengths."""

import gc
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault(
    "CUTLASS_CUTE_DSL_KERNEL_CACHE_DIR",
    os.path.expanduser("~/.cache/cutlass_dsl_kernels"),
)

import torch


def _time_fn(fn, warmup=2, iters=3):
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


def _fmt(val):
    if isinstance(val, str):
        return val
    return f"{val:.2f}ms"


def _speedup(dense, sparse):
    if isinstance(dense, str) or isinstance(sparse, str):
        return "N/A"
    return f"{dense / sparse:.2f}x"


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {mem_gb:.1f} GB")
    print(f"Config: B=1, H=32, H_kv=8, D=128")
    print("=" * 90)

    seq_lengths = [
        2097152,  # 2M
        3145728,  # 3M
    ]

    for N in seq_lengths:
        label = f"{N // (1024 * 1024)}M"
        print(f"\n--- N = {N:,} ({label}) ---")

        # Forward only
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Dense FA4 forward ... ", end="", flush=True)
        dense_fwd = benchmark_dense_fa4(N, backward=False)
        print(_fmt(dense_fwd))

        gc.collect()
        torch.cuda.empty_cache()
        print(f"  NSA forward ... ", end="", flush=True)
        nsa_fwd = benchmark_nsa(N, backward=False)
        print(_fmt(nsa_fwd))

        print(f"  Forward speedup: {_speedup(dense_fwd, nsa_fwd)}")

        # Forward + Backward
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Dense FA4 fwd+bwd ... ", end="", flush=True)
        dense_fwdbwd = benchmark_dense_fa4(N, backward=True)
        print(_fmt(dense_fwdbwd))

        gc.collect()
        torch.cuda.empty_cache()
        print(f"  NSA fwd+bwd ... ", end="", flush=True)
        nsa_fwdbwd = benchmark_nsa(N, backward=True)
        print(_fmt(nsa_fwdbwd))

        print(f"  Fwd+Bwd speedup: {_speedup(dense_fwdbwd, nsa_fwdbwd)}")


if __name__ == "__main__":
    main()
