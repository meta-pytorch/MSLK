# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Quick benchmark: NSA with sparse compressed branch vs original."""

import gc
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault(
    "CUTLASS_CUTE_DSL_KERNEL_CACHE_DIR",
    os.path.expanduser("~/.cache/cutlass_dsl_kernels"),
)

import torch


def _time_fn(fn, warmup=3, iters=5):
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


def main():
    from mslk.attention.sparse_attn import nsa_forward
    from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func

    B, H, H_kv, D = 1, 32, 8, 128

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: B={B}, H={H}, H_kv={H_kv}, D={D}")
    print()

    # Warmup JIT
    print("Warming up JIT compilation...")
    Q = torch.randn(B, 4096, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, 4096, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, 4096, H_kv, D, device="cuda", dtype=torch.bfloat16)
    nsa_forward(Q, K, V)
    nsa_forward(Q, K, V, num_cmp_selected_blocks=16)
    del Q, K, V
    gc.collect()
    torch.cuda.empty_cache()
    print("JIT warmup done.\n")

    header = (
        f"{'N':>8} | {'Dense FA4':>10} | {'NSA orig':>10} | "
        f"{'NSA+CmpSp':>10} | {'Orig sp':>8} | {'CmpSp sp':>8} | {'Improvement':>11}"
    )
    print(header)
    print("-" * len(header))

    seq_lengths = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

    for N in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        # Dense FA4
        try:
            dense = _time_fn(
                lambda: flash_attn_func(
                    Q, K, V, softmax_scale=1.0 / D**0.5, causal=True
                )
            )
        except Exception:
            dense = float("inf")

        gc.collect()
        torch.cuda.empty_cache()

        # NSA original (no sparse compressed)
        try:
            nsa_orig = _time_fn(lambda: nsa_forward(Q, K, V))
        except Exception:
            nsa_orig = float("inf")

        gc.collect()
        torch.cuda.empty_cache()

        # NSA with sparse compressed branch (k_cmp=16 FA4 blocks)
        try:
            nsa_sp = _time_fn(lambda: nsa_forward(Q, K, V, num_cmp_selected_blocks=16))
        except Exception:
            nsa_sp = float("inf")

        del Q, K, V

        sp_orig = dense / nsa_orig if nsa_orig < float("inf") else 0
        sp_new = dense / nsa_sp if nsa_sp < float("inf") else 0
        improvement = nsa_orig / nsa_sp if nsa_sp < float("inf") else 0

        print(
            f"{N:>8} | {dense:>9.2f}ms | {nsa_orig:>9.2f}ms | "
            f"{nsa_sp:>9.2f}ms | {sp_orig:>7.2f}x | {sp_new:>7.2f}x | {improvement:>10.2f}x"
        )


if __name__ == "__main__":
    main()
