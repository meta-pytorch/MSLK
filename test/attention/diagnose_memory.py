#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Diagnose GPU memory usage for NSA at large sequence lengths."""

import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault(
    "CUTLASS_CUTE_DSL_KERNEL_CACHE_DIR",
    os.path.expanduser("~/.cache/cutlass_dsl_kernels"),
)

import torch


def mem_mb():
    return torch.cuda.memory_allocated() / 1024**2


def mem_reserved_mb():
    return torch.cuda.memory_reserved() / 1024**2


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total GPU memory: {total:.1f} GiB")
    print()

    # Warmup at small N to compile all kernels
    print("=== Warmup (N=4096) ===")
    from mslk.attention.sparse_attn import nsa, nsa_forward

    B, H, H_kv, D = 1, 32, 8, 128
    N_warmup = 4096

    print(
        f"Before import: {mem_mb():.0f} MB allocated, {mem_reserved_mb():.0f} MB reserved"
    )

    Q = torch.randn(B, N_warmup, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, N_warmup, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N_warmup, H_kv, D, device="cuda", dtype=torch.bfloat16)

    print(
        f"After tensor alloc: {mem_mb():.0f} MB allocated, {mem_reserved_mb():.0f} MB reserved"
    )

    # Forward warmup
    O = nsa_forward(Q, K, V)
    torch.cuda.synchronize()
    print(
        f"After fwd warmup 1: {mem_mb():.0f} MB allocated, {mem_reserved_mb():.0f} MB reserved"
    )

    del O
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"After cleanup: {mem_mb():.0f} MB allocated, {mem_reserved_mb():.0f} MB reserved"
    )

    # Forward + backward warmup
    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    O = nsa(Q, K, V)
    O.sum().backward()
    torch.cuda.synchronize()
    print(
        f"After fwd+bwd warmup: {mem_mb():.0f} MB allocated, {mem_reserved_mb():.0f} MB reserved"
    )

    Q.grad = K.grad = V.grad = None
    del O
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"After cleanup: {mem_mb():.0f} MB allocated, {mem_reserved_mb():.0f} MB reserved"
    )

    # Now try larger N
    for N in [131072, 262144, 524288, 1048576, 2097152]:
        gc.collect()
        torch.cuda.empty_cache()

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        tensor_mb = 3 * B * N * max(H, H_kv) * D * 2 / 1024**2  # approx
        print(f"\n=== N={N} ({N // 1024}K) ===")
        print(f"Tensor size: ~{tensor_mb:.0f} MB")
        print(
            f"Before: {mem_mb():.0f} MB allocated, {mem_reserved_mb():.0f} MB reserved"
        )

        try:
            O = nsa_forward(Q, K, V)
            torch.cuda.synchronize()
            print(
                f"Forward OK: {mem_mb():.0f} MB allocated, {mem_reserved_mb():.0f} MB reserved"
            )
            del O
        except Exception as e:
            print(f"Forward FAILED: {e}")

        del Q, K, V
        gc.collect()
        torch.cuda.empty_cache()

        # Try fwd+bwd
        try:
            Q = torch.randn(
                B, N, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            K = torch.randn(
                B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            V = torch.randn(
                B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )

            O = nsa(Q, K, V)
            O.sum().backward()
            torch.cuda.synchronize()
            print(
                f"Fwd+Bwd OK: {mem_mb():.0f} MB allocated, {mem_reserved_mb():.0f} MB reserved"
            )
            Q.grad = K.grad = V.grad = None
            del O, Q, K, V
        except Exception as e:
            print(f"Fwd+Bwd FAILED: {e}")

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
