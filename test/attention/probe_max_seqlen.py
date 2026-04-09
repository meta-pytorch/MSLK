# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Quick memory probe: find the max seq len that fits on one B200 GPU."""

from __future__ import annotations

import traceback

import torch


def probe(N: int, B: int = 1, H: int = 8, H_kv: int = 2, D: int = 128):
    """Try allocating tensors and running Dense FA4 + NSA at sequence length N."""
    dtype = torch.bfloat16
    print(f"\n--- N={N:,} ({N // 1024}K) ---")

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        Q = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  Allocated Q/K/V: {alloc_gb:.1f} GB")

        # Try Dense FA4
        from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func

        O = flash_attn_func(Q, K, V, causal=True)
        torch.cuda.synchronize()
        peak_dense = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Dense FA4 OK. Peak: {peak_dense:.1f} GB")
        del O
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Try NSA
        from mslk.attention.sparse_attn.nsa_forward import nsa_forward

        O_nsa = nsa_forward(
            Q,
            K,
            V,
            compress_block_size=128,
            num_selected_blocks=16,
            window_size=512,
            causal=True,
        )
        torch.cuda.synchronize()
        peak_nsa = torch.cuda.max_memory_allocated() / 1e9
        print(f"  NSA OK. Peak: {peak_nsa:.1f} GB")
        del O_nsa, Q, K, V
        torch.cuda.empty_cache()
        return True

    except torch.cuda.OutOfMemoryError as e:
        print(f"  CUDA OOM: {e}")
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        return False


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    for n_k in [2048, 3072, 4096, 6144, 8192, 16384]:
        N = n_k * 1024
        ok = probe(N)
        if not ok:
            print(f"\n==> Max working sequence length: <{n_k}K")
            break
    else:
        print(f"\n==> All sizes up to {n_k}K fit!")


if __name__ == "__main__":
    main()
