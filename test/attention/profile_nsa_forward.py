# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Profile NSA forward components to find optimization targets."""

import gc
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


def time_fn(fn, warmup=3, iters=10):
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


def profile_nsa_forward(N, B=1, H=32, H_kv=8, D=128):
    from mslk.attention.sparse_attn.compress import fused_compress_kv
    from mslk.attention.sparse_attn.gating import fused_gate_and_combine
    from mslk.attention.sparse_attn.nsa_forward import (
        _fa4_fwd,
        _make_compressed_causal_mask,
    )
    from mslk.attention.sparse_attn.select import fused_score_and_select_blocks
    from mslk.attention.sparse_attn.sparsity_masks import (
        build_fa4_block_sparse_tensors,
    )

    compress_block_size = 64
    num_selected_blocks = 16
    window_size = 512
    q_tile_size = 256
    n_block_size = 128
    softmax_scale = 1.0 / D**0.5

    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

    # Step 1: Compress
    t_compress = time_fn(lambda: fused_compress_kv(K, V, compress_block_size))
    K_cmp, V_cmp = fused_compress_kv(K, V, compress_block_size)

    # Step 2: Score + select
    t_select = time_fn(
        lambda: fused_score_and_select_blocks(
            Q, K_cmp, num_selected_blocks, compress_block_size,
            causal=True, q_tile_size=q_tile_size, softmax_scale=softmax_scale,
        )
    )
    block_indices = fused_score_and_select_blocks(
        Q, K_cmp, num_selected_blocks, compress_block_size,
        causal=True, q_tile_size=q_tile_size, softmax_scale=softmax_scale,
    )

    # Step 3: Build sparsity masks
    t_masks = time_fn(
        lambda: build_fa4_block_sparse_tensors(
            block_indices, compress_block_size,
            n_block_size=n_block_size, seqlen_k=N,
        )
    )
    sparse_tensors = build_fa4_block_sparse_tensors(
        block_indices, compress_block_size,
        n_block_size=n_block_size, seqlen_k=N,
    )

    # Step 4a: Compressed attention
    from mslk.attention.sparse_attn.nsa_forward import _make_compressed_causal_mask
    compressed_mask = _make_compressed_causal_mask(compress_block_size)
    t_cmp = time_fn(
        lambda: _fa4_fwd(Q, K_cmp, V_cmp, causal=False,
                         softmax_scale=softmax_scale, mask_mod=compressed_mask)
    )

    # Step 4b: Selected attention
    t_slc = time_fn(
        lambda: _fa4_fwd(Q, K, V, causal=True, softmax_scale=softmax_scale,
                         block_sparse_tensors=sparse_tensors)
    )

    # Step 4c: Sliding window attention
    t_sld = time_fn(
        lambda: _fa4_fwd(Q, K, V, causal=True, softmax_scale=softmax_scale,
                         window_size_left=window_size, window_size_right=0)
    )

    # Step 5: Gating
    t_gate = time_fn(
        lambda: fused_gate_and_combine(
            Q,
            torch.randn_like(Q), torch.randn_like(Q), torch.randn_like(Q),
            None,
        )
    )

    total = t_compress + t_select + t_masks + t_cmp + t_slc + t_sld + t_gate
    print(f"N={N:>7d} | total={total:>7.2f}ms")
    print(f"  compress: {t_compress:>6.2f}ms ({t_compress/total*100:>4.1f}%)")
    print(f"  select:   {t_select:>6.2f}ms ({t_select/total*100:>4.1f}%)")
    print(f"  masks:    {t_masks:>6.2f}ms ({t_masks/total*100:>4.1f}%)")
    print(f"  FA4 cmp:  {t_cmp:>6.2f}ms ({t_cmp/total*100:>4.1f}%)")
    print(f"  FA4 slc:  {t_slc:>6.2f}ms ({t_slc/total*100:>4.1f}%)")
    print(f"  FA4 sld:  {t_sld:>6.2f}ms ({t_sld/total*100:>4.1f}%)")
    print(f"  gating:   {t_gate:>6.2f}ms ({t_gate/total*100:>4.1f}%)")
    print()


def main():
    print("NSA Forward Component Profile")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    for N in [4096, 8192, 16384, 32768, 65536, 131072]:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            profile_nsa_forward(N)
        except Exception as e:
            print(f"N={N:>7d} | FAILED: {e}")


if __name__ == "__main__":
    main()
