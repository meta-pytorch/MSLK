# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Minimal reproducer for FA4 block-sparse backward bug on SM100.

_flash_attn_bwd with block_sparse_tensors fails during CuTe DSL JIT
compilation: blocksparse_tensors is None inside the while-loop after-block
in flash_bwd_sm100.py, causing TypeError in get_total_q_block_count_bwd.

This test calls _flash_attn_fwd (which works) then _flash_attn_bwd (which
crashes) with identical block_sparse_tensors, isolating the bug to the
backward kernel compilation.
"""

import torch


def test_fa4_block_sparse_backward():
    """_flash_attn_bwd should support block_sparse_tensors on SM100."""
    from mslk.fb.mslk.attention.flash_attn.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
    )
    from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch

    B, N, H, D = 1, 1024, 4, 128
    H_kv = H
    q_tile_size = 256
    n_block_size = 128

    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
    dO = torch.randn_like(Q)

    n_q_tiles = N // q_tile_size  # 4
    n_kv_blocks = N // n_block_size  # 8

    # Simple block-sparse pattern: each Q-tile attends to 4 KV-blocks
    k_selected = 4
    full_block_cnt = torch.full(
        (B, H, n_q_tiles), k_selected, dtype=torch.int32, device="cuda"
    )
    full_block_idx = torch.zeros(
        B, H, n_q_tiles, n_kv_blocks, dtype=torch.int32, device="cuda"
    )
    for qt in range(n_q_tiles):
        for i in range(k_selected):
            full_block_idx[:, :, qt, i] = min(qt + i, n_kv_blocks - 1)

    mask_block_cnt = torch.zeros(
        B, H, n_q_tiles, dtype=torch.int32, device="cuda"
    )
    mask_block_idx = torch.zeros(
        B, H, n_q_tiles, n_kv_blocks, dtype=torch.int32, device="cuda"
    )

    sparse_tensors = BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(q_tile_size, n_block_size),
    )

    # Forward works
    out, lse = _flash_attn_fwd(
        Q, K, V,
        causal=True,
        block_sparse_tensors=sparse_tensors,
        return_lse=True,
    )
    assert out.shape == Q.shape, f"Forward output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "Forward output has NaN/Inf"

    # Backward: need transposed block-sparse tensors (KV-blocks → Q-tiles)
    bwd_cnt = torch.zeros(B, H, n_kv_blocks, dtype=torch.int32, device="cuda")
    bwd_idx = torch.zeros(
        B, H, n_kv_blocks, n_q_tiles, dtype=torch.int32, device="cuda"
    )
    # Build transpose manually
    for b in range(B):
        for h in range(H):
            for qt in range(n_q_tiles):
                cnt = full_block_cnt[b, h, qt].item()
                for i in range(cnt):
                    kv = full_block_idx[b, h, qt, i].item()
                    pos = bwd_cnt[b, h, kv].item()
                    bwd_idx[b, h, kv, pos] = qt
                    bwd_cnt[b, h, kv] += 1

    bwd_mask_cnt = torch.zeros(
        B, H, n_kv_blocks, dtype=torch.int32, device="cuda"
    )
    bwd_mask_idx = torch.zeros(
        B, H, n_kv_blocks, n_q_tiles, dtype=torch.int32, device="cuda"
    )

    bwd_sparse = BlockSparseTensorsTorch(
        mask_block_cnt=bwd_mask_cnt,
        mask_block_idx=bwd_mask_idx,
        full_block_cnt=bwd_cnt,
        full_block_idx=bwd_idx,
        block_size=(q_tile_size, n_block_size),
    )

    # This is the call that crashes:
    # TypeError: cannot unpack non-iterable NoneType object
    # in get_total_q_block_count_bwd (block_sparse_utils.py)
    dQ, dK, dV = _flash_attn_bwd(
        Q, K, V, out, dO, lse,
        causal=True,
        block_sparse_tensors=bwd_sparse,
    )
    assert dQ.shape == Q.shape, f"dQ shape mismatch: {dQ.shape}"
    assert dK.shape == K.shape, f"dK shape mismatch: {dK.shape}"
    assert dV.shape == V.shape, f"dV shape mismatch: {dV.shape}"
    assert torch.isfinite(dQ).all(), "dQ has NaN/Inf"
    assert torch.isfinite(dK).all(), "dK has NaN/Inf"
    assert torch.isfinite(dV).all(), "dV has NaN/Inf"
