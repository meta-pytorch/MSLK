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
    from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch
    from mslk.fb.mslk.attention.flash_attn.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
    )

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

    mask_block_cnt = torch.zeros(B, H, n_q_tiles, dtype=torch.int32, device="cuda")
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
        Q,
        K,
        V,
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

    bwd_mask_cnt = torch.zeros(B, H, n_kv_blocks, dtype=torch.int32, device="cuda")
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
        Q,
        K,
        V,
        out,
        dO,
        lse,
        causal=True,
        block_sparse_tensors=bwd_sparse,
    )
    assert dQ.shape == Q.shape, f"dQ shape mismatch: {dQ.shape}"
    assert dK.shape == K.shape, f"dK shape mismatch: {dK.shape}"
    assert dV.shape == V.shape, f"dV shape mismatch: {dV.shape}"
    assert torch.isfinite(dQ).all(), "dQ has NaN/Inf"
    assert torch.isfinite(dK).all(), "dK has NaN/Inf"
    assert torch.isfinite(dV).all(), "dV has NaN/Inf"


def test_fa4_block_sparse_backward_varlen():
    """_flash_attn_bwd should support varlen + block_sparse_tensors on SM100.

    Verifies that the backward pass with cu_seqlens and block_sparse_tensors
    produces the same gradients as the padded non-varlen version.
    """
    from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch
    from mslk.fb.mslk.attention.flash_attn.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
    )

    seqlens = [512, 256]
    total = sum(seqlens)
    max_seqlen = max(seqlens)
    batch_size = len(seqlens)
    H, D = 4, 128
    H_kv = H
    q_tile_size = 256
    n_block_size = 128

    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seqlens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    torch.manual_seed(42)
    Q_3d = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
    K_3d = torch.randn(total, H_kv, D, device="cuda", dtype=torch.bfloat16)
    V_3d = torch.randn(total, H_kv, D, device="cuda", dtype=torch.bfloat16)
    dO_3d = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)

    # Pad to 4D for reference
    pad_N = max_seqlen
    Q_4d = Q_3d.new_zeros(batch_size, pad_N, H, D)
    K_4d = K_3d.new_zeros(batch_size, pad_N, H_kv, D)
    V_4d = V_3d.new_zeros(batch_size, pad_N, H_kv, D)
    dO_4d = dO_3d.new_zeros(batch_size, pad_N, H, D)
    for i, slen in enumerate(seqlens):
        s = cu_seqlens[i].item()
        Q_4d[i, :slen] = Q_3d[s : s + slen]
        K_4d[i, :slen] = K_3d[s : s + slen]
        V_4d[i, :slen] = V_3d[s : s + slen]
        dO_4d[i, :slen] = dO_3d[s : s + slen]

    # Build simple block-sparse pattern per batch element
    n_q_tiles = pad_N // q_tile_size
    n_kv_blocks = pad_N // n_block_size
    k_selected = 2

    # Forward sparse tensors (Q-tile → KV-blocks)
    fwd_cnt = torch.full(
        (batch_size, H, n_q_tiles), k_selected, dtype=torch.int32, device="cuda"
    )
    fwd_idx = torch.zeros(
        batch_size, H, n_q_tiles, n_kv_blocks, dtype=torch.int32, device="cuda"
    )
    for qt in range(n_q_tiles):
        for i in range(k_selected):
            fwd_idx[:, :, qt, i] = min(qt + i, n_kv_blocks - 1)

    fwd_mask_cnt = torch.zeros(
        batch_size, H, n_q_tiles, dtype=torch.int32, device="cuda"
    )
    fwd_mask_idx = torch.zeros(
        batch_size, H, n_q_tiles, n_kv_blocks, dtype=torch.int32, device="cuda"
    )
    fwd_sparse = BlockSparseTensorsTorch(
        mask_block_cnt=fwd_mask_cnt,
        mask_block_idx=fwd_mask_idx,
        full_block_cnt=fwd_cnt,
        full_block_idx=fwd_idx,
        block_size=(q_tile_size, n_block_size),
    )

    # Backward sparse tensors (KV-block → Q-tiles)
    bwd_cnt = torch.zeros(batch_size, H, n_kv_blocks, dtype=torch.int32, device="cuda")
    bwd_idx = torch.zeros(
        batch_size, H, n_kv_blocks, n_q_tiles, dtype=torch.int32, device="cuda"
    )
    for b in range(batch_size):
        for h in range(H):
            for qt in range(n_q_tiles):
                cnt = fwd_cnt[b, h, qt].item()
                for i in range(cnt):
                    kv = fwd_idx[b, h, qt, i].item()
                    pos = bwd_cnt[b, h, kv].item()
                    bwd_idx[b, h, kv, pos] = qt
                    bwd_cnt[b, h, kv] += 1

    bwd_mask_cnt = torch.zeros(
        batch_size, H, n_kv_blocks, dtype=torch.int32, device="cuda"
    )
    bwd_mask_idx = torch.zeros(
        batch_size, H, n_kv_blocks, n_q_tiles, dtype=torch.int32, device="cuda"
    )
    bwd_sparse = BlockSparseTensorsTorch(
        mask_block_cnt=bwd_mask_cnt,
        mask_block_idx=bwd_mask_idx,
        full_block_cnt=bwd_cnt,
        full_block_idx=bwd_idx,
        block_size=(q_tile_size, n_block_size),
    )

    # Reference: padded 4D forward + backward
    out_4d, lse_4d = _flash_attn_fwd(
        Q_4d,
        K_4d,
        V_4d,
        causal=True,
        block_sparse_tensors=fwd_sparse,
        return_lse=True,
    )
    dQ_4d, dK_4d, dV_4d = _flash_attn_bwd(
        Q_4d,
        K_4d,
        V_4d,
        out_4d,
        dO_4d,
        lse_4d,
        causal=True,
        block_sparse_tensors=bwd_sparse,
    )

    # Varlen: 3D forward + backward with cu_seqlens
    out_3d, lse_3d = _flash_attn_fwd(
        Q_3d,
        K_3d,
        V_3d,
        causal=True,
        block_sparse_tensors=fwd_sparse,
        return_lse=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )
    dQ_3d, dK_3d, dV_3d = _flash_attn_bwd(
        Q_3d,
        K_3d,
        V_3d,
        out_3d,
        dO_3d,
        lse_3d,
        causal=True,
        block_sparse_tensors=bwd_sparse,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )

    # Compare: varlen gradients should match padded per-sequence
    assert dQ_3d.shape == Q_3d.shape, f"dQ shape: {dQ_3d.shape} vs {Q_3d.shape}"
    assert dK_3d.shape == K_3d.shape, f"dK shape: {dK_3d.shape} vs {K_3d.shape}"
    assert torch.isfinite(dQ_3d).all(), "dQ_3d has NaN/Inf"
    assert torch.isfinite(dK_3d).all(), "dK_3d has NaN/Inf"
    assert torch.isfinite(dV_3d).all(), "dV_3d has NaN/Inf"

    for i, slen in enumerate(seqlens):
        s = cu_seqlens[i].item()
        dQ_ref = dQ_4d[i, :slen]
        dQ_var = dQ_3d[s : s + slen]
        max_diff = (dQ_ref.float() - dQ_var.float()).abs().max().item()
        assert max_diff < 0.05, f"Seq {i}: dQ max_diff={max_diff:.6f} exceeds 0.05"
