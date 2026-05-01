# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""NSA autograd utilities: block-sparse index transpose for backward pass.

The full NSAFunction(torch.autograd.Function) will be added in a subsequent diff.
This module provides the _transpose_block_sparse_for_bwd() utility needed by
the FA4 block-sparse backward.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _transpose_block_sparse_for_bwd(
    fwd_tensors,
    seqlen_q: int,
    seqlen_k: int,
    n_block_size: int,
    use_mask_blocks: bool = False,
):
    """Transpose forward block-sparse tensors for the backward pass.

    Forward: (B, H, n_q_tiles, k) — for each Q-tile, which KV-blocks
    Backward: (B, H, n_kv_blocks, max_q_per_kv) — for each KV-block, which Q-tiles

    The backward kernel iterates over KV-blocks and needs to know which Q-tiles
    contribute gradients to each KV-block.

    Uses a sparse inverted-index construction: O(total_entries) time and memory,
    no dense (n_q_tiles x n_kv_blocks) intermediate.

    Args:
        fwd_tensors: Forward BlockSparseTensorsTorch.
        seqlen_q: Query sequence length.
        seqlen_k: KV sequence length.
        n_block_size: KV block size for backward.
        use_mask_blocks: If True, read from mask_block_cnt/idx and output as
            mask blocks. Used for the compressed branch where mask_mod is needed.
    """
    from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch

    if use_mask_blocks:
        fwd_cnt = fwd_tensors.mask_block_cnt
        fwd_idx = fwd_tensors.mask_block_idx
    else:
        fwd_cnt = fwd_tensors.full_block_cnt  # (B, H, n_q_tiles)
        fwd_idx = fwd_tensors.full_block_idx  # (B, H, n_q_tiles, k)
    q_block_size, _ = fwd_tensors.block_size

    B, H, n_q_tiles = fwd_cnt.shape
    k = fwd_idx.shape[3]
    n_kv_blocks = (seqlen_k + n_block_size - 1) // n_block_size
    device = fwd_cnt.device

    # --- Sparse transpose: inverted index construction ---
    # Step 1: Count how many Q-tiles reference each KV-block
    idx_range = torch.arange(k, device=device)
    valid_mask = idx_range < fwd_cnt.unsqueeze(-1)  # (B, H, n_q_tiles, k)

    kv_indices = fwd_idx.long().clamp(0, n_kv_blocks - 1)
    bwd_cnt = torch.zeros(B, H, n_kv_blocks, dtype=torch.int32, device=device)
    ones = valid_mask.int()
    flat_kv = kv_indices.reshape(B, H, -1)
    flat_ones = ones.reshape(B, H, -1)
    bwd_cnt.scatter_add_(2, flat_kv, flat_ones)

    max_q_per_kv = bwd_cnt.max().item()
    if max_q_per_kv == 0:
        max_q_per_kv = 1

    # Step 2: Build compact backward index
    bwd_idx = torch.zeros(
        B, H, n_kv_blocks, max_q_per_kv, dtype=torch.int32, device=device
    )

    # Per-batch-head loop. B*H is small (typically 1*32=32) and k is small (~16).
    for b in range(B):
        for h in range(H):
            cnt_bh = fwd_cnt[b, h]
            idx_bh = fwd_idx[b, h]
            write_offsets = torch.zeros(n_kv_blocks, dtype=torch.long, device=device)
            for t in range(n_q_tiles):
                n_valid = cnt_bh[t].item()
                for ki in range(min(n_valid, k)):
                    kv_blk = idx_bh[t, ki].item()
                    if 0 <= kv_blk < n_kv_blocks:
                        off = write_offsets[kv_blk].item()
                        if off < max_q_per_kv:
                            bwd_idx[b, h, kv_blk, off] = t
                            write_offsets[kv_blk] += 1

    bwd_mask_cnt = torch.zeros(B, H, n_kv_blocks, dtype=torch.int32, device=device)
    bwd_mask_idx = torch.zeros(B, H, n_kv_blocks, 1, dtype=torch.int32, device=device)

    if use_mask_blocks:
        return BlockSparseTensorsTorch(
            mask_block_cnt=bwd_cnt,
            mask_block_idx=bwd_idx,
            full_block_cnt=bwd_mask_cnt,
            full_block_idx=bwd_mask_idx,
            block_size=(q_block_size, n_block_size),
        )

    return BlockSparseTensorsTorch(
        mask_block_cnt=bwd_mask_cnt,
        mask_block_idx=bwd_mask_idx,
        full_block_cnt=bwd_cnt,
        full_block_idx=bwd_idx,
        block_size=(q_block_size, n_block_size),
    )
