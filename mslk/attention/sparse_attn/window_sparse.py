#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Build block-sparse tensors that represent a sliding window pattern.

Converts a sliding window (window_size_left) into FA4 BlockSparseTensorsTorch format,
allowing the sliding window branch to reuse the same compiled FA4 kernel as the
selected branch (block_sparse, causal=True).
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_sliding_window_block_sparse(
    seqlen: int,
    window_size: int,
    q_tile_size: int = 256,
    n_block_size: int = 128,
    batch_size: int = 1,
    num_heads: int = 1,
    device: str | torch.device = "cuda",
):
    """Build block-sparse tensors for a causal sliding window pattern.

    For each Q-tile, includes the KV-blocks that fall within [q - window_size, q]
    (causal window). FA4's causal=True handles the per-element masking within blocks.

    Args:
        seqlen: Total sequence length.
        window_size: Left window size (number of positions).
        q_tile_size: Size of each Q-tile (FA4 m_block_size).
        n_block_size: Size of each KV-block (FA4 n_block_size).
        batch_size: Batch size (pattern is replicated).
        num_heads: Number of heads (pattern is replicated).
        device: Device for output tensors.

    Returns:
        BlockSparseTensorsTorch with the window pattern.
    """
    from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch

    n_q_tiles = seqlen // q_tile_size
    n_kv_blocks = (seqlen + n_block_size - 1) // n_block_size

    # For each Q-tile t, compute which KV-blocks are in the window.
    # Q-tile t covers positions [t*q_tile_size, (t+1)*q_tile_size).
    # With causal + window_size_left, position q attends to [q-window_size, q].
    # For the tile, the earliest position is t*q_tile_size, which attends back to
    # t*q_tile_size - window_size. The latest is (t+1)*q_tile_size - 1.
    # KV-blocks needed: [max(0, (t*q_tile_size - window_size) // n_block_size),
    #                     ((t+1)*q_tile_size - 1) // n_block_size]

    # Build per-tile block lists
    max_blocks_per_tile = (window_size + q_tile_size + n_block_size - 1) // n_block_size + 1

    full_block_cnt = torch.zeros(1, 1, n_q_tiles, dtype=torch.int32, device=device)
    full_block_idx = torch.zeros(
        1, 1, n_q_tiles, n_kv_blocks, dtype=torch.int32, device=device
    )

    for t in range(n_q_tiles):
        q_start = t * q_tile_size
        q_end = (t + 1) * q_tile_size - 1
        kv_first = max(0, (q_start - window_size)) // n_block_size
        kv_last = min(q_end // n_block_size, n_kv_blocks - 1)
        cnt = kv_last - kv_first + 1
        full_block_cnt[0, 0, t] = cnt
        for i, kv in enumerate(range(kv_first, kv_last + 1)):
            full_block_idx[0, 0, t, i] = kv

    # Broadcast to (B, H, ...)
    full_block_cnt = full_block_cnt.expand(batch_size, num_heads, -1).contiguous()
    full_block_idx = full_block_idx.expand(batch_size, num_heads, -1, -1).contiguous()

    mask_block_cnt = torch.zeros(
        batch_size, num_heads, n_q_tiles, dtype=torch.int32, device=device
    )
    mask_block_idx = torch.zeros(
        batch_size, num_heads, n_q_tiles, n_kv_blocks, dtype=torch.int32, device=device
    )

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(q_tile_size, n_block_size),
    )
