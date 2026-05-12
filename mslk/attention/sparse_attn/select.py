# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Block scoring and top-k selection for NSA.

Provides score_and_select_blocks (reference, tiled for O(N) memory).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def score_and_select_blocks(
    Q: Tensor,  # (B, N, H, D)
    K_cmp: Tensor,  # (B, N_cmp, H_kv, D)
    num_selected_blocks: int,
    compress_block_size: int,
    causal: bool = True,
    q_tile_size: int = 256,
    softmax_scale: float | None = None,
) -> Tensor:
    """Score KV blocks by importance and select top-k for each query tile.

    Uses compressed keys to efficiently compute block importance scores.
    Each query tile (q_tile_size positions) independently selects its own
    top-k KV blocks.

    Processes query tiles in chunks to avoid materializing a full (B, H, N, N_cmp)
    score tensor, reducing peak memory from O(N * N_cmp) to
    O(chunk_tiles * q_tile_size * N_cmp).

    Args:
        Q: Query tensor, shape (B, N, H, D).
        K_cmp: Compressed key tensor, shape (B, N_cmp, H_kv, D).
        num_selected_blocks: Number of KV blocks to select per query tile.
        compress_block_size: Size of each KV block.
        causal: Whether to apply causal masking (prevent selecting future blocks).
        q_tile_size: Size of each query tile (should match FA4's CTA tile size, 256).
        softmax_scale: Scaling factor for attention scores.

    Returns:
        block_indices: Selected KV block indices, shape (B, H, N_q_tiles, k).
    """
    B, N, H, D = Q.shape
    H_kv = K_cmp.shape[2]
    N_cmp = K_cmp.shape[1]
    groups = H // H_kv

    N_q_tiles = N // q_tile_size
    assert N % q_tile_size == 0, (
        f"Sequence length {N} must be divisible by q_tile_size {q_tile_size}"
    )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    k_actual = min(num_selected_blocks, N_cmp)

    # Expand K_cmp for GQA: (B, N_cmp, H_kv, D) -> (B, N_cmp, H, D)
    if groups > 1:
        K_cmp_expanded = K_cmp.repeat_interleave(groups, dim=2)
    else:
        K_cmp_expanded = K_cmp

    # fp32 accumulation for numerical stability with bf16/fp16 inputs
    K_cmp_t = K_cmp_expanded.permute(0, 2, 3, 1).float()

    block_indices = torch.empty(
        B, H, N_q_tiles, k_actual, dtype=torch.int32, device=Q.device
    )

    if causal:
        q_tile_end = (torch.arange(N_q_tiles, device=Q.device) + 1) * q_tile_size
        kv_block_start = torch.arange(N_cmp, device=Q.device) * compress_block_size

    # Process in chunks to bound peak memory
    chunk_tiles = min(16, N_q_tiles)
    for chunk_start in range(0, N_q_tiles, chunk_tiles):
        chunk_end = min(chunk_start + chunk_tiles, N_q_tiles)
        n_tiles = chunk_end - chunk_start
        q_start = chunk_start * q_tile_size
        q_end = chunk_end * q_tile_size

        # fp32 scoring for numerical stability
        scores_chunk = (
            torch.matmul(Q[:, q_start:q_end].permute(0, 2, 1, 3).float(), K_cmp_t)
            * softmax_scale
        )

        # Average per tile
        scores_agg = scores_chunk.reshape(B, H, n_tiles, q_tile_size, N_cmp).mean(dim=3)

        # Causal mask: block j is future if j * block_size >= (i+1) * q_tile_size
        if causal:
            future_mask = kv_block_start.unsqueeze(0) >= q_tile_end[
                chunk_start:chunk_end
            ].unsqueeze(1)
            scores_agg.masked_fill_(
                future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        topk_scores, chunk_idx = scores_agg.topk(k_actual, dim=-1)

        # Replace -inf selections with block 0
        invalid = topk_scores == float("-inf")
        if invalid.any():
            chunk_idx = chunk_idx.masked_fill(invalid, 0)

        # Sort ascending for better memory access patterns
        block_indices[:, :, chunk_start:chunk_end] = chunk_idx.sort(dim=-1).values.to(
            torch.int32
        )

    return block_indices
