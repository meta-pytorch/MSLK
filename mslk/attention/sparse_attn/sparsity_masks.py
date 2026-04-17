# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Convert top-k block indices into FA4's BlockSparseTensorsTorch format."""

from __future__ import annotations

import torch
from torch import Tensor

from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch


def build_fa4_block_sparse_tensors(
    block_indices: Tensor,  # (B, H, N_q_tiles, k) int32
    compress_block_size: int,
    n_block_size: int = 128,
    seqlen_k: int | None = None,
) -> BlockSparseTensorsTorch:
    """Convert selected KV block indices into FA4's block sparsity format.

    FA4 block sparsity uses two pairs of tensors:
    - full_block_cnt/full_block_idx: KV blocks that need NO masking (fully attended).
    - mask_block_cnt/mask_block_idx: KV blocks that need masking (partially attended).

    Handles two cases:
    - compress_block_size >= n_block_size: Each NSA block expands to multiple FA4 blocks.
    - compress_block_size < n_block_size: Multiple NSA blocks contract to one FA4 block
      (duplicates are removed).

    All selected blocks are "full" blocks (no partial masking needed), since we're
    attending to entire KV blocks.

    Args:
        block_indices: Selected KV block indices, shape (B, H, N_q_tiles, k).
            Values are indices into the compressed KV sequence.
        compress_block_size: Size of each NSA KV block.
        n_block_size: FA4's KV tile size (default 128 for SM100).
        seqlen_k: Total KV sequence length (for computing n_blocks_k).

    Returns:
        BlockSparseTensorsTorch with full_block_cnt, full_block_idx,
        mask_block_cnt, mask_block_idx.
    """
    B, H, N_q_tiles, k = block_indices.shape
    device = block_indices.device

    if compress_block_size >= n_block_size:
        # Each NSA block maps to one or more FA4 blocks
        assert compress_block_size % n_block_size == 0, (
            f"compress_block_size ({compress_block_size}) must be divisible by "
            f"n_block_size ({n_block_size})"
        )
        fa4_blocks_per_nsa_block = compress_block_size // n_block_size

        # NSA block j maps to FA4 blocks [j*ratio, j*ratio+1, ..., j*ratio+ratio-1]
        base_indices = block_indices.long() * fa4_blocks_per_nsa_block
        offsets = torch.arange(fa4_blocks_per_nsa_block, device=device)
        expanded_indices = base_indices.unsqueeze(-1) + offsets
        fa4_indices = expanded_indices.reshape(B, H, N_q_tiles, k * fa4_blocks_per_nsa_block)
        k_fa4 = k * fa4_blocks_per_nsa_block
    else:
        # Multiple NSA blocks map to a single FA4 block
        assert n_block_size % compress_block_size == 0, (
            f"n_block_size ({n_block_size}) must be divisible by "
            f"compress_block_size ({compress_block_size})"
        )
        # NSA block j maps to FA4 block j * compress_block_size // n_block_size
        fa4_block_indices = (block_indices.long() * compress_block_size) // n_block_size

        # Remove duplicates per query tile: sort, then unique via consecutive dedup
        fa4_block_indices = fa4_block_indices.sort(dim=-1).values
        # Mark duplicates: compare with previous element
        shifted = torch.roll(fa4_block_indices, 1, dims=-1)
        is_dup = (fa4_block_indices == shifted) & (torch.arange(k, device=device) > 0)
        # Replace duplicates with a sentinel (max_n_blocks) that we'll filter out
        if seqlen_k is not None:
            sentinel = (seqlen_k + n_block_size - 1) // n_block_size
        else:
            sentinel = fa4_block_indices.max().item() + 1
        fa4_block_indices = fa4_block_indices.masked_fill(is_dup, sentinel)
        # Re-sort to push sentinels to the end
        fa4_block_indices = fa4_block_indices.sort(dim=-1).values
        # Count non-sentinel entries per query tile
        counts = (fa4_block_indices < sentinel).sum(dim=-1)  # (B, H, N_q_tiles)

        fa4_indices = fa4_block_indices
        k_fa4 = k  # max possible, actual counts may be smaller

    # full_block_cnt: (B, H, N_q_tiles)
    if compress_block_size >= n_block_size:
        full_block_cnt = torch.full(
            (B, H, N_q_tiles), k_fa4, dtype=torch.int32, device=device
        )
    else:
        full_block_cnt = counts.to(torch.int32)

    # full_block_idx: use compact shape (B, H, N_q_tiles, k_fa4) instead of
    # (B, H, N_q_tiles, N/n_block_size). FA4 only accesses indices 0..cnt-1,
    # so the last dimension only needs to be k_fa4. This reduces memory from
    # O(N²) to O(N * k_fa4) — e.g. 8 MB vs 8.6 GB at N=1M.
    full_block_idx = fa4_indices[:, :, :, :k_fa4].to(torch.int32)

    # mask_block_cnt/idx: no partial masking needed. Use shape (B,H,N_q_tiles,1)
    # for mask_block_idx since count is always 0.
    mask_block_cnt = torch.zeros(
        (B, H, N_q_tiles), dtype=torch.int32, device=device
    )
    mask_block_idx = torch.zeros(
        (B, H, N_q_tiles, 1), dtype=torch.int32, device=device
    )

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
    )
