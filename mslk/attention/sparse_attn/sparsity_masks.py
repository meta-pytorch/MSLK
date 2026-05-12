# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Convert top-k block indices into FA4's BlockSparseTensorsTorch format."""

from __future__ import annotations

import torch
from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch
from torch import Tensor


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
        fa4_indices = expanded_indices.reshape(
            B, H, N_q_tiles, k * fa4_blocks_per_nsa_block
        )
        k_fa4 = k * fa4_blocks_per_nsa_block
    else:
        # Multiple NSA blocks map to a single FA4 block — contraction case.
        # Optimized path: since k is small (typically 16), we dedup by sorting
        # the k indices per tile and removing consecutive duplicates. This avoids
        # materializing a large (B, H, N_q_tiles, n_blocks_k) attendance matrix.
        assert n_block_size % compress_block_size == 0, (
            f"n_block_size ({n_block_size}) must be divisible by "
            f"compress_block_size ({compress_block_size})"
        )
        fa4_block_indices = (block_indices.long() * compress_block_size) // n_block_size

        if seqlen_k is not None:
            n_blocks_k = (seqlen_k + n_block_size - 1) // n_block_size
        else:
            n_blocks_k = fa4_block_indices.max().item() + 1

        # Sort the k indices per tile, then remove consecutive duplicates
        sorted_idx, _ = fa4_block_indices.sort(dim=-1)  # (B, H, N_q_tiles, k)

        # Consecutive dedup: mark positions where value differs from previous
        is_new = torch.ones_like(sorted_idx, dtype=torch.bool)
        is_new[..., 1:] = sorted_idx[..., 1:] != sorted_idx[..., :-1]

        # Count unique per tile
        counts = is_new.sum(dim=-1)  # (B, H, N_q_tiles)
        max_unique = counts.max().item()

        # Pack unique indices: scatter unique values to front
        # Use cumsum of is_new as destination positions
        dest_positions = is_new.long().cumsum(dim=-1) - 1  # 0-indexed positions

        fa4_indices = torch.zeros(
            B, H, N_q_tiles, n_blocks_k, dtype=torch.int64, device=device
        )
        # Scatter sorted unique values into the packed output
        fa4_indices.scatter_(3, dest_positions, sorted_idx)

        full_block_cnt = counts.to(torch.int32)
        k_fa4 = k

    # Compute the full number of KV blocks for FA4
    if compress_block_size >= n_block_size:
        if seqlen_k is not None:
            n_blocks_k = (seqlen_k + n_block_size - 1) // n_block_size
        else:
            n_blocks_k = fa4_indices.max().item() + 1
        full_block_cnt = torch.full(
            (B, H, N_q_tiles), k_fa4, dtype=torch.int32, device=device
        )

    # full_block_idx: use dense shape (B, H, N_q_tiles, n_blocks_k) so that
    # FA4's backward can transpose to Q-direction indexing. The compact format
    # (k_fa4 << n_blocks_k) saves memory but is incompatible with backward.
    full_block_idx_dense = torch.zeros(
        (B, H, N_q_tiles, n_blocks_k), dtype=torch.int32, device=device
    )
    # Copy compact indices into the dense tensor
    copy_len = min(k_fa4, n_blocks_k)
    full_block_idx_dense[:, :, :, :copy_len] = fa4_indices[:, :, :, :copy_len].to(
        torch.int32
    )

    # mask_block_cnt/idx: no partial masking needed.
    mask_block_cnt = torch.zeros((B, H, N_q_tiles), dtype=torch.int32, device=device)
    mask_block_idx = torch.zeros(
        (B, H, N_q_tiles, n_blocks_k), dtype=torch.int32, device=device
    )

    # Determine the Q-direction block size. With q_tile_size positions per query tile
    # and N_q_tiles tiles, each m-block covers q_tile_size positions.
    # For the contraction case (compress_block_size < n_block_size), the Q block size
    # is inferred from the tensor shapes.
    q_block_size = (
        seqlen_k // N_q_tiles if seqlen_k is not None else compress_block_size
    )

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx_dense,
        block_size=(q_block_size, n_block_size),
    )


def build_compressed_block_sparse_tensors(
    cmp_block_indices: Tensor,  # (B, H, N_q_tiles, k_cmp) int32
    n_kv_blocks: int,
    q_tile_size: int = 256,
    cmp_n_block_size: int = 128,
) -> BlockSparseTensorsTorch:
    """Build block-sparse tensors for block-sparse compressed attention.

    All selected blocks go into mask_block (not full_block) because the
    compressed causal mask (mask_mod) must be applied within each block.

    Args:
        cmp_block_indices: FA4 block indices in the compressed KV sequence,
            shape (B, H, N_q_tiles, k_cmp). Values in [0, n_kv_blocks).
        n_kv_blocks: Total number of FA4 KV blocks in the compressed sequence.
        q_tile_size: Q tile size (256 for SM100).
        cmp_n_block_size: FA4 KV block size for the compressed branch (128).

    Returns:
        BlockSparseTensorsTorch with selected blocks in mask_block
        (so FA4 applies mask_mod for compressed causal masking).
    """
    B, H, N_q_tiles, k_cmp = cmp_block_indices.shape
    device = cmp_block_indices.device

    # All blocks in mask_block (need compressed causal mask via mask_mod)
    mask_block_cnt = torch.full(
        (B, H, N_q_tiles), k_cmp, dtype=torch.int32, device=device
    )
    mask_block_idx = torch.zeros(
        (B, H, N_q_tiles, n_kv_blocks), dtype=torch.int32, device=device
    )
    mask_block_idx[:, :, :, :k_cmp] = cmp_block_indices

    # No full blocks
    full_block_cnt = torch.zeros((B, H, N_q_tiles), dtype=torch.int32, device=device)
    full_block_idx = torch.zeros(
        (B, H, N_q_tiles, n_kv_blocks), dtype=torch.int32, device=device
    )

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(q_tile_size, cmp_n_block_size),
    )
