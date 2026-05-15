# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_build_fa4_block_sparse_expansion():
    """Expansion case: compress_block_size >= n_block_size."""
    from mslk.attention.sparse_attn.sparsity_masks import build_fa4_block_sparse_tensors

    B, H, N_q_tiles, k = 1, 2, 4, 3
    compress_block_size = 256
    n_block_size = 128
    ratio = compress_block_size // n_block_size

    block_indices = torch.tensor(
        [[[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]],
        dtype=torch.int32,
        device="cuda",
    ).expand(B, H, N_q_tiles, k)

    result = build_fa4_block_sparse_tensors(
        block_indices,
        compress_block_size,
        n_block_size=n_block_size,
        seqlen_k=2048,
    )

    assert result.full_block_cnt.shape == (B, H, N_q_tiles)
    assert (result.full_block_cnt == k * ratio).all()
    assert result.full_block_idx.shape == (B, H, N_q_tiles, k * ratio)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_build_fa4_block_sparse_contraction():
    """Contraction case: compress_block_size < n_block_size with dedup."""
    from mslk.attention.sparse_attn.sparsity_masks import build_fa4_block_sparse_tensors

    B, H = 1, 1
    compress_block_size = 64
    n_block_size = 128

    # Blocks 0,1 -> FA4 block 0; blocks 2,3 -> FA4 block 1
    block_indices = torch.tensor(
        [[[[0, 1, 2, 3], [0, 2, 4, 6]]]],
        dtype=torch.int32,
        device="cuda",
    )

    result = build_fa4_block_sparse_tensors(
        block_indices,
        compress_block_size,
        n_block_size=n_block_size,
        seqlen_k=1024,
    )

    # Tile 0: blocks 0,1,2,3 -> FA4 blocks 0,0,1,1 -> dedup -> [0,1], count=2
    assert result.full_block_cnt[0, 0, 0].item() == 2
    # Tile 1: blocks 0,2,4,6 -> FA4 blocks 0,1,2,3 -> all unique, count=4
    assert result.full_block_cnt[0, 0, 1].item() == 4
