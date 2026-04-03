# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_score_and_select_blocks_basic():
    """Block selection returns valid indices with correct shape."""
    from mslk.attention.sparse_attn.select import score_and_select_blocks

    B, N, H, D = 1, 1024, 4, 128
    H_kv = 2
    block_size = 64
    num_selected = 8
    q_tile_size = 256

    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    N_cmp = N // block_size
    K_cmp = torch.randn(B, N_cmp, H_kv, D, device="cuda", dtype=torch.bfloat16)

    indices = score_and_select_blocks(
        Q,
        K_cmp,
        num_selected,
        block_size,
        causal=True,
        q_tile_size=q_tile_size,
    )

    N_q_tiles = N // q_tile_size
    assert indices.shape == (B, H, N_q_tiles, num_selected)
    assert indices.dtype == torch.int32

    # All indices should be valid (in range [0, N_cmp))
    assert (indices >= 0).all()
    assert (indices < N_cmp).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_score_and_select_blocks_causal():
    """Causal masking prevents selecting future blocks."""
    from mslk.attention.sparse_attn.select import score_and_select_blocks

    B, N, H, D = 1, 1024, 2, 128
    H_kv = 2
    block_size = 64
    num_selected = 4
    q_tile_size = 256

    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    N_cmp = N // block_size
    K_cmp = torch.randn(B, N_cmp, H_kv, D, device="cuda", dtype=torch.bfloat16)

    indices = score_and_select_blocks(
        Q,
        K_cmp,
        num_selected,
        block_size,
        causal=True,
        q_tile_size=q_tile_size,
    )

    # For the first query tile (positions 0..255), future blocks start at
    # block j where j * block_size >= q_tile_size = 256, i.e. j >= 4
    first_tile_indices = indices[:, :, 0]
    max_allowed_block = q_tile_size // block_size - 1  # block 3
    # All selected blocks for tile 0 should be <= max_allowed_block
    assert (first_tile_indices <= max_allowed_block).all(), (
        f"First tile selected future blocks: {first_tile_indices}"
    )
