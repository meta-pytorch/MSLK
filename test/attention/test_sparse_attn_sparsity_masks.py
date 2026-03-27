# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for sparsity mask construction."""

import pytest
import torch


class TestBuildFA4BlockSparseTensors:
    """Test conversion from NSA block indices to FA4 format."""

    def test_output_types(self) -> None:
        """Output is a valid BlockSparseTensorsTorch."""
        from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch
        from mslk.attention.sparse_attn.sparsity_masks import (
            build_fa4_block_sparse_tensors,
        )

        B, H, N_q_tiles, k = 2, 4, 4, 8
        indices = torch.randint(
            0, 16, (B, H, N_q_tiles, k), device="cuda", dtype=torch.int32
        )

        result = build_fa4_block_sparse_tensors(
            indices,
            compress_block_size=128,
            n_block_size=128,
            seqlen_k=2048,
        )
        assert isinstance(result, BlockSparseTensorsTorch)
        assert result.full_block_cnt is not None
        assert result.full_block_idx is not None
        assert result.mask_block_cnt is not None
        assert result.mask_block_idx is not None

    def test_shapes(self) -> None:
        """Output tensors have correct shapes."""
        from mslk.attention.sparse_attn.sparsity_masks import (
            build_fa4_block_sparse_tensors,
        )

        B, H, N_q_tiles, k = 2, 4, 8, 4
        seqlen_k = 2048
        n_block_size = 128
        n_blocks_k = seqlen_k // n_block_size  # 16
        indices = torch.randint(
            0, 16, (B, H, N_q_tiles, k), device="cuda", dtype=torch.int32
        )

        result = build_fa4_block_sparse_tensors(
            indices,
            compress_block_size=128,
            n_block_size=n_block_size,
            seqlen_k=seqlen_k,
        )

        assert result.full_block_cnt.shape == (B, H, N_q_tiles)
        # full_block_idx uses dense shape (n_blocks_k) for backward transpose compatibility
        assert result.full_block_idx.shape == (B, H, N_q_tiles, n_blocks_k)
        assert result.mask_block_cnt.shape == (B, H, N_q_tiles)

    def test_full_block_count(self) -> None:
        """Full block count matches expected value."""
        from mslk.attention.sparse_attn.sparsity_masks import (
            build_fa4_block_sparse_tensors,
        )

        B, H, N_q_tiles, k = 1, 2, 4, 8
        indices = torch.randint(
            0, 16, (B, H, N_q_tiles, k), device="cuda", dtype=torch.int32
        )

        # compress_block_size == n_block_size, so 1:1 mapping
        result = build_fa4_block_sparse_tensors(
            indices,
            compress_block_size=128,
            n_block_size=128,
            seqlen_k=2048,
        )
        assert (result.full_block_cnt == k).all()

    def test_block_expansion(self) -> None:
        """When compress_block_size > n_block_size, blocks expand correctly."""
        from mslk.attention.sparse_attn.sparsity_masks import (
            build_fa4_block_sparse_tensors,
        )

        B, H, N_q_tiles, k = 1, 1, 2, 2
        # Select block 0 and block 3
        indices = torch.tensor([[[[0, 3]]]], device="cuda", dtype=torch.int32)

        result = build_fa4_block_sparse_tensors(
            indices,
            compress_block_size=256,
            n_block_size=128,
            seqlen_k=1024,
        )

        # Each NSA block of 256 -> 2 FA4 blocks of 128
        assert (result.full_block_cnt == 4).all()  # 2 blocks * 2 FA4 blocks each

        # Block 0 -> FA4 blocks [0, 1], Block 3 -> FA4 blocks [6, 7]
        expected_idx = result.full_block_idx[0, 0, 0, :4]
        assert expected_idx.tolist() == [0, 1, 6, 7]

    def test_mask_blocks_are_zero(self) -> None:
        """All mask block counts should be zero (selected blocks are fully attended)."""
        from mslk.attention.sparse_attn.sparsity_masks import (
            build_fa4_block_sparse_tensors,
        )

        B, H, N_q_tiles, k = 2, 4, 4, 8
        indices = torch.randint(
            0, 16, (B, H, N_q_tiles, k), device="cuda", dtype=torch.int32
        )

        result = build_fa4_block_sparse_tensors(
            indices,
            compress_block_size=128,
            n_block_size=128,
            seqlen_k=2048,
        )
        assert (result.mask_block_cnt == 0).all()

    def test_int32_dtype(self) -> None:
        """All output tensors are int32."""
        from mslk.attention.sparse_attn.sparsity_masks import (
            build_fa4_block_sparse_tensors,
        )

        indices = torch.randint(0, 8, (1, 2, 4, 4), device="cuda", dtype=torch.int32)
        result = build_fa4_block_sparse_tensors(
            indices,
            compress_block_size=128,
            n_block_size=128,
            seqlen_k=1024,
        )
        assert result.full_block_cnt.dtype == torch.int32
        assert result.full_block_idx.dtype == torch.int32
        assert result.mask_block_cnt.dtype == torch.int32
        assert result.mask_block_idx.dtype == torch.int32
