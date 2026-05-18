# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for block scoring and top-k selection."""

import pytest
import torch


class TestScoreAndSelectBlocks:
    """Test block scoring and selection."""

    def test_output_shape(self) -> None:
        """Selected indices have correct shape."""
        from mslk.attention.sparse_attn.select import score_and_select_blocks
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H, D = 2, 1024, 4, 64
        H_kv = 4
        block_size = 64
        num_selected = 8
        q_tile_size = 256

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = score_and_select_blocks(
            Q, K_cmp, num_selected, block_size,
            causal=True, q_tile_size=q_tile_size,
        )

        N_q_tiles = N // q_tile_size
        N_cmp = N // block_size
        k_actual = min(num_selected, N_cmp)
        assert indices.shape == (B, H, N_q_tiles, k_actual)
        assert indices.dtype == torch.int32

    def test_indices_in_range(self) -> None:
        """All selected indices are valid KV block indices."""
        from mslk.attention.sparse_attn.select import score_and_select_blocks
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H, D = 2, 1024, 4, 64
        block_size = 64
        num_selected = 8

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = score_and_select_blocks(
            Q, K_cmp, num_selected, block_size,
            causal=True, q_tile_size=256,
        )

        N_cmp = N // block_size
        assert (indices >= 0).all()
        assert (indices < N_cmp).all()

    def test_causal_no_future_blocks(self) -> None:
        """Causal mode: no future blocks are selected."""
        from mslk.attention.sparse_attn.select import score_and_select_blocks
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H, D = 1, 1024, 2, 64
        block_size = 64
        num_selected = 8
        q_tile_size = 256

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = score_and_select_blocks(
            Q, K_cmp, num_selected, block_size,
            causal=True, q_tile_size=q_tile_size,
        )

        N_q_tiles = N // q_tile_size
        for qt in range(N_q_tiles):
            # Query tile qt covers positions [qt*q_tile_size, (qt+1)*q_tile_size)
            # KV block idx covers positions [idx*block_size, (idx+1)*block_size)
            # Causal: KV block start must be < query tile end
            q_tile_end = (qt + 1) * q_tile_size
            max_allowed_block = q_tile_end // block_size
            assert (indices[:, :, qt, :] < max_allowed_block).all(), (
                f"Query tile {qt}: found future block indices"
            )

    def test_indices_sorted(self) -> None:
        """Selected indices are sorted per query tile."""
        from mslk.attention.sparse_attn.select import score_and_select_blocks
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H, D = 2, 1024, 4, 64
        block_size = 64
        num_selected = 8

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = score_and_select_blocks(
            Q, K_cmp, num_selected, block_size,
            causal=True, q_tile_size=256,
        )

        # Check sorted along last dim
        sorted_indices = indices.sort(dim=-1).values
        assert (indices == sorted_indices).all()

    def test_deterministic(self) -> None:
        """Same input produces same selection."""
        from mslk.attention.sparse_attn.select import score_and_select_blocks
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H, D = 1, 1024, 4, 64
        block_size = 64
        num_selected = 8

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices1 = score_and_select_blocks(
            Q, K_cmp, num_selected, block_size, causal=True, q_tile_size=256
        )
        indices2 = score_and_select_blocks(
            Q, K_cmp, num_selected, block_size, causal=True, q_tile_size=256
        )
        assert (indices1 == indices2).all()

    def test_gqa(self) -> None:
        """Works correctly with GQA (H > H_kv)."""
        from mslk.attention.sparse_attn.select import score_and_select_blocks
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H, D = 1, 1024, 8, 64
        H_kv = 2
        block_size = 64
        num_selected = 4

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = score_and_select_blocks(
            Q, K_cmp, num_selected, block_size, causal=True, q_tile_size=256
        )
        N_q_tiles = N // 256
        assert indices.shape == (B, H, N_q_tiles, num_selected)
