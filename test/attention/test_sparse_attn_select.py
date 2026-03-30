# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for block scoring and top-k selection."""

import pytest
import torch


class TestScoreAndSelectBlocks:
    """Test block scoring and selection."""

    def test_output_shape(self) -> None:
        """Selected indices have correct shape."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import score_and_select_blocks

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
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=q_tile_size,
        )

        N_q_tiles = N // q_tile_size
        N_cmp = N // block_size
        k_actual = min(num_selected, N_cmp)
        assert indices.shape == (B, H, N_q_tiles, k_actual)
        assert indices.dtype == torch.int32

    def test_indices_in_range(self) -> None:
        """All selected indices are valid KV block indices."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import score_and_select_blocks

        B, N, H, D = 2, 1024, 4, 64
        block_size = 64
        num_selected = 8

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = score_and_select_blocks(
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=256,
        )

        N_cmp = N // block_size
        assert (indices >= 0).all()
        assert (indices < N_cmp).all()

    def test_causal_no_future_blocks(self) -> None:
        """Causal mode: no future blocks are selected."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import score_and_select_blocks

        B, N, H, D = 1, 1024, 2, 64
        block_size = 64
        num_selected = 8
        q_tile_size = 256

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = score_and_select_blocks(
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=q_tile_size,
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
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import score_and_select_blocks

        B, N, H, D = 2, 1024, 4, 64
        block_size = 64
        num_selected = 8

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = score_and_select_blocks(
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=256,
        )

        # Check sorted along last dim
        sorted_indices = indices.sort(dim=-1).values
        assert (indices == sorted_indices).all()

    def test_deterministic(self) -> None:
        """Same input produces same selection."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import score_and_select_blocks

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
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import score_and_select_blocks

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


def _q_mean_reference(
    Q,
    K_cmp,
    num_selected_blocks,
    compress_block_size,
    causal=True,
    q_tile_size=256,
    softmax_scale=None,
):
    """Q_mean-based PyTorch reference for testing the fused kernel.

    Computes Q_mean in PyTorch, then scores via einsum. This isolates
    kernel correctness from the Q_mean algebraic equivalence.
    """
    import math

    B, N, H, D = Q.shape
    H_kv = K_cmp.shape[2]
    N_cmp = K_cmp.shape[1]
    N_q_tiles = N // q_tile_size

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    k_actual = min(num_selected_blocks, N_cmp)

    # Q_mean: (B, N_q_tiles, H, D)
    Q_mean = (
        Q.reshape(B, N_q_tiles, q_tile_size, H, D).mean(dim=2).float() * softmax_scale
    )

    # Expand K_cmp for GQA
    groups = H // H_kv
    if groups > 1:
        K_cmp_expanded = K_cmp.repeat_interleave(groups, dim=2)
    else:
        K_cmp_expanded = K_cmp

    # Score: (B, N_q_tiles, H, N_cmp)
    scores = torch.einsum(
        "bthd,bjhd->bthj", Q_mean, K_cmp_expanded.float()
    )  # (B, N_q_tiles, H, N_cmp)

    # Causal mask
    if causal:
        q_tile_end = (torch.arange(N_q_tiles, device=Q.device) + 1) * q_tile_size
        kv_block_start = torch.arange(N_cmp, device=Q.device) * compress_block_size
        future_mask = kv_block_start.unsqueeze(0) >= q_tile_end.unsqueeze(1)
        scores.masked_fill_(future_mask.unsqueeze(0).unsqueeze(2), float("-inf"))

    # Permute to (B, H, N_q_tiles, N_cmp) for topk
    scores = scores.permute(0, 2, 1, 3)

    topk_scores, topk_idx = scores.topk(k_actual, dim=-1)

    # Replace -inf entries with 0
    invalid = topk_scores == float("-inf")
    if invalid.any():
        topk_idx = topk_idx.masked_fill(invalid, 0)

    # Sort ascending
    block_indices = topk_idx.sort(dim=-1).values.to(torch.int32)
    return block_indices


class TestFusedScoreAndSelectBlocks:
    """Test fused CuteDSL block scoring and selection."""

    @pytest.mark.parametrize(
        "B, N, H, H_kv, D, block_size, num_selected",
        [
            (1, 1024, 4, 4, 64, 64, 8),
            (2, 1024, 4, 4, 64, 64, 8),
            (1, 2048, 8, 2, 128, 64, 16),
            (2, 4096, 4, 4, 64, 64, 8),
            (1, 1024, 4, 4, 128, 64, 4),
        ],
    )
    def test_index_agreement(self, B, N, H, H_kv, D, block_size, num_selected) -> None:
        """Fused kernel matches Q_mean-based PyTorch reference (exact)."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import fused_score_and_select_blocks

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        fused_idx = fused_score_and_select_blocks(
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=256,
        )
        ref_idx = _q_mean_reference(
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=256,
        )
        assert (fused_idx == ref_idx).all(), (
            f"Mismatch: fused vs Q_mean reference.\n"
            f"Fused:\n{fused_idx}\nRef:\n{ref_idx}"
        )

    def test_output_shape(self) -> None:
        """Selected indices have correct shape."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import fused_score_and_select_blocks

        B, N, H, D = 2, 1024, 4, 64
        H_kv = 4
        block_size = 64
        num_selected = 8
        q_tile_size = 256

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = fused_score_and_select_blocks(
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=q_tile_size,
        )

        N_q_tiles = N // q_tile_size
        N_cmp = N // block_size
        k_actual = min(num_selected, N_cmp)
        assert indices.shape == (B, H, N_q_tiles, k_actual)
        assert indices.dtype == torch.int32

    def test_indices_in_range(self) -> None:
        """All selected indices are valid KV block indices."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import fused_score_and_select_blocks

        B, N, H, D = 2, 1024, 4, 64
        block_size = 64
        num_selected = 8

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = fused_score_and_select_blocks(
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=256,
        )

        N_cmp = N // block_size
        assert (indices >= 0).all()
        assert (indices < N_cmp).all()

    def test_causal_no_future_blocks(self) -> None:
        """Causal mode: no future blocks are selected."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import fused_score_and_select_blocks

        B, N, H, D = 1, 1024, 2, 64
        block_size = 64
        num_selected = 8
        q_tile_size = 256

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = fused_score_and_select_blocks(
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=q_tile_size,
        )

        N_q_tiles = N // q_tile_size
        for qt in range(N_q_tiles):
            q_tile_end = (qt + 1) * q_tile_size
            max_allowed_block = q_tile_end // block_size
            assert (indices[:, :, qt, :] < max_allowed_block).all(), (
                f"Query tile {qt}: found future block indices"
            )

    def test_indices_sorted(self) -> None:
        """Selected indices are sorted per query tile."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import fused_score_and_select_blocks

        B, N, H, D = 2, 1024, 4, 64
        block_size = 64
        num_selected = 8

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = fused_score_and_select_blocks(
            Q,
            K_cmp,
            num_selected,
            block_size,
            causal=True,
            q_tile_size=256,
        )

        sorted_indices = indices.sort(dim=-1).values
        assert (indices == sorted_indices).all()

    def test_deterministic(self) -> None:
        """Same input produces same selection."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import fused_score_and_select_blocks

        B, N, H, D = 1, 1024, 4, 64
        block_size = 64
        num_selected = 8

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices1 = fused_score_and_select_blocks(
            Q, K_cmp, num_selected, block_size, causal=True, q_tile_size=256
        )
        indices2 = fused_score_and_select_blocks(
            Q, K_cmp, num_selected, block_size, causal=True, q_tile_size=256
        )
        assert (indices1 == indices2).all()

    def test_gqa(self) -> None:
        """Works correctly with GQA (H > H_kv)."""
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import fused_score_and_select_blocks

        B, N, H, D = 1, 1024, 8, 64
        H_kv = 2
        block_size = 64
        num_selected = 4

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        indices = fused_score_and_select_blocks(
            Q, K_cmp, num_selected, block_size, causal=True, q_tile_size=256
        )
        N_q_tiles = N // 256
        assert indices.shape == (B, H, N_q_tiles, num_selected)
