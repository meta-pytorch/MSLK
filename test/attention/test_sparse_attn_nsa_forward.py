# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""End-to-end correctness tests for NSA forward pass.

Compares the FA4-based NSA implementation against the pure PyTorch reference.
"""

import math

import pytest
import torch


def _make_inputs(B, N, H, H_kv, D, dtype, device="cuda"):
    """Create random Q, K, V inputs."""
    Q = torch.randn(B, N, H, D, device=device, dtype=dtype) * 0.1
    K = torch.randn(B, N, H_kv, D, device=device, dtype=dtype) * 0.1
    V = torch.randn(B, N, H_kv, D, device=device, dtype=dtype) * 0.1
    return Q, K, V


class TestNSAForwardCorrectness:
    """Compare FA4-based NSA against reference implementation."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("N", [1024, 2048])
    @pytest.mark.parametrize("causal", [True])
    def test_nsa_vs_reference(self, dtype, N, causal) -> None:
        """NSA forward pass matches reference within tolerance."""
        from mslk.attention.sparse_attn.nsa_forward import nsa_forward
        from mslk.attention.sparse_attn.reference import nsa_forward_reference

        B, H, H_kv, D = 1, 4, 4, 128
        compress_block_size = 64
        num_selected = 8
        window_size = 256
        q_tile_size = 256

        Q, K, V = _make_inputs(B, N, H, H_kv, D, dtype)

        # Reference (pure PyTorch)
        out_ref = nsa_forward_reference(
            Q, K, V,
            compress_block_size=compress_block_size,
            num_selected_blocks=num_selected,
            window_size=window_size,
            causal=causal,
            q_tile_size=q_tile_size,
        )

        # FA4-based
        out_fa4 = nsa_forward(
            Q, K, V,
            compress_block_size=compress_block_size,
            num_selected_blocks=num_selected,
            window_size=window_size,
            causal=causal,
            q_tile_size=q_tile_size,
        )

        max_diff = (out_fa4.float() - out_ref.float()).abs().max().item()
        mean_diff = (out_fa4.float() - out_ref.float()).abs().mean().item()

        assert max_diff < 0.1, f"Max diff too large: {max_diff}"
        assert mean_diff < 0.01, f"Mean diff too large: {mean_diff}"

    @pytest.mark.parametrize("H,H_kv", [(4, 4)])
    def test_nsa_gqa(self, H, H_kv) -> None:
        """NSA works with MHA configuration.

        Note: GQA (H > H_kv) is not yet supported with block sparsity in FA4
        (pack_gqa=True is not compatible with block_sparse_tensors).
        """
        from mslk.attention.sparse_attn.nsa_forward import nsa_forward

        B, N, D = 1, 1024, 128
        Q, K, V = _make_inputs(B, N, H, H_kv, D, torch.bfloat16)

        out = nsa_forward(
            Q, K, V,
            compress_block_size=64,
            num_selected_blocks=8,
            window_size=256,
            causal=True,
        )
        assert out.shape == (B, N, H, D)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("N", [65536])
    def test_nsa_large_n(self, N) -> None:
        """NSA handles large sequence lengths without OOM.

        Verifies that the tiled block scoring, compact index tensors, and
        chunked gating optimizations allow NSA to scale beyond 32K.
        """
        from mslk.attention.sparse_attn.nsa_forward import nsa_forward

        B, H, H_kv, D = 1, 4, 4, 128
        compress_block_size = 64
        num_selected = 16
        window_size = 512
        dtype = torch.bfloat16

        Q, K, V = _make_inputs(B, N, H, H_kv, D, dtype)

        out = nsa_forward(
            Q, K, V,
            compress_block_size=compress_block_size,
            num_selected_blocks=num_selected,
            window_size=window_size,
            causal=True,
        )
        assert out.shape == (B, N, H, D)
        assert torch.isfinite(out).all()


class TestNSAForwardComponentIntegration:
    """Test that individual components integrate correctly."""

    def test_compress_then_fa4(self) -> None:
        """FA4 can run on compressed KV sequence."""
        from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func
        from mslk.attention.sparse_attn.compress import compress_kv

        B, N, H, D = 1, 1024, 4, 128
        block_size = 64

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, V_cmp = compress_kv(K, V, block_size)
        # K_cmp: (B, N//64, H, D) = (1, 16, 4, 128)

        out, lse = flash_attn_func(Q, K_cmp, V_cmp, causal=True)
        assert out.shape == (B, N, H, D)
        assert torch.isfinite(out).all()

    def test_sliding_window_fa4(self) -> None:
        """FA4 sliding window attention works correctly."""
        from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func

        B, N, H, D = 1, 1024, 4, 128
        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        out, lse = flash_attn_func(
            Q, K, V, causal=True, window_size=(512, 0)
        )
        assert out.shape == (B, N, H, D)
        assert torch.isfinite(out).all()

    def test_block_sparse_fa4(self) -> None:
        """FA4 block-sparse attention works with our mask format."""
        from mslk.fb.mslk.attention.flash_attn.interface import _flash_attn_fwd
        from mslk.attention.sparse_attn.compress import compress_kv
        from mslk.attention.sparse_attn.select import score_and_select_blocks
        from mslk.attention.sparse_attn.sparsity_masks import build_fa4_block_sparse_tensors

        B, N, H, D = 1, 1024, 4, 128
        block_size = 64
        num_selected = 8

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        K_cmp, _ = compress_kv(K, V, block_size)
        block_indices = score_and_select_blocks(
            Q, K_cmp, num_selected, block_size,
            causal=True, q_tile_size=256,
        )
        sparse_tensors = build_fa4_block_sparse_tensors(
            block_indices, block_size, n_block_size=128, seqlen_k=N,
        )

        out, lse = _flash_attn_fwd(
            Q, K, V,
            causal=True,
            block_sparse_tensors=sparse_tensors,
        )
        assert out.shape == (B, N, H, D)
        assert torch.isfinite(out).all()
