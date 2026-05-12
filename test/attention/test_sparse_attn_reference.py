# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for the pure PyTorch NSA reference implementation."""

import math

import pytest
import torch


class TestNSAReferenceBasic:
    """Basic sanity checks for the reference implementation."""

    def test_output_shape(self) -> None:
        """Output shape matches input query shape."""
        from mslk.attention.sparse_attn.reference import nsa_forward_reference

        B, N, H, D = 1, 512, 4, 64
        H_kv = 4
        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        out = nsa_forward_reference(
            Q, K, V,
            compress_block_size=64,
            num_selected_blocks=4,
            window_size=256,
            causal=True,
        )
        assert out.shape == (B, N, H, D)
        assert out.dtype == torch.bfloat16

    def test_output_shape_gqa(self) -> None:
        """Output shape correct with GQA (H > H_kv)."""
        from mslk.attention.sparse_attn.reference import nsa_forward_reference

        B, N, H, D = 1, 512, 8, 64
        H_kv = 2
        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        out = nsa_forward_reference(
            Q, K, V,
            compress_block_size=64,
            num_selected_blocks=4,
            window_size=256,
            causal=True,
        )
        assert out.shape == (B, N, H, D)

    def test_output_finite(self) -> None:
        """Output contains no NaN or Inf values."""
        from mslk.attention.sparse_attn.reference import nsa_forward_reference

        B, N, H, D = 2, 512, 4, 64
        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)

        out = nsa_forward_reference(
            Q, K, V,
            compress_block_size=64,
            num_selected_blocks=4,
            window_size=256,
            causal=True,
        )
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_with_gate_weights(self) -> None:
        """Test with explicit gate projection weights."""
        from mslk.attention.sparse_attn.reference import nsa_forward_reference

        B, N, H, D = 1, 512, 4, 64
        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        gate_proj = torch.randn(H, 3, D, device="cuda", dtype=torch.bfloat16)

        out = nsa_forward_reference(
            Q, K, V,
            compress_block_size=64,
            num_selected_blocks=4,
            window_size=256,
            gate_proj_weight=gate_proj,
            causal=True,
        )
        assert out.shape == (B, N, H, D)
        assert torch.isfinite(out).all()

    def test_with_compression_weights(self) -> None:
        """Test with learned KV compression projections."""
        from mslk.attention.sparse_attn.reference import nsa_forward_reference

        B, N, H, D = 1, 512, 4, 64
        H_kv = 4
        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        W_k = torch.randn(H_kv, D, D, device="cuda", dtype=torch.bfloat16) * 0.02
        W_v = torch.randn(H_kv, D, D, device="cuda", dtype=torch.bfloat16) * 0.02

        out = nsa_forward_reference(
            Q, K, V,
            compress_block_size=64,
            num_selected_blocks=4,
            window_size=256,
            W_k_compress=W_k,
            W_v_compress=W_v,
            causal=True,
        )
        assert out.shape == (B, N, H, D)
        assert torch.isfinite(out).all()

    def test_noncausal(self) -> None:
        """Test non-causal mode."""
        from mslk.attention.sparse_attn.reference import nsa_forward_reference

        B, N, H, D = 1, 512, 4, 64
        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

        out = nsa_forward_reference(
            Q, K, V,
            compress_block_size=64,
            num_selected_blocks=4,
            window_size=256,
            causal=False,
        )
        assert out.shape == (B, N, H, D)
        assert torch.isfinite(out).all()
