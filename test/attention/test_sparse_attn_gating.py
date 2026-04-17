# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for fused CuteDSL gating kernel.

Compares fused_gate_and_combine against the PyTorch reference
(compute_gates + gate_and_combine).
"""

import pytest
import torch


def _ref_gate_and_combine(Q, O_cmp, O_slc, O_sld, gate_proj_weight=None):
    """PyTorch reference: compute_gates then gate_and_combine."""
    from mslk.attention.sparse_attn.gating import compute_gates, gate_and_combine

    gates = compute_gates(Q, gate_proj_weight)
    return gate_and_combine(O_cmp, O_slc, O_sld, gates)


class TestFusedGating:
    """Compare fused CuteDSL kernel against PyTorch reference."""

    @pytest.mark.parametrize(
        "B,N,H,D",
        [
            (1, 64, 4, 128),
            (2, 128, 8, 128),
            (1, 256, 32, 128),
            (2, 1024, 4, 128),
            (1, 64, 4, 256),
        ],
    )
    def test_with_gate_weight(self, B, N, H, D) -> None:
        """Fused kernel matches reference when gate_proj_weight is provided."""
        from mslk.attention.sparse_attn.gating import fused_gate_and_combine

        dtype = torch.bfloat16
        Q = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_cmp = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_slc = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_sld = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        gate_proj_weight = torch.randn(H, 3, D, device="cuda", dtype=dtype)

        out_ref = _ref_gate_and_combine(Q, O_cmp, O_slc, O_sld, gate_proj_weight)
        out_fused = fused_gate_and_combine(Q, O_cmp, O_slc, O_sld, gate_proj_weight)

        assert out_fused.shape == out_ref.shape
        max_diff = (out_fused.float() - out_ref.float()).abs().max().item()
        mean_diff = (out_fused.float() - out_ref.float()).abs().mean().item()
        # Tolerance accounts for bf16 intermediate rounding differences:
        # the reference casts gates to bf16 between compute_gates and
        # gate_and_combine, while the fused kernel keeps them in fp32.
        assert max_diff < 0.05, f"Max diff too large: {max_diff}"
        assert mean_diff < 5e-3, f"Mean diff too large: {mean_diff}"

    @pytest.mark.parametrize(
        "B,N,H,D",
        [
            (1, 64, 4, 128),
            (2, 256, 8, 128),
            (1, 1024, 32, 128),
        ],
    )
    def test_without_gate_weight(self, B, N, H, D) -> None:
        """Fused kernel matches reference with uniform gates (no gate_proj_weight)."""
        from mslk.attention.sparse_attn.gating import fused_gate_and_combine

        dtype = torch.bfloat16
        Q = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_cmp = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_slc = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_sld = torch.randn(B, N, H, D, device="cuda", dtype=dtype)

        out_ref = _ref_gate_and_combine(Q, O_cmp, O_slc, O_sld)
        out_fused = fused_gate_and_combine(Q, O_cmp, O_slc, O_sld)

        assert out_fused.shape == out_ref.shape
        max_diff = (out_fused.float() - out_ref.float()).abs().max().item()
        mean_diff = (out_fused.float() - out_ref.float()).abs().mean().item()
        assert max_diff < 0.05, f"Max diff too large: {max_diff}"
        assert mean_diff < 5e-3, f"Mean diff too large: {mean_diff}"

    def test_output_finite(self) -> None:
        """Fused kernel produces finite outputs."""
        from mslk.attention.sparse_attn.gating import fused_gate_and_combine

        B, N, H, D = 2, 512, 4, 128
        dtype = torch.bfloat16
        Q = torch.randn(B, N, H, D, device="cuda", dtype=dtype) * 0.1
        O_cmp = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_slc = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_sld = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        gate_proj_weight = torch.randn(H, 3, D, device="cuda", dtype=dtype) * 0.1

        out = fused_gate_and_combine(Q, O_cmp, O_slc, O_sld, gate_proj_weight)
        assert out.shape == (B, N, H, D)
        assert torch.isfinite(out).all()

    def test_dtype_float16(self) -> None:
        """Fused kernel works with float16."""
        from mslk.attention.sparse_attn.gating import fused_gate_and_combine

        B, N, H, D = 1, 64, 4, 128
        dtype = torch.float16
        Q = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_cmp = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_slc = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        O_sld = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        gate_proj_weight = torch.randn(H, 3, D, device="cuda", dtype=dtype)

        out_ref = _ref_gate_and_combine(Q, O_cmp, O_slc, O_sld, gate_proj_weight)
        out_fused = fused_gate_and_combine(Q, O_cmp, O_slc, O_sld, gate_proj_weight)

        max_diff = (out_fused.float() - out_ref.float()).abs().max().item()
        assert max_diff < 1e-2, f"Max diff too large: {max_diff}"
