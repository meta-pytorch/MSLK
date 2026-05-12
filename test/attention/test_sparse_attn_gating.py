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


class TestFusedGating3D:
    """Test gating functions with 3D varlen input (T, H, D)."""

    @pytest.mark.parametrize(
        "T,H,D",
        [
            (768, 4, 128),  # 512+256 packed
            (128, 8, 128),
            (2048, 32, 128),
        ],
    )
    def test_3d_matches_4d_with_gate_weight(self, T, H, D) -> None:
        """3D fused_gate_and_combine matches equivalent 4D computation."""
        from mslk.attention.sparse_attn.gating import fused_gate_and_combine

        dtype = torch.bfloat16
        torch.manual_seed(42)
        Q_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_cmp_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_slc_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_sld_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        gate_proj_weight = torch.randn(H, 3, D, device="cuda", dtype=dtype)

        # Run 3D path
        out_3d = fused_gate_and_combine(
            Q_3d, O_cmp_3d, O_slc_3d, O_sld_3d, gate_proj_weight
        )
        assert out_3d.shape == (T, H, D)

        # Run 4D path (B=1, N=T)
        out_4d = fused_gate_and_combine(
            Q_3d.unsqueeze(0),
            O_cmp_3d.unsqueeze(0),
            O_slc_3d.unsqueeze(0),
            O_sld_3d.unsqueeze(0),
            gate_proj_weight,
        ).squeeze(0)

        max_diff = (out_3d.float() - out_4d.float()).abs().max().item()
        assert max_diff < 1e-5, f"3D/4D max diff: {max_diff}"

    def test_3d_without_gate_weight(self) -> None:
        """3D fused_gate_and_combine matches 4D with uniform gates."""
        from mslk.attention.sparse_attn.gating import fused_gate_and_combine

        T, H, D = 512, 4, 128
        dtype = torch.bfloat16
        Q_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_cmp_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_slc_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_sld_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)

        out_3d = fused_gate_and_combine(Q_3d, O_cmp_3d, O_slc_3d, O_sld_3d)
        out_4d = fused_gate_and_combine(
            Q_3d.unsqueeze(0),
            O_cmp_3d.unsqueeze(0),
            O_slc_3d.unsqueeze(0),
            O_sld_3d.unsqueeze(0),
        ).squeeze(0)

        max_diff = (out_3d.float() - out_4d.float()).abs().max().item()
        assert max_diff < 1e-5, f"3D/4D max diff: {max_diff}"

    def test_3d_compute_gates(self) -> None:
        """3D compute_gates matches 4D."""
        from mslk.attention.sparse_attn.gating import compute_gates

        T, H, D = 512, 4, 128
        dtype = torch.bfloat16
        Q_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        gate_proj_weight = torch.randn(H, 3, D, device="cuda", dtype=dtype)

        gates_3d = compute_gates(Q_3d, gate_proj_weight)
        assert gates_3d.shape == (T, H, 3)

        gates_4d = compute_gates(Q_3d.unsqueeze(0), gate_proj_weight).squeeze(0)
        max_diff = (gates_3d.float() - gates_4d.float()).abs().max().item()
        assert max_diff < 1e-5, f"3D/4D gates max diff: {max_diff}"

    def test_3d_compute_gates_uniform(self) -> None:
        """3D compute_gates with no weight returns uniform 1/3."""
        from mslk.attention.sparse_attn.gating import compute_gates

        T, H, D = 256, 4, 128
        dtype = torch.bfloat16
        Q_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)

        gates = compute_gates(Q_3d)
        assert gates.shape == (T, H, 3)
        expected = torch.ones_like(gates) / 3.0
        assert (gates - expected).abs().max().item() < 1e-6

    def test_3d_gate_and_combine_reference(self) -> None:
        """3D gate_and_combine (PyTorch reference) matches 4D."""
        from mslk.attention.sparse_attn.gating import compute_gates, gate_and_combine

        T, H, D = 512, 4, 128
        dtype = torch.bfloat16
        torch.manual_seed(42)
        Q_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_cmp_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_slc_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_sld_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        gate_proj_weight = torch.randn(H, 3, D, device="cuda", dtype=dtype)

        gates_3d = compute_gates(Q_3d, gate_proj_weight)
        out_3d = gate_and_combine(O_cmp_3d, O_slc_3d, O_sld_3d, gates_3d)
        assert out_3d.shape == (T, H, D)

        gates_4d = compute_gates(Q_3d.unsqueeze(0), gate_proj_weight)
        out_4d = gate_and_combine(
            O_cmp_3d.unsqueeze(0),
            O_slc_3d.unsqueeze(0),
            O_sld_3d.unsqueeze(0),
            gates_4d,
        ).squeeze(0)

        max_diff = (out_3d.float() - out_4d.float()).abs().max().item()
        assert max_diff < 1e-5, f"3D/4D max diff: {max_diff}"
        """3D fused_gating_backward matches 4D."""
        from mslk.attention.sparse_attn.gating import (
            compute_gates,
            fused_gating_backward,
        )

        T, H, D = 512, 4, 128
        dtype = torch.bfloat16
        torch.manual_seed(42)
        Q_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        dO_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_cmp_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_slc_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        O_sld_3d = torch.randn(T, H, D, device="cuda", dtype=dtype)
        gate_proj_weight = torch.randn(H, 3, D, device="cuda", dtype=dtype)

        gates_3d = compute_gates(Q_3d, gate_proj_weight)
        gates_4d = compute_gates(Q_3d.unsqueeze(0), gate_proj_weight)

        dO_cmp_3d, dO_slc_3d, dO_sld_3d, dQ_gate_3d, dW_gate_3d = fused_gating_backward(
            Q_3d,
            dO_3d,
            O_cmp_3d,
            O_slc_3d,
            O_sld_3d,
            gates_3d,
            gate_proj_weight,
        )
        dO_cmp_4d, dO_slc_4d, dO_sld_4d, dQ_gate_4d, dW_gate_4d = fused_gating_backward(
            Q_3d.unsqueeze(0),
            dO_3d.unsqueeze(0),
            O_cmp_3d.unsqueeze(0),
            O_slc_3d.unsqueeze(0),
            O_sld_3d.unsqueeze(0),
            gates_4d,
            gate_proj_weight,
        )

        for name, t3, t4 in [
            ("dO_cmp", dO_cmp_3d, dO_cmp_4d.squeeze(0)),
            ("dO_slc", dO_slc_3d, dO_slc_4d.squeeze(0)),
            ("dO_sld", dO_sld_3d, dO_sld_4d.squeeze(0)),
            ("dQ_gate", dQ_gate_3d, dQ_gate_4d.squeeze(0)),
        ]:
            max_diff = (t3.float() - t4.float()).abs().max().item()
            assert max_diff < 1e-5, f"{name} 3D/4D max diff: {max_diff}"

        dW_diff = (dW_gate_3d.float() - dW_gate_4d.float()).abs().max().item()
        assert dW_diff < 0.01, f"dW_gate 3D/4D max diff: {dW_diff}"
