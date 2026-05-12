# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_gates_with_weight():
    """Gates are in [0, 1] range (sigmoid output)."""
    from mslk.attention.sparse_attn.gating import compute_gates

    B, N, H, D = 1, 256, 4, 128
    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(H, 3, D, device="cuda", dtype=torch.bfloat16)

    gates = compute_gates(Q, W)
    assert gates.shape == (B, N, H, 3)
    assert (gates >= 0).all() and (gates <= 1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_gates_no_weight():
    """Without gate weights, gates are uniform 1/3."""
    from mslk.attention.sparse_attn.gating import compute_gates

    B, N, H, D = 1, 256, 4, 128
    Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)

    gates = compute_gates(Q, None)
    assert gates.shape == (B, N, H, 3)
    torch.testing.assert_close(
        gates.float().mean(), torch.tensor(1.0 / 3.0), atol=1e-5, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gate_and_combine():
    """Gate and combine with uniform gates produces average."""
    from mslk.attention.sparse_attn.gating import gate_and_combine

    B, N, H, D = 1, 256, 4, 128
    O_cmp = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    O_slc = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    O_sld = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    gates = torch.ones(B, N, H, 3, device="cuda", dtype=torch.bfloat16) / 3.0

    O = gate_and_combine(O_cmp, O_slc, O_sld, gates)
    expected = (O_cmp.float() + O_slc.float() + O_sld.float()) / 3.0

    torch.testing.assert_close(
        O.float(), expected.to(O.dtype).float(), atol=1e-2, rtol=1e-2
    )
