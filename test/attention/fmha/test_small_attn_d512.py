# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for D=512 small-attention forward and backward kernels on gfx950."""

import math

import pytest
import torch

from mslk.attention import fmha

_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
_HAS_CUDA = torch.cuda.is_available()


def _is_gfx950() -> bool:
    if not _HAS_CUDA:
        return False
    try:
        return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


def _flydsl_available() -> bool:
    try:
        return fmha.flydsl._is_flydsl_available()
    except Exception:
        return False


_SKIP_REASON = "requires ROCm + gfx950 + FlyDSL"
_SKIP = not (_ROCM and _HAS_CUDA and _is_gfx950() and _flydsl_available())


def _ref_attention(q, k, v, scale, causal=False):
    """PyTorch reference attention for correctness comparison."""
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        Lq, Lkv = s.shape[-2], s.shape[-1]
        mask = torch.triu(torch.ones(Lq, Lkv, device=s.device, dtype=torch.bool), diagonal=1)
        s.masked_fill_(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v.float())
    lse = torch.logsumexp(s, dim=-1)
    return o, lse, p


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestSmallAttnForwardD512:
    """Forward correctness for D=512 small-attention kernel."""

    @pytest.mark.parametrize(
        "B,H,Lq,Lkv",
        [
            (4, 1, 16, 25),   # W3 shape
            (1, 4, 32, 32),
            (2, 2, 16, 16),
            (1, 1, 1, 1),     # single token
            (8, 1, 8, 8),
        ],
    )
    def test_forward_noncausal(self, B, H, Lq, Lkv):
        D = 512
        dtype = torch.bfloat16
        device = "cuda"
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, Lq, H, D, dtype=dtype, device=device)
        k = torch.randn(B, Lkv, H, D, dtype=dtype, device=device)
        v = torch.randn(B, Lkv, H, D, dtype=dtype, device=device)

        # Run through the op interface
        inp = fmha.Inputs(query=q, key=k, value=v, scale=scale)
        reasons = fmha.flydsl.FwOp.not_supported_reasons(inp)
        if reasons:
            pytest.skip(f"FwOp declined: {reasons}")

        out, ctx = fmha.flydsl.FwOp.apply(inp, needs_gradient=True)

        # Reference
        q_bhmd = q.permute(0, 2, 1, 3)  # [B, H, Lq, D]
        k_bhmd = k.permute(0, 2, 1, 3)
        v_bhmd = v.permute(0, 2, 1, 3)
        ref_o, ref_lse, _ = _ref_attention(q_bhmd, k_bhmd, v_bhmd, scale)
        ref_o = ref_o.permute(0, 2, 1, 3)  # back to [B, Lq, H, D]

        torch.testing.assert_close(
            out.float(), ref_o.to(out.dtype).float(),
            atol=3e-2, rtol=2e-2,
        )

    @pytest.mark.parametrize(
        "B,H,Lq,Lkv",
        [
            (4, 1, 16, 25),
            (1, 2, 16, 16),
        ],
    )
    def test_forward_causal(self, B, H, Lq, Lkv):
        D = 512
        dtype = torch.bfloat16
        device = "cuda"
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, Lq, H, D, dtype=dtype, device=device)
        k = torch.randn(B, Lkv, H, D, dtype=dtype, device=device)
        v = torch.randn(B, Lkv, H, D, dtype=dtype, device=device)

        inp = fmha.Inputs(
            query=q, key=k, value=v, scale=scale,
            attn_bias=fmha.attn_bias.LowerTriangularMask(),
        )
        reasons = fmha.flydsl.FwOp.not_supported_reasons(inp)
        if reasons:
            pytest.skip(f"FwOp declined: {reasons}")

        out, ctx = fmha.flydsl.FwOp.apply(inp, needs_gradient=True)

        q_bhmd = q.permute(0, 2, 1, 3)
        k_bhmd = k.permute(0, 2, 1, 3)
        v_bhmd = v.permute(0, 2, 1, 3)
        ref_o, _, _ = _ref_attention(q_bhmd, k_bhmd, v_bhmd, scale, causal=True)
        ref_o = ref_o.permute(0, 2, 1, 3)

        torch.testing.assert_close(
            out.float(), ref_o.to(out.dtype).float(),
            atol=3e-2, rtol=2e-2,
        )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestSmallAttnBackwardD512:
    """Backward correctness for D=512 small-attention kernel."""

    @pytest.mark.parametrize(
        "B,H,Lq,Lkv",
        [
            (4, 1, 16, 25),   # W3 shape
            (1, 2, 16, 16),
            (2, 1, 8, 8),
        ],
    )
    def test_backward_noncausal(self, B, H, Lq, Lkv):
        D = 512
        dtype = torch.bfloat16
        device = "cuda"
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, Lq, H, D, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(B, Lkv, H, D, dtype=dtype, device=device, requires_grad=True)
        v = torch.randn(B, Lkv, H, D, dtype=dtype, device=device, requires_grad=True)

        inp = fmha.Inputs(query=q, key=k, value=v, scale=scale)

        fw_reasons = fmha.flydsl.FwOp.not_supported_reasons(inp)
        bw_reasons = fmha.flydsl.BwOp.not_supported_reasons(inp)
        if fw_reasons or bw_reasons:
            pytest.skip(f"Declined: fw={fw_reasons}, bw={bw_reasons}")

        out_fwd, ctx = fmha.flydsl.FwOp.apply(inp, needs_gradient=True)
        grad_out = torch.randn_like(out_fwd)
        grads = fmha.flydsl.BwOp.apply(ctx, inp, grad_out)

        # Reference backward via autograd
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        q_bhmd = q_ref.permute(0, 2, 1, 3)
        k_bhmd = k_ref.permute(0, 2, 1, 3)
        v_bhmd = v_ref.permute(0, 2, 1, 3)
        ref_o, _, _ = _ref_attention(q_bhmd, k_bhmd, v_bhmd, scale)
        ref_o = ref_o.permute(0, 2, 1, 3)
        ref_o.backward(grad_out.float())

        atol = 5e-2
        rtol = 3e-2

        torch.testing.assert_close(
            grads.dq.float(), q_ref.grad.to(dtype).float(),
            atol=atol, rtol=rtol, msg="dQ mismatch",
        )
        torch.testing.assert_close(
            grads.dk.float(), k_ref.grad.to(dtype).float(),
            atol=atol, rtol=rtol, msg="dK mismatch",
        )
        torch.testing.assert_close(
            grads.dv.float(), v_ref.grad.to(dtype).float(),
            atol=atol, rtol=rtol, msg="dV mismatch",
        )

    def test_not_supported_large_seq(self):
        """D=512 with large sequences should be declined by not_supported_reasons."""
        D = 512
        Lq = 200
        Lkv = 200
        q = torch.randn(1, Lq, 1, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, Lkv, 1, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, Lkv, 1, D, dtype=torch.bfloat16, device="cuda")

        inp = fmha.Inputs(query=q, key=k, value=v)
        fw_reasons = fmha.flydsl.FwOp.not_supported_reasons(inp)
        bw_reasons = fmha.flydsl.BwOp.not_supported_reasons(inp)
        assert any("LDS budget" in r for r in fw_reasons), (
            f"Expected LDS budget rejection for Lq={Lq}, Lkv={Lkv}, D={D}, "
            f"got: {fw_reasons}"
        )
        assert any("LDS budget" in r for r in bw_reasons), (
            f"Expected LDS budget rejection for backward, got: {bw_reasons}"
        )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_round_trip_d512():
    """End-to-end forward + backward through memory_efficient_attention API."""
    B, H, Lq, Lkv, D = 4, 1, 16, 25, 512
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(B, Lq, H, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, Lkv, H, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, Lkv, H, D, dtype=dtype, device=device, requires_grad=True)

    op = (fmha.flydsl.FwOp, fmha.flydsl.BwOp)
    try:
        out = fmha.memory_efficient_attention(q, k, v, op=op)
    except NotImplementedError as e:
        pytest.skip(f"Op not supported: {e}")

    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape
