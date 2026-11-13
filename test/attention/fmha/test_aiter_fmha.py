# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Correctness tests for the AITER fmha FwOp/BwOp.

Skipped when AITER is unavailable or the device is not gfx950.
"""

import pytest
import torch

from mslk.attention import fmha
from mslk.attention.fmha import rocm_aiter as aiter_fmha
from mslk.attention.fmha.attn_bias_utils import ref_attention_bmhk


def _aiter_available() -> bool:
    try:
        from mslk.aiter.common import is_aiter_available

        return is_aiter_available()
    except Exception:
        return False


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        return "gfx950" in props.gcnArchName
    except Exception:
        return False


skip_no_aiter_gfx950 = pytest.mark.skipif(
    not (_aiter_available() and _is_gfx950()),
    reason="requires AITER on gfx950",
)


@skip_no_aiter_gfx950
class TestAiterFmhaForward:
    """Forward correctness: AITER vs. reference attention."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize(
        "B,Mq,Mkv,H,causal",
        [
            (1, 32, 32, 4, False),
            (2, 64, 64, 8, False),
            (1, 32, 32, 4, True),
            (2, 64, 64, 8, True),
        ],
    )
    def test_forward(self, dtype, head_dim, causal, B, Mq, Mkv, H):
        device = "cuda"
        query = torch.randn(B, Mq, H, head_dim, device=device, dtype=dtype)
        key = torch.randn(B, Mkv, H, head_dim, device=device, dtype=dtype)
        value = torch.randn(B, Mkv, H, head_dim, device=device, dtype=dtype)

        bias = fmha.LowerTriangularMask() if causal else None

        out = fmha.memory_efficient_attention_forward(
            query, key, value, attn_bias=bias, op=aiter_fmha.FwOp
        )

        ref = ref_attention_bmhk(query, key, value, attn_bias=bias)

        atol = aiter_fmha.FwOp.ERROR_ATOL[dtype]
        rtol = aiter_fmha.FwOp.ERROR_RTOL[dtype]
        torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)


@skip_no_aiter_gfx950
class TestAiterFmhaBackward:
    """Backward: uses saved out/lse directly, no redundant forward."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize(
        "B,Mq,Mkv,H",
        [
            (1, 32, 32, 4),
            (2, 64, 64, 8),
        ],
    )
    def test_backward(self, dtype, head_dim, B, Mq, Mkv, H):
        device = "cuda"
        query = torch.randn(
            B, Mq, H, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        key = torch.randn(
            B, Mkv, H, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        value = torch.randn(
            B, Mkv, H, head_dim, device=device, dtype=dtype, requires_grad=True
        )

        out = fmha.memory_efficient_attention(
            query, key, value, op=(aiter_fmha.FwOp, aiter_fmha.BwOp)
        )

        grad = torch.randn_like(out)
        out.backward(grad)

        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        assert query.grad.shape == query.shape


@skip_no_aiter_gfx950
class TestAiterFmhaNotSupported:
    """Verify not_supported_reasons gates correctly."""

    def test_fw_rejects_dropout(self):
        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(query=q, key=k, value=v, p=0.1)
        reasons = aiter_fmha.FwOp.not_supported_reasons(inp)
        assert any("dropout" in r.lower() for r in reasons)

    def test_fw_rejects_causal_cross_attn(self):
        """LowerTriangularMask with Lq != Lkv should be declined."""
        q = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(
            query=q, key=k, value=v,
            attn_bias=fmha.LowerTriangularMask(),
        )
        reasons = aiter_fmha.FwOp.not_supported_reasons(inp)
        assert any("top-left" in r or "Lq != Lkv" in r for r in reasons), reasons

    def test_fw_rejects_packed_varlen(self):
        """BlockDiagonalCausalMask (packed varlen) should be declined."""
        q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(
            query=q, key=k, value=v,
            attn_bias=fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens([32, 32]),
        )
        reasons = aiter_fmha.FwOp.not_supported_reasons(inp)
        assert any("bias type" in r.lower() for r in reasons), reasons

    def test_fw_not_in_auto_dispatch(self):
        """AITER should not be auto-selected; only via explicit op=."""
        from mslk.attention.fmha.dispatch import _dispatch_fw_priority_list
        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(query=q, key=k, value=v)
        priority = _dispatch_fw_priority_list(inp, needs_gradient=False)
        assert aiter_fmha.FwOp not in priority, (
            "AITER should not be in auto-dispatch priority list"
        )

    def test_bw_rejects_dropout(self):
        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(query=q, key=k, value=v, p=0.1)
        reasons = aiter_fmha.BwOp.not_supported_reasons(inp)
        assert any("dropout" in r.lower() for r in reasons)
