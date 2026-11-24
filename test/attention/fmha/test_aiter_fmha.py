# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AITER fmha rejection and routing tests.

Correctness is covered by the generic fmha test suite (test_forward.py,
test_backward.py) which exercises aiterF/aiterB via ALL_FW_OPS/ALL_BW_OPS.
"""

import pytest
import torch

from mslk.attention import fmha
from mslk.attention.fmha import rocm_aiter as aiter_fmha


def _aiter_available() -> bool:
    try:
        from mslk.aiter.common import is_aiter_available

        return is_aiter_available()
    except Exception:
        return False


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "gcnArchName", "").startswith("gfx950")


skip_no_aiter_gfx950 = pytest.mark.skipif(
    not (torch.cuda.is_available() and _is_gfx950() and _aiter_available()),
    reason="requires AITER on gfx950",
)


@skip_no_aiter_gfx950
class TestAiterFmhaNotSupported:
    """Verify not_supported_reasons gates correctly."""

    def test_fw_rejects_dropout(self):
        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(query=q, key=q, value=q, p=0.1)
        reasons = aiter_fmha.FwOp.not_supported_reasons(inp)
        assert any("dropout" in r.lower() for r in reasons)

    def test_fw_rejects_causal_cross_attn(self):
        q = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(
            query=q,
            key=k,
            value=v,
            attn_bias=fmha.LowerTriangularMask(),
        )
        reasons = aiter_fmha.FwOp.not_supported_reasons(inp)
        assert any("top-left" in r or "Lq != Lkv" in r for r in reasons), reasons

    def test_fw_rejects_packed_varlen(self):
        q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(
            query=q,
            key=q,
            value=q,
            attn_bias=fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                [32, 32]
            ),
        )
        reasons = aiter_fmha.FwOp.not_supported_reasons(inp)
        assert any("bias type" in r.lower() for r in reasons), reasons

    def test_fw_dispatch_empty_table(self):
        """Empty tuning table: AITER should not appear in auto-dispatch."""
        from mslk.attention.fmha.dispatch import _dispatch_fw_priority_list

        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(query=q, key=q, value=q)
        priority = _dispatch_fw_priority_list(inp, needs_gradient=False)
        assert aiter_fmha.FwOp not in priority, (
            "AITER should not be in auto-dispatch with an empty tuning table"
        )

    def test_bw_rejects_dropout(self):
        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
        inp = fmha.Inputs(query=q, key=q, value=q, p=0.1)
        reasons = aiter_fmha.BwOp.not_supported_reasons(inp)
        assert any("dropout" in r.lower() for r in reasons)
