# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlyDSL-vs-CK parity sweep and exported forward/backward end-to-end test.

This is the reproducible source behind the PR's parity claim: it force-compares
``flydsl.FwOp`` against ``ck.FwOp`` (the operator it drops in for) across a
matrix of dtype / causal / shape / head-dim / bias-type, checking the forward
output and (when gradients are requested) the packed LSE. Cases the FlyDSL op
declines are skipped, so the effective case count is the supported subset of the
full matrix.

Run just this sweep with::

    pytest test/attention/fmha/test_flydsl_parity.py -q
"""

from typing import List, Tuple

import pytest
import torch
from mslk.attention import fmha
from mslk.attention.fmha import flydsl

from .utils import assert_allclose, rocm_only

_DTYPES = [torch.float16, torch.bfloat16]
_HEAD_DIMS = [64, 96, 128, 256]  # 96 exercises head-dim padding.
# (B, Mq, Mkv, H) — self-attention (Mq == Mkv) plus a couple of cross-length shapes.
_SELF_SHAPES: List[Tuple[int, int, int, int]] = [
    (1, 32, 32, 1),
    (1, 128, 128, 3),
    (2, 256, 256, 4),
    (1, 512, 512, 8),
    (3, 384, 384, 2),
]
_BIAS = ["none", "causal"]


def _make_bias(kind: str):
    if kind == "none":
        return None
    if kind == "causal":
        return fmha.attn_bias.LowerTriangularMask()
    raise ValueError(kind)


@rocm_only
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
@pytest.mark.parametrize("shape", _SELF_SHAPES, ids=[str(s) for s in _SELF_SHAPES])
@pytest.mark.parametrize("bias_kind", _BIAS)
def test_flydsl_matches_ck(dtype, head_dim, shape, bias_kind):
    if not flydsl.FwOp.is_available():
        pytest.skip("flydslF unavailable")
    if not fmha.ck.FwOp.is_available():
        pytest.skip("ck.FwOp unavailable")

    device = "cuda"
    B, Mq, Mkv, H = shape
    attn_bias = _make_bias(bias_kind)

    q = torch.randn((B, Mq, H, head_dim), device=device, dtype=dtype)
    k = torch.randn((B, Mkv, H, head_dim), device=device, dtype=dtype)
    v = torch.randn((B, Mkv, H, head_dim), device=device, dtype=dtype)

    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=attn_bias)
    if not flydsl.FwOp.supports(inp):
        pytest.skip("flydslF declined this case")
    if not fmha.ck.FwOp.supports(inp):
        pytest.skip("ck.FwOp declined this case")

    out_fly = fmha.memory_efficient_attention_forward(
        q, k, v, attn_bias, op=flydsl.FwOp
    )
    out_ck = fmha.memory_efficient_attention_forward(
        q, k, v, attn_bias, op=fmha.ck.FwOp
    )
    # Both are MFMA f32-accumulate kernels; compare at the op's own tolerance.
    atol = flydsl.FwOp.ERROR_ATOL[dtype]
    rtol = flydsl.FwOp.ERROR_RTOL[dtype]
    assert_allclose(out_fly.float(), out_ck.float(), "fwd parity", atol=atol, rtol=rtol)


@rocm_only
@pytest.mark.parametrize("dtype", _DTYPES)
def test_flydsl_varlen_exported_fw_bw(dtype):
    """End-to-end forward/backward for the exported varlen pair (item: packed LSE).

    Multi-sequence BlockDiagonalCausalMask with gradients: flydslF forward emits a
    packed LSE (VARLEN_LSE_PACKED=True) and dispatch must pick a backward operator
    (ck.BwOp). Compare against ck.FwOp+ck.BwOp end to end.
    """
    if not flydsl.FwOp.is_available() or not fmha.ck.FwOp.is_available():
        pytest.skip("flydslF / ck.FwOp unavailable")
    if not fmha.ck.BwOp.is_available():
        pytest.skip("ck.BwOp unavailable")

    device = "cuda"
    seqlens = [16, 48, 32]
    H, D = 2, 128
    total = sum(seqlens)

    attn_bias = fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(seqlens)

    def mk(requires_grad: bool):
        torch.manual_seed(0)
        t = [
            torch.randn(
                (1, total, H, D),
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )
            for _ in range(3)
        ]
        return t

    q, k, v = mk(True)
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=attn_bias)
    if not flydsl.FwOp.supports(inp):
        pytest.skip("flydslF declined this varlen case")

    out_fly, ctx_fly = flydsl.FwOp.apply(inp, needs_gradient=True)
    # Packed LSE: [1, H, total_q].
    assert ctx_fly.lse.shape == (1, H, total), ctx_fly.lse.shape
    assert flydsl.FwOp.VARLEN_LSE_PACKED is True

    out_ck, ctx_ck = fmha.ck.FwOp.apply(inp, needs_gradient=True)
    assert ctx_ck.lse.shape == ctx_fly.lse.shape
    assert_allclose(out_fly.float(), out_ck.float(), "varlen fwd", atol=2.8e-2)
    assert_allclose(
        ctx_fly.lse.float(), ctx_ck.lse.float(), "varlen lse", atol=2e-4, rtol=2e-4
    )
