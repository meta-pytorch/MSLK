# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from mslk.attention import fmha
from mslk.attention.fmha import flydsl

from .utils import rocm_only


def _bias_and_lengths(mask):
    if mask == "none":
        return None, (2, 150, 150)
    if mask == "causal":
        return fmha.attn_bias.LowerTriangularMask(), (2, 150, 150)
    assert mask == "varlen_causal"
    lengths = [60, 100, 45]
    return fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(lengths), (
        1,
        sum(lengths),
        sum(lengths),
    )


def _flydsl_grads(q, k, v, grad, bias, **kw):
    q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
    out = fmha.memory_efficient_attention(
        q, k, v, bias, op=(flydsl.FwOp, flydsl.BwOp), **kw
    )
    out.backward(grad)
    return out, q.grad, k.grad, v.grad


def _gfx950_only():
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("head-dim masking in flydsl.BwOp is gfx950-only")


@rocm_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mask", ["none", "causal", "varlen_causal"])
@pytest.mark.parametrize(
    "K,Kv", [(96, 96), (40, 40), (64, 128), (128, 64), (200, 160), (72, 40), (3, 5)]
)
@pytest.mark.parametrize("mqa", [False, True], ids=["mha", "mqa"])
def test_backward_flydsl_head_dims(K, Kv, mask, dtype, mqa):
    # Non-native head dims and Kv != K train without padded copies.
    _gfx950_only()
    torch.manual_seed(0)
    H = 4
    bias, (B, Mq, Mkv) = _bias_and_lengths(mask)
    q = torch.randn(B, Mq, H, K, device="cuda", dtype=dtype)
    grad = torch.randn(B, Mq, H, Kv, device="cuda", dtype=dtype)
    k, v = (
        torch.randn(B, Mkv, 1 if mqa else H, d, device="cuda", dtype=dtype)
        for d in (K, Kv)
    )
    q, k, v = (t.requires_grad_(True) for t in (q, k, v))
    out = fmha.memory_efficient_attention(
        q,
        k.expand(-1, -1, H, -1),
        v.expand(-1, -1, H, -1),
        bias,
        op=(flydsl.FwOp, flydsl.BwOp),
    )
    out.backward(grad)

    qr, kr, vr = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    scores = torch.einsum("bqhd,bkhd->bhqk", qr, kr.expand(-1, -1, H, -1)) * K**-0.5
    if isinstance(bias, torch.Tensor):
        scores = scores + bias.float()
    elif bias is not None:
        scores = scores + bias.materialize(
            (B, H, Mq, Mkv), device="cuda", dtype=torch.float32
        )
    ref = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), vr.expand(-1, -1, H, -1))
    ref.backward(grad.float())
    tol = 2e-2 if dtype is torch.bfloat16 else 5e-3
    for name, got, want in (
        ("out", out, ref),
        ("dq", q.grad, qr.grad),
        ("dk", k.grad, kr.grad),
        ("dv", v.grad, vr.grad),
    ):
        err = float((got.float() - want).norm() / want.norm())
        assert err < tol, f"{name}: relative L2 {err:.3e} exceeds {tol}"


@rocm_only
@pytest.mark.parametrize("K,Kv", [(72, 72), (64, 128), (72, 40)])
def test_backward_flydsl_head_dims_match_padding(K, Kv):
    # Masking in-kernel matches zero-padding to the compute width.
    _gfx950_only()
    torch.manual_seed(0)
    q, k = (torch.randn(2, 300, 4, K, device="cuda") for _ in range(2))
    v, grad = (torch.randn(2, 300, 4, Kv, device="cuda") for _ in range(2))
    q, k, v, grad = (t.bfloat16() for t in (q, k, v, grad))
    bias = fmha.attn_bias.LowerTriangularMask()
    got = _flydsl_grads(q, k, v, grad, bias)

    def pad(t):
        return torch.nn.functional.pad(t, (0, 128 - t.shape[-1]))

    want = _flydsl_grads(pad(q), pad(k), pad(v), pad(grad), bias, scale=K**-0.5)
    for name, g, w in zip(("out", "dq", "dk", "dv"), got, want):
        w = w[..., : g.shape[-1]]
        # dQ accumulates with atomics, so its summation order may differ.
        err = float((g.float() - w.float()).norm() / w.float().norm())
        assert err < 1e-4, f"{name}: relative L2 {err:.3e}"
