# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Large attention logits on the gfx950 dualwave kernels (dense bf16, paged bf16,
dense fp8), against an fp32 reference.

Q is scaled up to grow the logits. The lazy online-softmax rescale used to lower
a row's running max to a smaller tile max, overflowing exp2 to Inf/NaN, and bf16
Q was rounded after being pre-scaled, costing precision. Each case asserts the
dualwave builder served it, so a routing change cannot quietly move it to the
generic kernel.
"""

import math

import pytest
import torch
from mslk.attention.flydsl import flash_attn_interface as fai, flydsl_flash_attn_func
from mslk.flydsl.common import is_flydsl_available
from paged_reference import paged_reference, spy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flydsl_available(),
    reason="requires a ROCm GPU with FlyDSL",
)

PAGE = 64
H, HKV, D = 16, 4, 128
SQ = 512
SCALES = (1.0, 8.0, 32.0, 100.0)
# Max |err| against fp32. The fixed kernels stay under 0.02 at every scale; the
# Q rounding alone cost 0.05-0.08 at 8x, and the NaN shows from 16-32x.
TOL = 4e-2


def _gfx950() -> bool:
    from flydsl.runtime.device import get_rocm_arch

    return get_rocm_arch().startswith("gfx950")


@pytest.fixture(autouse=True)
def _needs_gfx950():
    if not _gfx950():
        pytest.skip("the dualwave kernels are gfx950-only")


def _reference(q, k, v, causal):
    """fp32 attention for [1, Sq, H, D] q and [1, Skv, HKV, D] k/v (bottom-right
    causal), returned as [1, Sq, H, D]."""
    kf = k[0].float().repeat_interleave(H // HKV, dim=1)
    vf = v[0].float().repeat_interleave(H // HKV, dim=1)
    s = torch.einsum("qhd,khd->hqk", q[0].float(), kf) / math.sqrt(D)
    if causal:
        sq, skv = q.shape[1], k.shape[1]
        keep = torch.arange(skv, device=q.device)[None, :] <= (
            torch.arange(sq, device=q.device)[:, None] + (skv - sq)
        )
        s = s.masked_fill(~keep, float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), vf)[None]


def _check(out, ref):
    out = out.float()
    assert torch.isfinite(out).all(), f"{int((~torch.isfinite(out)).sum())} non-finite"
    err = (out - ref).abs().max().item()
    assert err < TOL, err


def _dense_bf16(monkeypatch, causal, scale):
    builds = spy(monkeypatch, fai, "_build_dense_dualwave")
    torch.manual_seed(0)
    q = (torch.randn(1, SQ, H, D, device="cuda") * scale).to(torch.bfloat16)
    k = torch.randn(1, SQ, HKV, D, device="cuda").to(torch.bfloat16)
    v = torch.randn_like(k)
    out = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=HKV)
    assert builds, "not served by the dense dualwave kernel"
    return out, _reference(q, k, v, causal)


def _paged_bf16(monkeypatch, ctx, scale):
    builds = spy(monkeypatch, fai, "_build_paged")
    torch.manual_seed(0)
    pages = -(-ctx // PAGE)
    q = (torch.randn(1, SQ, H, D, device="cuda") * scale).to(torch.bfloat16)
    k = torch.randn(pages, PAGE, HKV, D, device="cuda").to(torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.arange(pages, device="cuda", dtype=torch.int32).view(1, pages)
    seqlen_k = torch.full((1,), ctx, device="cuda", dtype=torch.int32)
    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=False,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=1,
    )
    assert builds, "not served by the paged dualwave kernel"
    return out, paged_reference(q, k, v, block_table, seqlen_k, False)


def _dense_fp8(monkeypatch, causal, scale):
    builds = spy(monkeypatch, fai, "_build_dense_fp8")
    torch.manual_seed(0)
    fp8 = torch.float8_e4m3fn
    q = torch.randn(1, SQ, H, D, device="cuda").to(fp8)
    k = torch.randn(1, SQ, HKV, D, device="cuda").to(fp8)
    v = torch.randn(1, SQ, HKV, D, device="cuda").to(fp8)
    # The logit scale enters through Q's descale, as a real fp8 caller's would.
    q_descale = torch.full((1,), scale, device="cuda", dtype=torch.float32)
    one = torch.ones(1, device="cuda", dtype=torch.float32)
    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        q_descale=q_descale,
        k_descale=one,
        v_descale=one,
    )
    assert builds, "not served by the fp8 dualwave kernel"
    return out, _reference(q.float() * scale, k.float(), v.float(), causal)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("causal", [True, False])
def test_dense_bf16_large_logits(monkeypatch, causal, scale):
    _check(*_dense_bf16(monkeypatch, causal, scale))


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("ctx", [4096, 512])
def test_paged_bf16_large_logits(monkeypatch, ctx, scale):
    _check(*_paged_bf16(monkeypatch, ctx, scale))


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("causal", [True, False])
def test_dense_fp8_large_logits(monkeypatch, causal, scale):
    _check(*_dense_fp8(monkeypatch, causal, scale))
