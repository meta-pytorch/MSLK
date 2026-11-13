# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
from mslk.attention.flydsl import flash_attn_interface as fai
from mslk.attention.flydsl.dropout_philox import draw_dropout_seed, reference_keep_mask
from mslk.attention.flydsl.flash_attn_interface import flydsl_flash_attn_func
from paged_reference import spy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="Requires AMD MI350/gfx950",
)


def _inputs(D, dtype, Sq, Skv, variant):
    torch.manual_seed(37)
    q = torch.randn(1, Sq, 4, D, device="cuda", dtype=dtype)
    k = torch.randn(1, Skv, 2, D, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    kw = {"causal": variant != "noncausal_window", "num_kv_heads": 2}
    if "window" in variant or variant == "combined":
        kw["window_left"] = 31
    if variant in ("scale", "combined"):
        kw["sm_scale"] = 0.11
    if variant in ("bias", "combined"):
        kw["bias"] = torch.randn(1, 4, Sq, Skv, device="cuda", dtype=dtype) * 0.3
    if variant in ("dropout", "combined"):
        kw["dropout_p"] = 0.25
        kw["dropout_seed"] = draw_dropout_seed(q.device)
    if variant == "cross_top_left":
        kw["causal_top_left"] = True
    return q, k, v, kw


def _reference(q, k, v, kw):
    k = k.float().repeat_interleave(2, dim=2)
    v = v.float().repeat_interleave(2, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k)
    scores *= kw.get("sm_scale", q.shape[-1] ** -0.5)
    if "bias" in kw:
        scores += kw["bias"].float()
    Sq, Skv = q.shape[1], k.shape[1]
    offset = 0 if kw.get("causal_top_left") else Skv - Sq
    pos = torch.arange(Sq, device=q.device)[:, None] + offset
    keys = torch.arange(Skv, device=q.device)[None, :]
    keep = torch.ones(Sq, Skv, dtype=torch.bool, device=q.device)
    if kw["causal"]:
        keep &= keys <= pos
    if "window_left" in kw:
        keep &= keys >= pos - kw["window_left"]
    scores.masked_fill_(~keep, -math.inf)
    probs = scores.softmax(-1)
    if "dropout_seed" in kw:
        # The kernel applies the keep bits but not the 1/(1-p) scale.
        B, H = q.shape[0], q.shape[2]
        seed = int(kw["dropout_seed"].item())
        probs *= reference_keep_mask(seed, kw["dropout_p"], B, H, Sq, Skv, q.device)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v), scores.logsumexp(-1)


def _check(q, k, v, kw):
    out, lse = flydsl_flash_attn_func(q, k, v, return_lse=True, **kw)
    ref, ref_lse = _reference(q, k, v, kw)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize(
    "variant",
    [
        "window",
        "noncausal_window",
        "scale",
        "bias",
        "dropout",
        "cross",
        "cross_top_left",
        "combined",
    ],
)
def test_forced_splitk_preserves_dense_options(monkeypatch, D, dtype, variant):
    builds = spy(monkeypatch, fai, "_build_dense")
    Skv = 700 if variant.startswith("cross") else 512
    q, k, v, kw = _inputs(D, dtype, 512, Skv, variant)
    kw["num_kv_splits"] = 4
    _check(q, k, v, kw)
    assert builds and builds[0]["num_kv_splits"] == 4, builds


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "variant,D,Skv,expected_split",
    [
        ("cross", 64, 2047, False),
        ("cross", 64, 2048, True),
        ("dropout", 64, 767, False),
        ("dropout", 64, 768, True),
        ("dropout", 128, 767, False),
        ("dropout", 128, 768, True),
    ],
)
def test_auto_splitk_cross_and_dropout(
    monkeypatch, dtype, variant, D, Skv, expected_split
):
    builds = spy(monkeypatch, fai, "_build_dense")
    q, k, v, kw = _inputs(D, dtype, 16, Skv, variant)
    _check(q, k, v, kw)
    assert builds, "generic route was not exercised"
    assert (builds[0].get("num_kv_splits", 1) > 1) == expected_split, builds


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("splits", [-1, -4])
def test_negative_split_count_rejected(varlen, splits):
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    kw = {}
    if varlen:
        q = q[0]
        cu = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
        kw = {
            "cu_seqlens_q": cu,
            "cu_seqlens_kv": cu,
            "max_seqlen_q": 64,
            "max_seqlen_kv": 64,
            "cross_seqlen": False,
        }
    with pytest.raises(ValueError, match="num_kv_splits must be nonnegative"):
        flydsl_flash_attn_func(q, q, q, num_kv_splits=splits, **kw)


_SEED = torch.zeros(1, dtype=torch.int64)


@pytest.mark.parametrize(
    "modifier,kwargs",
    [
        ("bias", {"bias": (torch.float32, "cuda")}),
        ("bias", {"bias": (torch.float16, "cuda")}),
        ("bias", {"bias": (torch.bfloat16, "cpu")}),
        # Dropout needs both a nonzero dropout_p and a seed tensor.
        ("dropout_p and dropout_seed", {"dropout_p": 0.25}),
        ("dropout_p and dropout_seed", {"dropout_seed": _SEED}),
        ("dropout_p and dropout_seed", {"dropout_p": 0.0, "dropout_seed": _SEED}),
    ],
)
def test_dense_modifier_abi_rejected(monkeypatch, modifier, kwargs):
    def reject_build(*args, **kwargs):
        raise AssertionError("Invalid modifier reached kernel compilation")

    monkeypatch.setattr(fai, "_build_dense", reject_build)
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    kw = {}
    for name, value in kwargs.items():
        if name == "bias":
            dtype, device = value
            value = torch.zeros(1, 4, 64, 64, device=device, dtype=dtype)
        elif isinstance(value, torch.Tensor):
            value = value.to("cuda")
        kw[name] = value
    with pytest.raises(ValueError, match=modifier):
        flydsl_flash_attn_func(q, q, q, **kw)
