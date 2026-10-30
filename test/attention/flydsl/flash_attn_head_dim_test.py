# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Head dims the kernels do not build natively (e.g. 72, 96, 160, 192, 224) run
on the generic kernel at the next of 64/128/256, which reads Q/K/V at their own
head dim and masks the missing columns: nothing is padded or copied, and a paged
cache is read in place. Checked against fp32."""

import math

import pytest
import torch
from mslk.attention.flydsl import flydsl_flash_attn_func
from mslk.flydsl.common import is_flydsl_available
from paged_reference import paged_reference, spy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flydsl_available(),
    reason="requires a ROCm GPU with FlyDSL",
)

PAGE = 64


def _reference(q, k, v, causal, scale=None):
    """fp32 attention for [B, Sq, H, D] q and [B, Skv, Hkv, D] k/v, bottom-right
    causal; returns ([B, Sq, H, D], natural-log LSE [B, H, Sq])."""
    group = q.shape[2] // k.shape[2]
    kf = k.float().repeat_interleave(group, dim=2)
    vf = v.float().repeat_interleave(group, dim=2)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kf) * scale
    if causal:
        sq, skv = q.shape[1], k.shape[1]
        keep = torch.arange(skv, device=q.device)[None, :] <= (
            torch.arange(sq, device=q.device)[:, None] + (skv - sq)
        )
        s = s.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    o = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, dim=-1), vf)
    return o, lse


def _close(out, ref):
    err = (out.float() - ref).abs().max().item()
    assert math.isfinite(err) and err < 2e-2 * max(1.0, ref.abs().max().item()), err


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("D", [96, 160, 192, 224])
@pytest.mark.parametrize("S,causal", [(128, True), (512, True), (512, False)])
def test_dense_masked_head_dim(dtype, D, S, causal):
    torch.manual_seed(0)
    q = torch.randn(2, S, 8, D, device="cuda", dtype=dtype)
    k = torch.randn(2, S, 2, D, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    out = flydsl_flash_attn_func(q, k, v, causal=causal, num_kv_heads=2)
    assert out.shape == q.shape and out.is_contiguous()
    _close(out, _reference(q, k, v, causal)[0])


def test_dense_masked_head_dim_lse_scale_and_out():
    """LSE, a custom scale and a caller's ``out`` survive the masking."""
    torch.manual_seed(0)
    q = torch.randn(2, 200, 8, 160, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 300, 8, 160, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    res, lse = flydsl_flash_attn_func(
        q, k, v, causal=True, sm_scale=0.05, return_lse=True, out=out
    )
    assert res is out
    ref, ref_lse = _reference(q, k, v, True, scale=0.05)
    _close(out, ref)
    assert (lse - ref_lse).abs().max().item() < 1e-2


def test_dense_masked_head_dim_side_stream():
    """A caller's noncurrent stream runs the masked head dim on that stream."""
    torch.manual_seed(0)
    q = torch.randn(2, 512, 8, 96, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    out = flydsl_flash_attn_func(q, k, v, causal=True, stream=side)
    # Churn the current stream's allocator while the side stream may still run.
    junk = [torch.full_like(q, float("nan")) for _ in range(4)]
    torch.cuda.current_stream().wait_stream(side)
    del junk
    _close(out, _reference(q, k, v, True)[0])


def test_varlen_masked_head_dim():
    torch.manual_seed(0)
    lens = [1, 31, 65, 300]
    D = 160
    q = torch.randn(sum(lens), 8, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lens), 2, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu = torch.tensor([0] + lens, device="cuda", dtype=torch.int32).cumsum(0)
    cu = cu.to(torch.int32)
    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=2,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max(lens),
        max_seqlen_kv=max(lens),
        cross_seqlen=False,
    )
    for b, n in enumerate(lens):
        s = slice(int(cu[b]), int(cu[b + 1]))
        ref = _reference(q[None, s], k[None, s], v[None, s], True)[0][0]
        _close(out[s], ref)


@pytest.mark.parametrize("D", [96, 72])
@pytest.mark.parametrize(
    "B,Sq,H,HKV,ctx,causal,splits",
    [
        (2, 1, 32, 4, 4096, True, 0),  # GQA-packed decode
        (1, 1, 32, 4, 32768, True, 0),  # starved grid, auto split-K
        (3, 16, 32, 4, 4096, True, 0),
        (2, 300, 16, 4, 2048, False, 0),  # dualwave territory at native D
        (2, 4, 32, 4, 32768, True, 4),  # explicit split-K
        (2, 1, 8, 8, 4096, True, 0),  # MHA
        (8, 4, 16, 16, 4096, True, 4),  # fused split-K combine
    ],
)
def test_paged_masked_head_dim_reads_cache_in_place(
    monkeypatch, D, B, Sq, H, HKV, ctx, causal, splits
):
    """Paged non-native D: the generic kernel computes at 128 and reads Q and the
    cache at D (head_dim_qk), so nothing is copied out to a padded one."""
    from mslk.attention.flydsl import flash_attn_interface as fai

    builds = spy(monkeypatch, fai, "_build_paged_light")
    torch.manual_seed(0)
    pages = ctx // PAGE
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16) * 2
    k = torch.randn(B * pages, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.randperm(B * pages, device="cuda", dtype=torch.int32).view(
        B, pages
    )
    lens = [ctx, max(Sq, 65), ctx // 2 + 3]
    seqlen_k = torch.tensor(
        [max(Sq, lens[b % len(lens)]) for b in range(B)],
        device="cuda",
        dtype=torch.int32,
    )
    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        max_seqlen_kv=ctx,
        num_kv_splits=splits,
    )
    assert builds and all(
        c.get("head_dim") == 128 and c.get("head_dim_qk") == D for c in builds
    ), builds
    if B == 8:
        assert builds[0]["fused_combine"], builds
    assert out.shape == q.shape
    _close(out, paged_reference(q, k, v, block_table, seqlen_k, causal))


@pytest.mark.parametrize(
    "kw,D,exc",
    [
        ({}, 81, ValueError),  # odd
        ({}, 288, ValueError),  # above 256
        ({"debug_counts": True}, 96, NotImplementedError),  # dualwave-only
        ({"paged": "linear"}, 192, NotImplementedError),  # paged runs D <= 128
        ({"paged": "vectorized"}, 96, NotImplementedError),
    ],
)
def test_head_dim_rejections(kw, D, exc):
    q = torch.randn(1, 4, 8, D, device="cuda", dtype=torch.bfloat16)
    if "paged" not in kw:
        extra = {}
        if kw.get("debug_counts"):
            extra["debug_counts"] = torch.zeros(2, device="cuda")
        with pytest.raises(exc):
            flydsl_flash_attn_func(q, q, q, causal=True, **extra)
        return
    if kw["paged"] == "vectorized":
        kvs = 8
        k = torch.randn(4, 2, D // kvs, PAGE, kvs, device="cuda", dtype=q.dtype)
        v = torch.randn(4, 2, PAGE // kvs, D, kvs, device="cuda", dtype=q.dtype)
    else:
        k = torch.randn(4, PAGE, 2, D, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
    with pytest.raises(exc):
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=2,
            block_table=torch.arange(4, device="cuda", dtype=torch.int32).view(1, 4),
            seqlen_k=torch.tensor([200], device="cuda", dtype=torch.int32),
            kv_cache_layout=kw["paged"],
        )
