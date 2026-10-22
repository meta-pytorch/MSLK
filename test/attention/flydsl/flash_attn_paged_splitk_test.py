# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Automatic paged split-K selection.

Decode shapes (short Q against a long paged cache) walk the whole context as
one serial chain per workgroup. ``_paged_num_kv_splits`` recovers that
parallelism along KV. These tests pin the two properties that matter:
splitting must not change the result, and the split count must stay within
the chain, grid and workspace bounds.
"""

import math

import pytest
import torch
from mslk.attention.flydsl import flash_attn_interface as fai
from mslk.attention.flydsl.flash_attn_interface import flydsl_flash_attn_func
from mslk.flydsl.common import is_flydsl_available

PAGE = 64  # _PAGED_PAGE_SIZE: the only page size the native paged kernel takes
H, HKV = 32, 4


def test_target_requires_gfx950_and_flydsl():
    assert torch.version.hip, "requires a ROCm build"
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch == "gfx950", f"requires gfx950, got {arch}"
    assert is_flydsl_available(), "requires FlyDSL"


def _paged_inputs(B, Sq, ctx, D, device="cuda", dtype=torch.bfloat16):
    pages_per_req = ctx // PAGE
    total = B * pages_per_req
    torch.manual_seed(0)
    q = torch.randn(B, Sq, H, D, device=device, dtype=dtype)
    k = torch.randn(total, PAGE, HKV, D, device=device, dtype=dtype)
    v = torch.randn_like(k)
    block_table = torch.arange(total, device=device, dtype=torch.int32).view(
        B, pages_per_req
    )
    seqlen_k = torch.full((B,), ctx, device=device, dtype=torch.int32)
    return q, k, v, block_table, seqlen_k


def _run(q, k, v, block_table, seqlen_k, num_kv_splits):
    return flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=num_kv_splits,
    )


def _run_single_pass(monkeypatch, *args):
    """Reference with auto-sizing pinned to one split (1 alone auto-sizes)."""
    with monkeypatch.context() as m:
        m.setattr(fai, "_paged_num_kv_splits", lambda *a, **kw: 1)
        m.setattr(fai, "_paged_window_num_kv_splits", lambda *a, **kw: 1)
        return _run(*args, num_kv_splits=1)


def _windowed_reference(q, k, v, left, right, num_kv_heads):
    batch, seqlen_q, num_heads, head_dim = q.shape
    seqlen_k = k.shape[0] * k.shape[1]
    k = k.reshape(seqlen_k, num_kv_heads, head_dim).repeat_interleave(
        num_heads // num_kv_heads, dim=1
    )
    v = v.reshape(seqlen_k, num_kv_heads, head_dim).repeat_interleave(
        num_heads // num_kv_heads, dim=1
    )
    scores = torch.einsum("bthd,khd->bhtk", q.float(), k.float()) / math.sqrt(head_dim)
    q_pos = torch.arange(seqlen_q, device=q.device) + seqlen_k - seqlen_q
    k_pos = torch.arange(seqlen_k, device=q.device)
    mask = (k_pos[None, :] >= q_pos[:, None] - left) & (
        k_pos[None, :] <= torch.minimum(q_pos[:, None] + right, q_pos[:, None])
    )
    probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    return torch.einsum("bhtk,khd->bthd", probs, v.float()).reshape(
        batch, seqlen_q, num_heads, head_dim
    )


def _paged_reference(q, k, v, block_table, lengths, causal):
    batch, seqlen_q, num_heads, head_dim = q.shape
    outputs = []
    for b in range(batch):
        seqlen_k = int(lengths[b])
        if seqlen_k == 0:
            outputs.append(torch.zeros_like(q[b], dtype=torch.float32))
            continue
        kb = k[block_table[b]].reshape(-1, HKV, head_dim)[:seqlen_k].float()
        vb = v[block_table[b]].reshape(-1, HKV, head_dim)[:seqlen_k].float()
        kb = kb.repeat_interleave(num_heads // HKV, dim=1)
        vb = vb.repeat_interleave(num_heads // HKV, dim=1)
        scores = torch.einsum("thd,khd->htk", q[b].float(), kb) / math.sqrt(head_dim)
        if causal:
            q_pos = torch.arange(seqlen_q, device=q.device) + seqlen_k - seqlen_q
            k_pos = torch.arange(seqlen_k, device=q.device)
            scores.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -math.inf)
        probs = scores.softmax(-1)
        outputs.append(torch.einsum("htk,khd->thd", probs, vb))
    return torch.stack(outputs)


def _paged_varlen_reference(q, cu_q, k, v, block_table, lengths, causal):
    outputs = []
    for b in range(block_table.shape[0]):
        start, end = int(cu_q[b]), int(cu_q[b + 1])
        outputs.append(
            _paged_reference(
                q[start:end].unsqueeze(0),
                k,
                v,
                block_table[b : b + 1],
                lengths[b : b + 1],
                causal,
            )[0]
        )
    return torch.cat(outputs)


def _paged_gappy_reference(q, k, v, block_table, starts, lengths, causal):
    batch, seqlen_q, num_heads, head_dim = q.shape
    outputs = []
    for b in range(batch):
        start = int(starts[b])
        seqlen_k = int(lengths[b])
        if seqlen_k == 0:
            outputs.append(torch.zeros_like(q[b], dtype=torch.float32))
            continue
        row_k = k[block_table[b]].reshape(-1, HKV, head_dim)
        row_v = v[block_table[b]].reshape(-1, HKV, head_dim)
        kb = row_k[start : start + seqlen_k].float()
        vb = row_v[start : start + seqlen_k].float()
        kb = kb.repeat_interleave(num_heads // HKV, dim=1)
        vb = vb.repeat_interleave(num_heads // HKV, dim=1)
        scores = torch.einsum("thd,khd->htk", q[b].float(), kb) / math.sqrt(head_dim)
        if causal:
            q_pos = torch.arange(seqlen_q, device=q.device) + seqlen_k - seqlen_q
            k_pos = torch.arange(seqlen_k, device=q.device)
            scores.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -math.inf)
        outputs.append(torch.einsum("htk,khd->thd", scores.softmax(-1), vb))
    return torch.stack(outputs)


@pytest.mark.parametrize("ctx", [128, 512])
@pytest.mark.parametrize("Sq,D", [(1, 64), (4, 128)])
def test_single_split_gqa_packed_matches_reference(ctx, Sq, D):
    """Short contexts run one split with GQA stride packing.

    The packed M tile has padding rows past the real (group, token) rows; they
    sit HEAD_DIM apart in the caller's buffer, so an unmasked store lands on
    the next KV heads' output. Split-K masked them, the single pass did not.
    """
    q, k, v, block_table, seqlen_k = _paged_inputs(1, Sq, ctx, D)
    got = _run(q, k, v, block_table, seqlen_k, 0)
    ref = _windowed_reference(q, k, v, ctx, ctx, HKV)
    torch.testing.assert_close(got[0].float(), ref[0], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_kv_splits", [0, 1])
def test_default_split_counts_are_auto_sized(monkeypatch, num_kv_splits):
    """Both 0 and the historical default 1 reach the paged split sizing."""
    seen = []
    real = fai._paged_num_kv_splits
    monkeypatch.setattr(
        fai, "_paged_num_kv_splits", lambda *a: (seen.append(real(*a)), seen[-1])[1]
    )
    _run(*_paged_inputs(1, 1, 32768, 128), num_kv_splits)
    assert len(seen) == 1 and seen[0] > 1


def test_auto_split_stride_packed_gqa_matches_single_pass_and_reference(monkeypatch):
    batch, seqlen_q, ctx, head_dim = 1, 64, 4096, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    auto_selection_calls = []

    def choose_two_splits(*args, **kwargs):
        auto_selection_calls.append((args, kwargs))
        return 2

    monkeypatch.setattr(fai, "_paged_num_kv_splits", choose_two_splits)
    guard_elements = 256
    guard_value = 42
    storage = torch.full(
        (q.numel() + 2 * guard_elements,),
        guard_value,
        device=q.device,
        dtype=q.dtype,
    )
    out = storage[guard_elements:-guard_elements].view_as(q)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=0,
        out=out,
    )
    single_pass = _run_single_pass(monkeypatch, q, k, v, block_table, seqlen_k)
    reference = _paged_reference(q, k, v, block_table, seqlen_k, causal=True)

    assert len(auto_selection_calls) == 1
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(got, single_pass, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got.float(), reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        storage[:guard_elements],
        torch.full_like(storage[:guard_elements], guard_value),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        storage[-guard_elements:],
        torch.full_like(storage[-guard_elements:], guard_value),
        atol=0,
        rtol=0,
    )


def test_ragged_varlen_does_not_auto_select_split_k(monkeypatch):
    ctx, head_dim = 512, 64
    q_dense, k, v, block_table, seqlen_k = _paged_inputs(2, 384, ctx, head_dim)
    q_lengths = torch.tensor([383, 384], device="cuda", dtype=torch.int32)
    cu_q = torch.nn.functional.pad(q_lengths.cumsum(0), (1, 0))
    q = torch.cat((q_dense[0, :383], q_dense[1]))
    cu_kv = torch.tensor([0, ctx, 2 * ctx], device="cuda", dtype=torch.int32)
    monkeypatch.setattr(fai, "_paged_num_kv_splits", lambda *a, **kw: 2)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=384,
        max_seqlen_kv=ctx,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    ref = _paged_varlen_reference(q, cu_q, k, v, block_table, seqlen_k, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_stride_packed_varlen_gqa_uses_token_row_lengths():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q_dense, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q = q_dense.reshape(-1, H, head_dim)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    cu_kv = torch.tensor([0, ctx, 2 * ctx], device="cuda", dtype=torch.int32)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=ctx,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    ref = _paged_varlen_reference(q, cu_q, k, v, block_table, seqlen_k, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_per_request_seqlen_k_is_honoured():
    """A request shorter than ``max_seqlen_kv`` must not attend to stale cache.

    The paged launch hands the kernel a scalar ``seq_len_kv`` -- the batch
    maximum -- and never the per-request tensor, so a short request reads cache
    past its own end. Error grows with the gap, not with tile misalignment:
    29952 is tile-aligned and still wrong.

    Measured here (B=8, ctx=32000, D=64, Sq=13), varying only the declared
    batch maximum, against an fp32 reference::

        seqlen_k   max_seqlen_kv     error
           32000           32000  2.49e-04
           31999           32000  2.03e-03
           30001           32000  1.29e-02
           29952           32000  1.30e-02

    Tolerance is bf16 epsilon (7.81e-03): it clears split-K's reordering noise
    (~2e-04) by a wide margin while still rejecting the two wrong answers.

    Fixed by forwarding ``seqlen_k`` to the paged launch and reading each
    request's length in the kernel under the KV_LENS trait, so the light route
    is correct for ragged batches without a device-to-host sync or a host-side
    scan, and stays legal under CUDA-graph capture.

    This was xfail(strict) while unfixed; the marker is gone because the
    behaviour it guarded now holds.
    """
    ctx, short, B = 32000, 30001, 8
    q, k, v, block_table, _ = _paged_inputs(B, 13, ctx, 64)
    seqlen_k = torch.full((B,), short, device="cuda", dtype=torch.int32)

    def _attend(max_seqlen_kv):
        return flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            max_seqlen_kv=max_seqlen_kv,
        )

    # Identical request lengths either way, so the *declared* batch maximum
    # must not change the answer.
    torch.testing.assert_close(
        _attend(ctx).float(), _attend(short).float(), atol=7.81e-3, rtol=7.81e-3
    )


@pytest.mark.parametrize("num_kv_splits", [0, 2])
def test_per_request_seqlen_k_on_dualwave_route(num_kv_splits):
    """bf16 non-causal Sq > 256 on gfx950 takes the dualwave kernel, which has
    no dense per-request KV length; a short request must still stop at its own
    end rather than read the stale cache up to ``max_seqlen_kv``."""
    ctx, Sq, D = 4096, 512, 64
    q, k, v, block_table, _ = _paged_inputs(2, Sq, ctx, D)
    seqlen_k = torch.tensor([ctx, 3001], device="cuda", dtype=torch.int32)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=False,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=num_kv_splits,
        max_seqlen_kv=ctx,
    )
    for b in range(2):
        n = int(seqlen_k[b])
        pages = block_table[b]
        kb = k[pages].reshape(-1, HKV, D)[:n].float()
        vb = v[pages].reshape(-1, HKV, D)[:n].float()
        kb = kb.repeat_interleave(H // HKV, 1)
        vb = vb.repeat_interleave(H // HKV, 1)
        scores = torch.einsum("thd,khd->htk", q[b].float(), kb) / math.sqrt(D)
        ref = torch.einsum("htk,khd->thd", scores.softmax(-1), vb)
        torch.testing.assert_close(got[b].float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal,Sq,num_kv_splits", [(True, 13, 0), (False, 512, 0)])
def test_per_request_seqlen_k_is_bounded_by_block_table_capacity(
    causal, Sq, num_kv_splits
):
    ctx, D = 512, 64
    q, k, v, block_table, _ = _paged_inputs(3, Sq, ctx, D)
    backing = torch.tensor(
        [ctx, 0, -PAGE, 0, ctx + PAGE, 0], device="cuda", dtype=torch.int64
    )
    seqlen_k = backing[::2]
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=num_kv_splits,
    )
    bounded = torch.tensor([ctx, 0, ctx], device="cuda", dtype=torch.int32)
    ref = _paged_reference(q, k, v, block_table, bounded, causal)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal,Sq", [(True, 13), (False, 512)])
def test_int64_dense_lengths_are_clamped_before_narrowing(causal, Sq):
    ctx, head_dim = 512, 64
    q, k, v, block_table, _ = _paged_inputs(2, Sq, ctx, head_dim)
    backing = torch.tensor([2**32 + 128, 0, 2**31, 0], device="cuda", dtype=torch.int64)
    seqlen_k = backing[::2]
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        max_seqlen_kv=ctx,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    bounded = torch.full((2,), ctx, device="cuda", dtype=torch.int32)
    ref = _paged_reference(q, k, v, block_table, bounded, causal)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_int64_cumulative_lengths_are_clamped_before_narrowing():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q_dense, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q = q_dense.reshape(-1, H, head_dim)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    huge = 2**32 + 128
    cu_kv = torch.tensor([0, huge, 2 * huge], device="cuda", dtype=torch.int64)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=ctx,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    bounded = torch.full((batch,), ctx, device="cuda", dtype=torch.int32)
    ref = _paged_varlen_reference(q, cu_q, k, v, block_table, bounded, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_int64_gappy_starts_are_clamped_before_narrowing():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q, k, v, block_table, _ = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q_packed = q.reshape(-1, H, head_dim)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    huge = 2**32 + 128
    cu_kv = torch.tensor([0, huge, 2 * huge], device="cuda", dtype=torch.int64)
    starts = torch.tensor([huge, 0], device="cuda", dtype=torch.int64)
    got = flydsl_flash_attn_func(
        q_packed,
        k,
        v,
        causal=False,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=None,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        kv_seqstart=starts,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=ctx,
        cross_seqlen=True,
    ).view_as(q)
    normalized_starts = torch.tensor([ctx, 0], device="cuda", dtype=torch.int32)
    bounded = torch.tensor([0, ctx], device="cuda", dtype=torch.int32)
    ref = _paged_gappy_reference(
        q, k, v, block_table, normalized_starts, bounded, causal=False
    )
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal,Sq", [(True, 13), (False, 512)])
def test_explicit_max_seqlen_kv_bounds_each_request(causal, Sq):
    ctx, bound, D = 512, 384, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, D)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        max_seqlen_kv=bound,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    bounded = torch.full_like(seqlen_k, bound)
    ref = _paged_reference(q, k, v, block_table, bounded, causal)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "route,Sq,varlen,causal",
    [
        ("generic", 4, False, True),
        ("generic", 16, True, True),
        ("generic", 1, False, False),
        ("head_packed", 4, False, True),
    ],
)
@pytest.mark.parametrize("num_kv_splits", [0, 1, 4])
def test_int32_lengths_are_bounded_in_kernel(
    monkeypatch, route, Sq, varlen, causal, num_kv_splits
):
    """Kernels clamp raw int32 lengths and write empty requests as 0, not NaN."""
    if route == "head_packed":
        calls = _count_hp_calls(monkeypatch)
    else:
        monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
        calls = []
        real = fai._build_paged_light
        monkeypatch.setattr(
            fai, "_build_paged_light", lambda **kw: (calls.append(kw), real(**kw))[1]
        )
    ctx, D = 512, 64
    bound = ctx - PAGE
    q, k, v, block_table, _ = _paged_inputs(4, Sq, ctx, D)
    seqlen_k = torch.tensor([-1, 2**30, 0, 100], device="cuda", dtype=torch.int32)
    bounded = torch.tensor([0, bound, 0, 100], device="cuda", dtype=torch.int32)
    kw = dict(
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=num_kv_splits,
        max_seqlen_kv=bound,
    )
    if varlen:
        q_packed = q.reshape(-1, H, D)
        cu_q = torch.arange(0, (4 + 1) * Sq, Sq, device="cuda", dtype=torch.int32)
        cu_kv = torch.nn.functional.pad(seqlen_k.clamp(min=0).cumsum(0), (1, 0))
        got = flydsl_flash_attn_func(
            q_packed,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv.to(torch.int32),
            max_seqlen_q=Sq,
            cross_seqlen=True,
            **kw,
        )
        ref = _paged_varlen_reference(
            q_packed, cu_q, k, v, block_table, bounded, causal=causal
        )
    else:
        got = flydsl_flash_attn_func(q, k, v, **kw)
        ref = _paged_reference(q, k, v, block_table, bounded, causal=causal)
    assert len(calls) == 1
    assert not got.isnan().any()
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_zero_length_has_empty_attention_result():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 512, 64)
    seqlen_k.fill_(-1)
    out, lse = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        return_lse=True,
    )
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    assert torch.isneginf(lse).all()


def test_gappy_lengths_are_bounded_by_remaining_block_table_capacity():
    batch, seqlen_q, ctx, head_dim = 3, 4, 512, 64
    q, k, v, block_table, _ = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    starts = torch.tensor([-64, 480, 576], device="cuda", dtype=torch.int32)
    raw_lengths = torch.tensor([128, 128, 128], device="cuda", dtype=torch.int32)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    cu_kv = torch.nn.functional.pad(raw_lengths.cumsum(0), (1, 0))
    got = flydsl_flash_attn_func(
        q.reshape(-1, H, head_dim),
        k,
        v,
        causal=False,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=None,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        kv_seqstart=starts,
        max_seqlen_q=seqlen_q,
        cross_seqlen=True,
    ).view_as(q)
    normalized_starts = torch.tensor([0, 480, 512], device="cuda", dtype=torch.int32)
    bounded = torch.tensor([128, 32, 0], device="cuda", dtype=torch.int32)
    ref = _paged_gappy_reference(
        q, k, v, block_table, normalized_starts, bounded, causal=False
    )
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("max_seqlen_kv", [-1, 513])
def test_rejects_max_seqlen_kv_outside_block_table_capacity(max_seqlen_kv):
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 512, 64)
    with pytest.raises(ValueError, match="max_seqlen_kv must be between"):
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            max_seqlen_kv=max_seqlen_kv,
        )


@pytest.mark.parametrize("mode", ["dense", "varlen", "gappy"])
def test_rejects_zero_width_block_table_before_build(mode):
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    kwargs = {}
    q_arg = q
    if mode != "dense":
        q_arg = q.reshape(-1, H, head_dim)
        cu_q = torch.arange(
            0,
            (batch + 1) * seqlen_q,
            seqlen_q,
            device="cuda",
            dtype=torch.int32,
        )
        cu_kv = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
        kwargs.update(
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=seqlen_q,
        )
        if mode == "gappy":
            seqlen_k = None
            kwargs["kv_seqstart"] = torch.zeros(batch, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="at least one page column"):
        flydsl_flash_attn_func(
            q_arg,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table[:, :0],
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            **kwargs,
        )


@pytest.mark.parametrize("causal,Sq", [(True, 1), (True, 4), (False, 512)])
def test_cuda_graph_sees_updated_seqlen_k(causal, Sq):
    """Capture without a scalar bound must not read ``seqlen_k`` on the host."""
    ctx = 4096
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, 64)
    out = torch.empty_like(q)

    def step():
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            out=out,
        )

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    seqlen_k.copy_(torch.tensor([ctx, 1000], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    ref = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k.clone(),
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_cuda_graph_clamps_int64_lengths_before_narrowing():
    ctx, seqlen_q, head_dim = 512, 4, 64
    q, k, v, block_table, _ = _paged_inputs(2, seqlen_q, ctx, head_dim)
    seqlen_k = torch.full((2,), ctx, device="cuda", dtype=torch.int64)
    out = torch.empty_like(q)

    def step():
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            max_seqlen_kv=ctx,
            kv_cache_layout="linear",
            num_kv_splits=0,
            out=out,
        )

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    seqlen_k.copy_(torch.tensor([2**32 + 128, 2**31], device="cuda", dtype=torch.int64))
    graph.replay()
    torch.cuda.synchronize()

    bounded = torch.full((2,), ctx, device="cuda", dtype=torch.int32)
    ref = _paged_reference(q, k, v, block_table, bounded, causal=True)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_cuda_graph_normalizes_gappy_metadata_without_dense_lengths():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q, k, v, block_table, _ = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q_packed = q.reshape(-1, H, head_dim)
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
    )
    raw_lengths = torch.tensor([128, 128], device="cuda", dtype=torch.int32)
    cu_kv = torch.nn.functional.pad(raw_lengths.cumsum(0), (1, 0))
    starts = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q_packed)

    def step():
        flydsl_flash_attn_func(
            q_packed,
            k,
            v,
            causal=False,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=None,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            kv_seqstart=starts,
            max_seqlen_q=seqlen_q,
            cross_seqlen=True,
            out=out,
        )

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    starts.copy_(torch.tensor([64, 480], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    bounded = torch.tensor([128, 32], device="cuda", dtype=torch.int32)
    ref = _paged_gappy_reference(q, k, v, block_table, starts, bounded, causal=False)
    torch.testing.assert_close(out.view_as(q).float(), ref, atol=2e-2, rtol=2e-2)


def test_updater_only_graph_invalidates_dualwave_lengths():
    ctx, Sq, D = 4096, 512, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, D)
    staging = seqlen_k.clone()

    def attend(lengths):
        return flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=False,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=lengths,
            kv_cache_layout="linear",
            num_kv_splits=0,
            max_seqlen_kv=ctx,
        )

    attend(seqlen_k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        seqlen_k.copy_(staging)
    staging.copy_(torch.tensor([ctx, 3001], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    got = attend(seqlen_k)
    ref = attend(seqlen_k.clone())
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_dualwave_accepts_inference_tensor_lengths():
    ctx, Sq, D = 4096, 512, 64
    q, k, v, block_table, _ = _paged_inputs(2, Sq, ctx, D)
    with torch.inference_mode():
        seqlen_k = torch.tensor([ctx, 3001], device="cuda", dtype=torch.int32)
        got = flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=False,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            max_seqlen_kv=ctx,
        )
    assert torch.isfinite(got).all()


def test_rejects_wrong_device_seqlen_k():
    q, k, v, block_table, _ = _paged_inputs(2, 4, 512, 64)
    seqlen_k = torch.tensor([512, 511], device="cpu", dtype=torch.int32)
    with pytest.raises(ValueError, match="seqlen_k must be on"):
        _run(q, k, v, block_table, seqlen_k, 0)


def test_rejects_wrong_device_block_table():
    q, k, v, block_table, seqlen_k = _paged_inputs(2, 4, 512, 64)
    with pytest.raises(ValueError, match="block_table must be on"):
        _run(q, k, v, block_table.cpu(), seqlen_k, 0)


def test_rejects_negative_num_kv_splits():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 384, 512, 64)
    with pytest.raises(ValueError, match="num_kv_splits must be nonnegative"):
        _run(q, k, v, block_table, seqlen_k, -1)


def test_explicit_stream_launch_matches_default_stream():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 384, 512, 64)
    launch_stream = torch.cuda.Stream()
    with torch.cuda.stream(launch_stream):
        got = flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=2,
            max_seqlen_kv=512,
            stream=launch_stream,
        )
    launch_stream.synchronize()
    ref = _run(q, k, v, block_table, seqlen_k, 2)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("varlen", [False, True])
def test_materialized_gqa_pack_and_unpack_stay_on_explicit_stream(monkeypatch, varlen):
    batch, seqlen_q, ctx, head_dim = 1, 4, 512, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    monkeypatch.setattr(fai, "_PAGED_GQA_STRIDE", False)
    q_arg = q
    kwargs = {}
    if varlen:
        q_arg = q.reshape(-1, H, head_dim)
        kwargs.update(
            cu_seqlens_q=torch.tensor([0, seqlen_q], device="cuda", dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, ctx], device="cuda", dtype=torch.int32),
            max_seqlen_q=seqlen_q,
        )

    def attend(*, out=None, stream=None):
        return flydsl_flash_attn_func(
            q_arg,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=0,
            out=out,
            stream=stream,
            **kwargs,
        )

    # Compile and initialize the materialized-pack variant before installing the
    # stream gate; otherwise compilation time could let the gate finish early.
    expected = attend()
    out = torch.full_like(q_arg, math.nan)
    torch.cuda.synchronize()

    producer_stream = torch.cuda.current_stream()
    launch_stream = torch.cuda.Stream()
    gate = torch.cuda.Event()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    for _ in range(128):
        torch.mm(a, b, out=c)
    gate.record(producer_stream)
    assert not gate.query(), "stream blocker completed before the test launch"

    got = attend(out=out, stream=launch_stream)
    launch_stream.synchronize()
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    producer_stream.synchronize()


def test_chain_sized_splits_are_bounded():
    short = fai._PAGED_TARGET_CHAIN * fai._PAGED_BLOCK_N_OUT
    assert fai._paged_num_kv_splits(1, HKV, 8, short, 128) == 1
    assert fai._paged_num_kv_splits(1, HKV, 8, 32768, 128) > 1

    for ctx in range(1024, 131072, 512):
        splits = fai._paged_num_kv_splits(4, HKV, 32, ctx, 128)
        assert splits & (splits - 1) == 0
        assert splits <= fai._PAGED_MAX_SPLITS


def test_chain_sized_splits_respect_workspace(monkeypatch):
    monkeypatch.setattr(fai, "_PAGED_WS_BUDGET_MB", 1)
    batch, seqlen_q, head_dim = 4, 16, 128
    splits = fai._paged_num_kv_splits(batch, H, seqlen_q, 131072, head_dim)
    elems = splits * batch * H * seqlen_q * (head_dim // 2 + 2)
    assert splits == 1 or elems * 4 <= 1024 * 1024


def test_window_sized_splits_respect_workspace_after_lds_floor(monkeypatch):
    monkeypatch.setattr(fai, "_PAGED_WS_BUDGET_MB", 1)
    batch, heads, seqlen_q, head_dim = 4, 8, 64, 128
    max_kv_pages = 2 * fai._PAGED_BT_LDS_SIZE
    splits = fai._paged_window_num_kv_splits(
        batch, heads, seqlen_q, 255, head_dim, max_kv_pages
    )
    elems = splits * batch * heads * seqlen_q * (head_dim // 2 + 2)

    assert splits == 1
    assert elems * 4 <= 1024 * 1024
    assert (max_kv_pages + splits - 1) // splits > fai._PAGED_BT_LDS_SIZE


def test_window_sized_splits_do_not_exceed_tile_or_configured_caps(monkeypatch):
    monkeypatch.setattr(fai, "_PAGED_WS_BUDGET_MB", 512)
    monkeypatch.setattr(fai, "_PAGED_WINDOW_MAX_SPLITS", 4)
    max_kv_pages = 8 * fai._PAGED_BT_LDS_SIZE

    by_configured_cap = fai._paged_window_num_kv_splits(1, 1, 1, 4096, 64, max_kv_pages)
    by_tile_cap = fai._paged_window_num_kv_splits(
        1, 1, 1, fai._PAGED_BLOCK_N_OUT, 64, max_kv_pages
    )

    assert by_configured_cap <= 4
    assert by_tile_cap == 1


def _paged_window_reference(q, k, v, ctx, window):
    _, seqlen_q, num_heads, head_dim = q.shape
    flat_k = k.reshape(-1, HKV, head_dim).float()[:ctx]
    flat_v = v.reshape(-1, HKV, head_dim).float()[:ctx]
    columns = torch.arange(ctx, device=q.device)
    output = torch.empty(seqlen_q, num_heads, head_dim, device=q.device)
    for head in range(num_heads):
        kv_head = head // (num_heads // HKV)
        for row in range(seqlen_q):
            query_position = ctx - seqlen_q + row
            scores = (q[0, row, head].float() @ flat_k[:, kv_head].T) / math.sqrt(
                head_dim
            )
            keep = (columns <= query_position) & (columns > query_position - window)
            output[row, head] = (
                scores.masked_fill(~keep, -math.inf).softmax(-1) @ flat_v[:, kv_head]
            )
    return output


@pytest.mark.parametrize("num_kv_splits", [1, 4, 0])
@pytest.mark.parametrize("seqlen_q", [1, 16, 64])
def test_paged_window_matches_reference(num_kv_splits, seqlen_q):
    ctx, window, head_dim = 2048, 255, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(1, seqlen_q, ctx, head_dim)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=num_kv_splits,
        max_seqlen_kv=ctx,
        window_left=window,
    )
    reference = _paged_window_reference(q, k, v, ctx, window)
    torch.testing.assert_close(got[0].float(), reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("window", [-1, 2047])
@pytest.mark.parametrize("seqlen_q", [1, 16])
def test_default_split_count_runs_past_one_block_table_window(window, seqlen_q):
    """``num_kv_splits=1`` past one split's block-table window is auto-sized."""
    ctx, head_dim = 262144, 64
    assert ctx // PAGE > fai._PAGED_BT_LDS_SIZE
    q, k, v, block_table, seqlen_k = _paged_inputs(1, seqlen_q, ctx, head_dim)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=1,
        max_seqlen_kv=ctx,
        window_left=window,
    )
    reference = _paged_window_reference(q, k, v, ctx, ctx if window < 0 else window)
    torch.testing.assert_close(got[0].float(), reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("seqlen_q", [1, 16])
def test_paged_window_rejects_noncausal(monkeypatch, seqlen_q):
    monkeypatch.setattr(
        fai,
        "_build_paged_light",
        lambda **_kwargs: pytest.fail("non-causal window reached kernel construction"),
    )
    q, k, v, block_table, seqlen_k = _paged_inputs(1, seqlen_q, 2048, 64)

    with pytest.raises(NotImplementedError, match="paged KV requires causal=True"):
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=False,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            max_seqlen_kv=2048,
            window_left=255,
        )


def test_paged_window_reaches_light_builder(monkeypatch):
    seen = []
    real = fai._build_paged_light
    monkeypatch.setattr(
        fai,
        "_build_paged_light",
        lambda **kwargs: (seen.append(kwargs), real(**kwargs))[1],
    )
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 2048, 64)
    flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=1,
        max_seqlen_kv=2048,
        window_left=255,
    )
    assert seen[-1]["window_left"] == 255
    assert seen[-1]["q_pack_qlen"] == 4


def test_paged_window_sizes_from_logical_gqa_query_length(monkeypatch):
    seen = []
    real_sizer = fai._paged_window_num_kv_splits

    def capture_sizer(*args):
        seen.append(args)
        return real_sizer(*args)

    monkeypatch.setattr(fai, "_paged_window_num_kv_splits", capture_sizer)
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 2048, 64)
    flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=0,
        max_seqlen_kv=2048,
        window_left=255,
    )

    assert seen[-1][2] == 4


def test_paged_window_rejects_gappy_kv():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 2048, 64)
    with pytest.raises(NotImplementedError, match="gappy"):
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_seqstart=torch.zeros(2, device="cuda", dtype=torch.int32),
            kv_cache_layout="linear",
            num_kv_splits=1,
            max_seqlen_kv=2048,
            window_left=255,
        )


# The dualwave paged kernel maps the MFMA M-axis to query rows, so at Sq=1 only
# 1 of 32 rows is real work. `decode/pa_decode_gfx950.py` packs query heads onto
# M instead; `_flydsl_flash_attn_paged` routes Sq=1 to it. These tests pin the
# routing conditions and the equivalence of the two kernels.


def _count_hp_calls(monkeypatch):
    """Patch the head-packed launcher to record invocations; returns the log.

    Also disables GQA head packing. Packing removes the same fan-out the
    head-packed decode kernel was routed here to avoid, covers Sq up to 64
    rather than the M-tile budget's 4-16, and therefore takes precedence by
    default (see test_gqa_packing_takes_precedence). The decode kernel remains
    the route for everything packing declines, and that is what these tests
    pin -- so they select it explicitly rather than depending on which
    mechanism happens to win.
    """
    from mslk.attention.flydsl.decode import pa_decode_dense

    monkeypatch.setattr(fai, "_PAGED_GQA_PACK", False)
    calls = []
    real = pa_decode_dense.pa_decode_paged_launch

    def _spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(pa_decode_dense, "pa_decode_paged_launch", _spy)
    return calls


def test_gqa_packing_takes_precedence(monkeypatch):
    """With both available, GQA packing owns the shape and the decode kernel idles.

    Packing folds query heads into M inside the generic paged kernel, so it
    fixes the same fan-out with a far wider reach. Measured across the paged
    decode shapes it is equal at a single query token, within noise at four, and
    2.8x better for longer query blocks, so it must win wherever it applies.
    """
    from mslk.attention.flydsl.decode import pa_decode_dense

    calls = []
    real = pa_decode_dense.pa_decode_paged_launch

    def _spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(pa_decode_dense, "pa_decode_paged_launch", _spy)
    _run(*_paged_inputs(1, 1, 32768, 64), 0)
    assert calls == []


def calls_groups(calls):
    """Query-group count the routing layer asked for on the first call."""
    return calls[0].get("query_groups", 1)


def test_head_packed_matches_dualwave_at_sq1(monkeypatch):
    """The two kernels must agree; the head-packed one is the faster path."""

    args = _paged_inputs(2, 1, 32768, 64)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
    ref = _run_single_pass(monkeypatch, *args)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", False)
    got = _run(*args, 0)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_gqa_packing_infers_num_kv_heads(monkeypatch):
    """Omitting num_kv_heads must not change the route: it is read from K."""
    seen = []
    real = fai._build_paged_light
    monkeypatch.setattr(
        fai, "_build_paged_light", lambda **kw: (seen.append(kw), real(**kw))[1]
    )
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 2048, 64)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    assert seen[-1]["q_pack_qlen"] == 4
    ref = _run(q, k, v, block_table, seqlen_k, 0)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def test_head_packed_selected_at_sq1(monkeypatch):
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, 1, 32768, 64), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("Sq", [2, 4])
def test_head_packed_selected_for_short_query_blocks(monkeypatch, Sq):
    """M holds ratio*Sq pairs over max_m_tiles(D) tiles; Sq=2/4 fit at any D."""
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, Sq, 32768, 64), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("Sq", [8, 13, 16])
def test_head_packed_selected_for_deep_tiling_at_d64(monkeypatch, Sq):
    """D=64 is measured clean to 8 M-tiles, so Sq up to 16 is in range."""
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, Sq, 32768, 64), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("Sq", [13, 15])
def test_head_packed_declined_when_d128_would_spill(monkeypatch, Sq):
    """D=128 spills past 6 M-tiles (measured 280B at 7, 916B at 8).

    Spilling a bandwidth-bound decode kernel is self-defeating, so these fall
    through to the existing path. 13 and 15 are odd, so equal-span query
    grouping cannot rescue them either -- see the Sq=16 case below.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, Sq, 32768, 128), 0)
    assert calls == []


def test_head_packed_uses_query_groups_when_single_pass_would_spill(monkeypatch):
    """D=128 / Sq=16 needs 8 tiles in one pass, which spills.

    Two passes of 8 query tokens need 4 tiles each -- under the budget -- at the
    cost of reading KV twice. Measured 1.9x faster than the path it replaces.
    """

    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, 16, 32768, 128), 0)
    assert len(calls) == 1
    assert calls_groups(calls) == 2


def test_query_grouping_kill_switch(monkeypatch):
    """With grouping disabled, a shape that only fits via groups declines."""

    monkeypatch.setattr(fai, "_DISABLE_PAGED_QGROUPS", True)
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, 16, 32768, 128), 0)
    assert calls == []


def test_query_grouping_not_used_when_single_pass_fits(monkeypatch):
    """Grouping is only for rescuing shapes the register budget would reject.

    D=64 fits Sq=16 in one pass, and grouping there measured inside run-to-run
    noise, so the extra KV pass must not be spent.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, 16, 32768, 64), 0)
    assert len(calls) == 1
    assert calls_groups(calls) == 1


def test_head_packed_declined_when_too_few_ctas(monkeypatch):
    """Deep tiling costs occupancy, so it needs enough CTAs to stay resident.

    One CTA per CU measured 0.74x the path it replaced; the floor rejects it.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, 16, 32000, 64), 0)  # 8 tiles, batch 1 -> ~1 CTA/CU
    assert calls == []


def test_head_packed_floor_does_not_reject_shallow_tiling(monkeypatch):
    """The CTA floor applies to deep tiling only.

    Sq=4 is two tiles and still holds 6 waves/SIMD; it was measured winning at
    batch 1, so the floor must not take it away.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, 4, 32000, 64), 0)
    assert len(calls) == 1


@pytest.mark.parametrize("Sq,D", [(2, 64), (4, 64), (4, 128)])
def test_head_packed_matches_dualwave_multi_token(monkeypatch, Sq, D):
    """Packed (qtok, head) rows must agree with the dualwave path.

    This is the check on the per-query-token causal bound: the kernel applies
    `min(t_end, t_full - Sq + qtok + 1)` itself, so a wrong bound shows up here
    as a mismatch on the earlier query rows only.
    """

    args = _paged_inputs(2, Sq, 32768, D)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
    ref = _run_single_pass(monkeypatch, *args)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", False)
    got = _run(*args, 0)
    assert got.shape == ref.shape == (2, Sq, H, D)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_head_packed_declined_for_wide_gqa_ratio(monkeypatch):
    """ratio = H // HKV must be <= MFMA_M (16); here it is 32."""
    calls = _count_hp_calls(monkeypatch)
    ctx, D, pages = 32768, 64, 32768 // PAGE
    torch.manual_seed(0)
    q = torch.randn(1, 1, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(pages, PAGE, 1, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    bt = torch.arange(pages, device="cuda", dtype=torch.int32).view(1, pages)
    sk = torch.full((1,), ctx, device="cuda", dtype=torch.int32)
    flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=1,
        block_table=bt,
        seqlen_k=sk,
        kv_cache_layout="linear",
        num_kv_splits=0,
    )
    assert calls == []


def test_head_packed_declined_for_non_causal_multi_token(monkeypatch):
    """The decode kernel always masks causally, so Sq > 1 non-causal must not
    reach it; Sq=1 is unaffected by the mask and still may."""
    calls = _count_hp_calls(monkeypatch)
    q, k, v, block_table, seqlen_k = _paged_inputs(2, 4, 4096, 128)
    kw = {
        "num_kv_heads": HKV,
        "block_table": block_table,
        "seqlen_k": seqlen_k,
        "kv_cache_layout": "linear",
        "num_kv_splits": 0,
    }
    flydsl_flash_attn_func(q, k, v, causal=False, **kw)
    assert calls == []
    flydsl_flash_attn_func(q[:, :1].contiguous(), k, v, causal=False, **kw)
    assert len(calls) == 1


@pytest.mark.parametrize("split_k", [1, 0])
def test_head_packed_non_contiguous_q(monkeypatch, split_k):
    """A Q sliced out of a fused QKV must not change the answer.

    The single-split epilogue addresses the output with Q's strides.
    """
    from mslk.attention.flydsl.decode import pa_decode_dense

    if split_k:
        monkeypatch.setattr(pa_decode_dense, "auto_split_k_hp", lambda *a: split_k)
    calls = _count_hp_calls(monkeypatch)
    B, Sq, ctx, D = 8, 2, 1024, 64
    _, k, v, block_table, seqlen_k = _paged_inputs(B, Sq, ctx, D)
    qkv = torch.randn(B, Sq, H + 2 * HKV, D, device="cuda", dtype=torch.bfloat16)
    q = qkv[:, :, :H]
    got = _run(q, k, v, block_table, seqlen_k, 0)
    ref = _run(q.contiguous(), k, v, block_table, seqlen_k, 0)
    assert len(calls) == 2
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
def test_head_packed_normalizes_strided_metadata(monkeypatch, metadata_dtype):
    calls = _count_hp_calls(monkeypatch)
    B, Sq, ctx, D = 2, 2, 1024, 64
    q, k, v, block_table, _ = _paged_inputs(B, Sq, ctx, D)
    table_storage = torch.full(
        (B, block_table.shape[1] * 2),
        -1,
        device="cuda",
        dtype=metadata_dtype,
    )
    strided_table = table_storage[:, ::2]
    strided_table.copy_(block_table.flip(1))
    length_storage = torch.full((B * 2,), -1, device="cuda", dtype=metadata_dtype)
    strided_lengths = length_storage[::2]
    logical_lengths = torch.tensor([ctx, ctx - PAGE], device="cuda", dtype=torch.int32)
    strided_lengths.copy_(logical_lengths)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=strided_table,
        seqlen_k=strided_lengths,
        kv_cache_layout="linear",
        num_kv_splits=1,
        max_seqlen_kv=ctx,
    )
    reference = _paged_reference(q, k, v, strided_table, logical_lengths, causal=True)

    assert len(calls) == 1
    torch.testing.assert_close(got.float(), reference, atol=2e-2, rtol=2e-2)


def test_head_packed_clamps_lengths_before_launch(monkeypatch):
    calls = _count_hp_calls(monkeypatch)
    B, Sq, ctx, D = 2, 1, 1024, 64
    q, k, v, block_table, _ = _paged_inputs(B, Sq, ctx, D)
    raw_lengths = torch.tensor(
        [ctx + 7 * PAGE, ctx // 2], device="cuda", dtype=torch.int64
    )
    bounded_lengths = torch.tensor([ctx, ctx // 2], device="cuda", dtype=torch.int32)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=raw_lengths,
        kv_cache_layout="linear",
        num_kv_splits=1,
        max_seqlen_kv=ctx,
    )
    reference = _paged_reference(q, k, v, block_table, bounded_lengths, causal=True)

    assert len(calls) == 1
    torch.testing.assert_close(got.float(), reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("split_k", [1, 0])
def test_head_packed_independent_kv_strides(monkeypatch, split_k):
    calls = _count_hp_calls(monkeypatch)
    B, Sq, ctx, D = 2, 2, 1024, 64
    pages = B * ctx // PAGE
    torch.manual_seed(0)
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k_storage = torch.randn(
        pages, PAGE * 2, HKV, D, device="cuda", dtype=torch.bfloat16
    )
    v_storage = torch.randn(
        pages, PAGE, HKV * 2, D, device="cuda", dtype=torch.bfloat16
    )
    k = k_storage[:, ::2]
    v = v_storage[:, :, ::2]
    block_table = torch.arange(pages, device="cuda", dtype=torch.int32).view(
        B, ctx // PAGE
    )
    lengths = torch.full((B,), ctx, device="cuda", dtype=torch.int32)

    assert k.shape == v.shape
    assert k.stride() != v.stride()
    assert not k.is_contiguous()
    assert not v.is_contiguous()
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=lengths,
        kv_cache_layout="linear",
        num_kv_splits=split_k,
        max_seqlen_kv=ctx,
    )
    reference = _paged_reference(q, k, v, block_table, lengths, causal=True)

    assert len(calls) == 1
    torch.testing.assert_close(got.float(), reference, atol=2e-2, rtol=2e-2)


def test_head_packed_rejects_mismatched_kv_shapes(monkeypatch):
    _count_hp_calls(monkeypatch)
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 1, 1024, 64)

    with pytest.raises(ValueError, match="matching shapes"):
        flydsl_flash_attn_func(
            q,
            k,
            v[:-1],
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=1,
            max_seqlen_kv=1024,
        )


def test_head_packed_runs_on_explicit_stream(monkeypatch):
    calls = _count_hp_calls(monkeypatch)
    q, k, v, block_table, seqlen_k = _paged_inputs(2, 2, 1024, 64)
    reference = _paged_reference(q, k, v, block_table, seqlen_k, causal=True)
    out = torch.full_like(q, math.nan)
    launch_stream = torch.cuda.Stream()
    ready = torch.cuda.Event()
    done = torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    launch_stream.wait_event(ready)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=1,
        max_seqlen_kv=1024,
        out=out,
        stream=launch_stream,
    )
    done.record(launch_stream)
    torch.cuda.current_stream().wait_event(done)

    assert len(calls) == 1
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(got.float(), reference, atol=2e-2, rtol=2e-2)


def test_head_packed_cuda_graph_replays_metadata(monkeypatch):
    calls = _count_hp_calls(monkeypatch)
    B, Sq, ctx, D = 2, 1, 1024, 64
    q, k, v, block_table, _ = _paged_inputs(B, Sq, ctx, D)
    static_q = q.clone()
    static_lengths = torch.tensor([ctx, ctx], device="cuda", dtype=torch.int32)

    def attend():
        return flydsl_flash_attn_func(
            static_q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=static_lengths,
            kv_cache_layout="linear",
            num_kv_splits=1,
            max_seqlen_kv=ctx,
        )

    attend()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = attend()

    graph.replay()
    torch.cuda.synchronize()
    first_reference = _paged_reference(
        static_q, k, v, block_table, static_lengths, causal=True
    )
    torch.testing.assert_close(captured.float(), first_reference, atol=2e-2, rtol=2e-2)

    static_q.copy_(q * 0.5)
    static_lengths.copy_(torch.tensor([ctx, ctx // 2], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    second_reference = _paged_reference(
        static_q, k, v, block_table, static_lengths, causal=True
    )

    assert len(calls) == 2
    torch.testing.assert_close(captured.float(), second_reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("split_k", [128, 256])
def test_head_packed_large_split_counts_match_reference(monkeypatch, split_k):
    calls = _count_hp_calls(monkeypatch)
    ctx, D = split_k * PAGE, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 1, ctx, D)
    out = torch.empty_like(q)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=split_k,
        max_seqlen_kv=ctx,
        out=out,
    )
    reference = _paged_reference(q, k, v, block_table, seqlen_k, causal=True)

    assert len(calls) == 1
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(got.float(), reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("split_k", [1, 0])
def test_head_packed_empty_request(monkeypatch, split_k):
    """seqlen_k == 0 (e.g. batch padding) attends to nothing, not to kv_max."""
    from mslk.attention.flydsl.decode import pa_decode_dense

    if split_k:
        monkeypatch.setattr(pa_decode_dense, "auto_split_k_hp", lambda *a: split_k)
    calls = _count_hp_calls(monkeypatch)
    q, k, v, block_table, _ = _paged_inputs(2, 1, 4096, 64)
    seqlen_k = torch.tensor([4096, 0], device="cuda", dtype=torch.int32)
    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=0,
        max_seqlen_kv=4096,
    )
    assert len(calls) == 1
    assert torch.equal(out[1], torch.zeros_like(out[1]))


def test_empty_batch_returns_without_launching():
    q = torch.empty(0, 1, H, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, PAGE, HKV, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    block_table = torch.empty(0, 1, device="cuda", dtype=torch.int32)
    seqlen_k = torch.empty(0, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)

    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=0,
        max_seqlen_kv=PAGE,
        out=out,
    )

    assert got.data_ptr() == out.data_ptr()
    assert got.shape == q.shape
    assert got.numel() == 0


def test_head_packed_cache_over_4gib(monkeypatch):
    """Page offsets must not wrap in 32 bits on a cache larger than 4 GiB.

    MHA with 32 KV heads at D=128 is 512 KiB per page, so page 8192 already
    sits at 4 GiB. Place the live pages above that and zero the rest, so a
    wrapped offset reads zeros instead of the request's KV.
    """
    hkv, D, ctx = H, 128, 1024
    n = ctx // PAGE
    pages = 8400
    need = 2 * pages * PAGE * hkv * D * 2
    free, _ = torch.cuda.mem_get_info()
    if free < need * 1.2:
        pytest.skip("needs ~9 GiB of free device memory")
    # MHA declines GQA packing, so this must reach the head-packed kernel; the
    # light kernel already addresses pages in 64 bits and would not test it.
    calls = _count_hp_calls(monkeypatch)
    torch.manual_seed(0)
    k = torch.zeros(pages, PAGE, hkv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    ids = torch.arange(pages - n, pages, device="cuda", dtype=torch.int32)
    k[ids.long()] = torch.randn(n, PAGE, hkv, D, device="cuda", dtype=torch.bfloat16)
    v[ids.long()] = torch.randn(n, PAGE, hkv, D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(1, 1, H, D, device="cuda", dtype=torch.bfloat16)
    got = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=hkv,
        block_table=ids.view(1, n),
        seqlen_k=torch.tensor([ctx], device="cuda", dtype=torch.int32),
        kv_cache_layout="linear",
        num_kv_splits=0,
        max_seqlen_kv=ctx,
    )
    assert len(calls) == 1
    kk = k[ids.long()].reshape(-1, hkv, D).float()
    vv = v[ids.long()].reshape(-1, hkv, D).float()
    scores = torch.einsum("hd,khd->hk", q[0, 0].float(), kk) / math.sqrt(D)
    ref = torch.einsum("hk,khd->hd", scores.softmax(-1), vv)
    torch.testing.assert_close(got[0, 0].float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("ratio", [3, 6])
def test_dense_decode_head_packed_non_dividing_ratio(monkeypatch, ratio):
    """A single query token fits any GQA ratio <= 16 heads-only, so dense
    decode must not fall back to the generic kernel for ratios like 3 or 6."""
    from mslk.attention.flydsl.decode import pa_decode_generic
    from mslk.attention.flydsl.decode.pa_decode_gfx950 import pa_decode_gfx950_launch

    def _no_fallback(*a, **kw):
        raise AssertionError("fell back to pa_decode_generic")

    monkeypatch.setattr(pa_decode_generic, "pa_decode_generic_launch", _no_fallback)
    B, ctx, hkv, D = 2, 1024, 4, 128
    torch.manual_seed(0)
    q = torch.randn(B, 1, 1, hkv * ratio, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, ctx, 1, hkv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    got = pa_decode_gfx950_launch(q, k, v, None, D**-0.5, 0, torch.bfloat16)
    kk = k[:, :, 0].float().repeat_interleave(ratio, 2)
    vv = v[:, :, 0].float().repeat_interleave(ratio, 2)
    scores = torch.einsum("bhd,bkhd->bhk", q[:, 0, 0].float(), kk) * D**-0.5
    ref = torch.einsum("bhk,bkhd->bhd", scores.softmax(-1), vv)
    torch.testing.assert_close(got[:, 0, 0].float(), ref, atol=2e-2, rtol=2e-2)


def test_head_packed_kill_switch(monkeypatch):
    calls = _count_hp_calls(monkeypatch)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
    _run(*_paged_inputs(1, 1, 32768, 64), 0)
    assert calls == []


def test_head_packed_exceeds_dualwave_page_table_cap():
    """The dualwave path caps at 2048 pages/split (131072 tokens at page 64).

    The decode kernel reads the block table straight from memory, so it has no
    such window -- this context would raise on the old path.
    """
    ctx = 262144
    out = _run(*_paged_inputs(1, 1, ctx, 64), 0)
    assert out.shape == (1, 1, H, 64)
    assert torch.isfinite(out).all()


def test_query_grouping_clamps_empty_prefix(monkeypatch):
    from mslk.attention.flydsl.decode import pa_decode_dense

    launches = []

    def fake_launch(q, _k, _v, seq_positions, *_args, kv_max, **_kwargs):
        launches.append((kv_max, seq_positions.clone()))
        return torch.zeros_like(q)

    monkeypatch.setattr(
        "mslk.attention.flydsl.decode.pa_decode_gfx950.pa_decode_gfx950_launch",
        fake_launch,
    )
    q = torch.empty(1, 16, 1, H, 64, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, PAGE, HKV, 64, device="cuda", dtype=torch.bfloat16)
    pa_decode_dense.pa_decode_paged_launch(
        q,
        cache,
        cache,
        torch.zeros(1, 1, device="cuda", dtype=torch.int32),
        torch.tensor([4], device="cuda", dtype=torch.int32),
        64**-0.5,
        page_size=PAGE,
        max_seqlen_kv=4,
        split_k=1,
        output_dtype=torch.bfloat16,
        query_groups=2,
    )
    assert [kv_max for kv_max, _ in launches] == [0, 4]
    assert [int(length.item()) for _, length in launches] == [0, 4]
