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
from mslk.attention.flydsl.flash_attn_interface import (
    _paged_num_kv_splits,
    flydsl_flash_attn_func,
)
from mslk.flydsl.common import is_flydsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flydsl_available(),
    reason="requires a ROCm GPU with FlyDSL",
)

PAGE = 64  # _PAGED_PAGE_SIZE: the only page size the native paged kernel takes
H, HKV = 32, 4


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
    """Reference with auto-sizing pinned to one split.

    ``num_kv_splits=1`` cannot serve as the reference: on the paged light route
    it is auto-sized exactly like ``0``.
    """
    with monkeypatch.context() as m:
        m.setattr(fai, "_paged_num_kv_splits", lambda *a, **kw: 1)
        return _run(*args, num_kv_splits=1)


@pytest.mark.parametrize("B,Sq", [(1, 1), (1, 16), (2, 16), (4, 4)])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("num_kv_splits", [0, 1])
def test_auto_splitk_matches_single_split(monkeypatch, B, Sq, D, num_kv_splits):
    """Auto split-K must be numerically equivalent to the single-pass kernel."""
    args = _paged_inputs(B, Sq, 32768, D)
    ref = _run_single_pass(monkeypatch, *args)
    got = _run(*args, num_kv_splits=num_kv_splits)
    # Split-K reorders the softmax reduction, so allow bf16 rounding but nothing
    # structural: bf16 epsilon is ~7.8e-3 and observed deviation is ~2e-4.
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("nks", [2, 4, 8])
def test_forced_splitk_matches_single_split_at_short_q(monkeypatch, nks):
    """Explicit split-K at Sq < 384 (previously rejected outright) is correct."""
    args = _paged_inputs(1, 1, 32768, 128)
    ref = _run_single_pass(monkeypatch, *args)
    got = _run(*args, num_kv_splits=nks)
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-3, rtol=2e-3)


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


def test_paged_splits_decline_short_chain():
    """A context that fits in the target chain gains nothing from a combine pass."""
    ctx = fai._PAGED_TARGET_CHAIN * fai._PAGED_BLOCK_N_OUT
    assert _paged_num_kv_splits(1, HKV, 8, ctx, 128) == 1


def test_paged_splits_when_chain_is_long():
    assert _paged_num_kv_splits(1, HKV, 8, 32768, 128) > 1


def test_paged_splits_capped_by_grid():
    """A large base grid must not be split past the workgroup ceiling."""
    B = 256
    blocks = B * H
    splits = _paged_num_kv_splits(B, H, 1, 131072, 128)
    assert splits <= max(1, fai._PAGED_MAX_WG // blocks)


def test_paged_splits_capped_by_workspace(monkeypatch):
    """The fp32 workspace must stay under its budget."""
    monkeypatch.setattr(fai, "_PAGED_WS_BUDGET_MB", 1)
    B, Sq, D = 4, 16, 128
    splits = _paged_num_kv_splits(B, H, Sq, 131072, D)
    elems = splits * B * H * Sq * (D // 2 + 2)
    assert splits == 1 or elems * 4 <= 1024 * 1024


def test_paged_splits_are_powers_of_two():
    """The split count is a compile-time trait; a growing decode context must
    not walk through a new value (and a new JIT) every few hundred tokens."""
    seen = set()
    for ctx in range(1024, 131072, 512):
        for B, Sq in ((1, 8), (4, 32), (16, 8)):
            splits = _paged_num_kv_splits(B, HKV, Sq, ctx, 128)
            assert splits & (splits - 1) == 0, (ctx, B, Sq, splits)
            seen.add(splits)
    assert len(seen) <= 8


def test_paged_splits_respect_user_cap(monkeypatch):
    """Power-of-two rounding must never exceed FLYDSL_PAGED_MAX_SPLITS."""
    monkeypatch.setattr(fai, "_PAGED_MAX_SPLITS", 48)
    for B, Sq, ctx in ((1, 8, 32000), (16, 8, 128000), (1, 1, 512000)):
        blocks = B * HKV * math.ceil(Sq / 64)
        cap = 48 * (2 if blocks < fai._PAGED_STARVED_BLOCKS else 1)
        assert _paged_num_kv_splits(B, HKV, Sq, ctx, 128) <= cap


def test_paged_splits_keep_block_table_window():
    """Rounding down must not leave a split owning more pages than the
    block-table LDS window holds; that shape would raise instead of run."""
    B, ctx = 640, 300000  # by_grid allows 3 splits; 2 would need 2344 pages
    splits = _paged_num_kv_splits(B, HKV, 1, ctx, 128)
    pages = math.ceil(ctx / PAGE)
    assert math.ceil(pages / splits) <= fai._PAGED_BT_LDS_SIZE


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
    torch.testing.assert_close(got[0].float(), ref, atol=2e-2, rtol=2e-2)


def test_return_lse_still_works_under_auto():
    """return_lse needs the generic light kernel, which requires splits <= 1."""
    args = _paged_inputs(1, 1, 32768, 128)
    out, lse = flydsl_flash_attn_func(
        args[0],
        args[1],
        args[2],
        causal=True,
        num_kv_heads=HKV,
        block_table=args[3],
        seqlen_k=args[4],
        kv_cache_layout="linear",
        num_kv_splits=0,
        return_lse=True,
    )
    assert out.shape == args[0].shape
    assert lse.shape == (1, H, 1)


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


@pytest.mark.parametrize("Sq", [1, 4])
def test_cuda_graph_sees_updated_seqlen_k(Sq):
    """The KV-length scan is memoised on ``seqlen_k``, but a graph captured
    after an eager warm-up must record it: replays after ``seqlen_k.copy_()``
    have to use the new lengths. The same holds for the packed-Q prefix sums."""
    ctx = 4096
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, 64)
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
            kv_cache_layout="linear",
            num_kv_splits=0,
            max_seqlen_kv=ctx,
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
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k.clone(),
        kv_cache_layout="linear",
        num_kv_splits=0,
        max_seqlen_kv=ctx,
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


# ── Paged sliding window ──────────────────────────────────────────────────────
# The window mask lives in the generic kernel's N32 path, which the paged light
# route already builds. Every other paged route masks by its own rules and has
# no window term, so it must decline a windowed shape rather than drop the
# window silently.


def _windowed_reference(q, k, v, ctx, window, num_kv_heads):
    """Bottom-right causal attention restricted to the last ``window`` keys.

    ``window`` is the kernel's ``window_left``: it keeps ``kv > q_pos - window``,
    so exactly ``window`` keys including the query's own position.
    """
    B, Sq, H, D = q.shape
    flat_k = k.reshape(-1, num_kv_heads, D).float()[:ctx]
    flat_v = v.reshape(-1, num_kv_heads, D).float()[:ctx]
    col = torch.arange(ctx, device=q.device)
    out = torch.zeros(Sq, H, D, device=q.device)
    for h in range(H):
        kv = h // (H // num_kv_heads)
        for t in range(Sq):
            q_abs = ctx - Sq + t
            scores = (q.float()[0, t, h] @ flat_k[:, kv].T) / math.sqrt(D)
            keep = (col <= q_abs) & (col > q_abs - window)
            out[t, h] = (
                torch.softmax(scores.masked_fill(~keep, float("-inf")), -1)
                @ flat_v[:, kv]
            )
    return out


@pytest.mark.parametrize("num_kv_splits", [1, 4, 0])
@pytest.mark.parametrize("Sq", [1, 4, 16, 32, 64])
@pytest.mark.parametrize("window", [17, 256, 2048])
def test_paged_window_matches_reference(num_kv_splits, Sq, window):
    """A windowed paged call must match an explicitly windowed reference.

    Parametrised over split counts because split-K partitions the KV range: the
    window bound has to hold inside each partition, not just end to end. Query
    lengths cover GQA packing (rows = group x Sq, 128 rows at Sq=16), and the
    windows span less than one KV tile up to the whole context.
    """
    ctx, B, D = 2048, 1, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(B, Sq, ctx, D)
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
    ref = _windowed_reference(q, k, v, ctx, window, HKV)
    torch.testing.assert_close(got[0].float(), ref, atol=2e-2, rtol=2e-2)


def test_paged_window_splits_bounded():
    """Windowed sizing splits by grid, never past the reachable KV tiles, and
    stays a power of two (the split count is compiled in)."""
    for B, rows, reach in ((1, 8, 2049), (64, 32, 2052), (4, 128, 17), (16, 8, 300)):
        s = fai._paged_window_num_kv_splits(B, HKV, rows, reach, 128)
        assert s & (s - 1) == 0
        assert s <= max(1, math.ceil(reach / fai._PAGED_BLOCK_N_OUT))
        assert s <= fai._PAGED_WINDOW_MAX_SPLITS


def test_paged_window_long_context_matches_reference():
    """A window over a context past one split's block-table window (2048 pages)
    still needs enough splits for the pages, even though it reads few keys."""
    ctx, window, D = 262144, 2048, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 1, ctx, D)
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
        max_seqlen_kv=ctx,
        window_left=window,
    )
    ref = _windowed_reference(q, k, v, ctx, window, HKV)
    torch.testing.assert_close(got[0].float(), ref, atol=2e-2, rtol=2e-2)


def test_paged_window_keeps_gqa_packing(monkeypatch):
    """A causal window stays on the light kernel under GQA packing: the packed
    rows carry their query position, so the window bound still applies per row."""

    seen = []
    real = fai._build_paged_light
    monkeypatch.setattr(
        fai, "_build_paged_light", lambda **kw: (seen.append(kw), real(**kw))[1]
    )
    args = _paged_inputs(1, 4, 2048, 64)

    _run(*args, num_kv_splits=1)
    assert seen[-1]["window_left"] == -1
    assert seen[-1]["q_pack_qlen"] == 4, "unwindowed shapes should still pack"

    flydsl_flash_attn_func(
        args[0],
        args[1],
        args[2],
        causal=True,
        num_kv_heads=HKV,
        block_table=args[3],
        seqlen_k=args[4],
        kv_cache_layout="linear",
        num_kv_splits=1,
        max_seqlen_kv=2048,
        window_left=255,
    )
    assert seen[-1]["window_left"] == 255, "window must reach the light builder"
    assert seen[-1]["q_pack_qlen"] == 4, "causal windowed shapes should pack"


@pytest.mark.parametrize(
    "Sq,causal", [(1, True), (4, True), (512, False)], ids=["sq1", "sq4", "sq512"]
)
def test_paged_window_is_never_silently_dropped(Sq, causal):
    """The window must change the answer on every route a shape can take.

    This is the sharp form of the routing guard: a route with no window term
    returns a full-context answer rather than failing, so identical output is
    the bug. ``Sq=512`` non-causal is the case that previously reached the
    native kernel and produced bit-identical results with and without a window.
    """
    ctx = 2048
    q, k, v, block_table, seqlen_k = _paged_inputs(1, Sq, ctx, 64)

    def attend(window_left):
        return flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=causal,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            num_kv_splits=1,
            max_seqlen_kv=ctx,
            window_left=window_left,
        )

    assert not torch.equal(attend(-1), attend(255))


def test_paged_window_still_rejected_for_gappy_kv():
    """Gappy paged indexes KV differently, so the flat-column window bound does
    not line up. It must fail loudly rather than answer incorrectly."""
    ctx, B = 2048, 1
    q, k, v, block_table, seqlen_k = _paged_inputs(B, 4, ctx, 64)
    with pytest.raises(NotImplementedError, match="gappy"):
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_seqstart=torch.zeros(B + 1, device="cuda", dtype=torch.int32),
            kv_cache_layout="linear",
            num_kv_splits=1,
            max_seqlen_kv=ctx,
            window_left=255,
        )
