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
from paged_reference import paged_reference, spy

PAGE = 64  # _PAGED_PAGE_SIZE: the only page size the native paged kernel takes
H, HKV = 32, 4


@pytest.fixture(autouse=True)
def _generic_paged_route(monkeypatch):
    """Keep decodes on the generic split-K kernel these tests inspect, rather
    than pa_decode_swa (covered by pa_decode_swa_test.py)."""
    monkeypatch.setattr(fai, "_PAGED_SWA_KERNEL", False)


def _paged_inputs(B, Sq, ctx, D, device="cuda", dtype=torch.bfloat16, hkv=HKV):
    pages_per_req = ctx // PAGE
    total = B * pages_per_req
    torch.manual_seed(0)
    q = torch.randn(B, Sq, H, D, device=device, dtype=dtype)
    k = torch.randn(total, PAGE, hkv, D, device=device, dtype=dtype)
    v = torch.randn_like(k)
    block_table = torch.arange(total, device=device, dtype=torch.int32).view(
        B, pages_per_req
    )
    seqlen_k = torch.full((B,), ctx, device=device, dtype=torch.int32)
    return q, k, v, block_table, seqlen_k


def _run(q, k, v, block_table, seqlen_k, num_kv_splits=0, **kw):
    """A paged call; ``num_kv_heads`` defaults to K's head count and every
    other keyword to the interface default (causal, linear layout)."""
    kw.setdefault("num_kv_heads", k.shape[2])
    return flydsl_flash_attn_func(
        q,
        k,
        v,
        block_table=block_table,
        seqlen_k=seqlen_k,
        num_kv_splits=num_kv_splits,
        **kw,
    )


def _run_single_pass(monkeypatch, *args):
    """Reference with auto-sizing pinned to one split.

    ``num_kv_splits=1`` cannot serve as the reference: on the paged light route
    it is auto-sized exactly like ``0``.
    """
    with monkeypatch.context() as m:
        m.setattr(fai, "_paged_num_kv_splits", lambda *a, **kw: 1)
        return _run(*args, num_kv_splits=1)


def _varlen_q(q):
    """Pack a dense [B, Sq, H, D] Q into varlen form with its cu_seqlens_q."""
    batch, seqlen_q = q.shape[:2]
    cu_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, device=q.device, dtype=torch.int32
    )
    return q.reshape(-1, H, q.shape[-1]), cu_q


def _paged_varlen_reference(q, cu_q, k, v, block_table, lengths, causal):
    outputs = []
    for b in range(block_table.shape[0]):
        start, end = int(cu_q[b]), int(cu_q[b + 1])
        outputs.append(
            paged_reference(
                q[start:end].unsqueeze(0),
                k,
                v,
                block_table[b : b + 1],
                lengths[b : b + 1],
                causal,
            )[0]
        )
    return torch.cat(outputs)


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


def _wave_splitk_shapes():
    """Shapes the wave-split-K gate admits, beyond the 32x4 bf16 pair above
    (``test_auto_splitk_matches_single_split[0-128-1-1]``, which is not repeated).

    The gate turns on for packed-GQA decode at HEAD_DIM >= 128 over a full
    context, so it covers every GQA ratio and both 16-bit dtypes -- a far wider
    set than the benchmark suite, which only ever runs 32 query heads over 4 KV
    heads in bf16. The ratio matters because the gate's
    ``q_pack_group * q_pack_qlen <= rows_per_wave`` bound is what keeps a wave's
    share of the packed rows addressable; the dtype and head dim matter because
    they set how much MFMA work each tile carries against the merge's fixed cost.

    Only HEAD_DIM 128 appears: the gate says ``head_dim >= 128``, but the native
    paged route rejects anything outside {64, 128} before the builder is reached,
    and the gappy route that does take other head dims cannot GQA-pack (packing
    needs ``kv_seqstart is None``, gappy needs it set), so ``q_pack_group > 0``
    fails there. 128 is therefore the whole of the gate's reach -- see
    ``test_native_paged_head_dim_bound`` below, which pins that.
    """
    shapes = [
        (128, hkv, dt)
        for hkv in (1, 2, 4, 16, 32)  # H // hkv = GQA ratio 32, 16, 8, 2, 1
        for dt in (torch.bfloat16, torch.float16)
    ]
    # Ids keep the position in the full list, so they stay stable.
    return [
        pytest.param(*shape, id=f"{shape[0]}-{shape[1]}-dtype{i}")
        for i, shape in enumerate(shapes)
        if shape[1:] != (HKV, torch.bfloat16)
    ]


@pytest.mark.parametrize("D,hkv,dtype", _wave_splitk_shapes())
def test_wave_splitk_matches_single_split(monkeypatch, D, hkv, dtype):
    """Wave-split-K must not change the result, at any admitted shape.

    The single-split reference cannot reach wave-split-K -- the gate requires
    ``num_kv_splits > 1`` -- so it is a genuine oracle here. Toggling the env
    var would not work: the builders are lru_cached on their arguments and the
    flag is read inside the trait builder, so a second call with the flag
    flipped returns the already-built kernel.
    """
    args = _paged_inputs(1, 1, 32768, D, dtype=dtype, hkv=hkv)
    ref = _run_single_pass(monkeypatch, *args)
    got = _run(*args)
    # Same tolerance as the 32x4 case: wave-split-K reorders the softmax
    # reduction within a workgroup, which is 16-bit rounding, not structure.
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-3, rtol=2e-3)


def test_native_paged_head_dim_bound():
    """Native paged KV takes 64 or 128 only.

    This is what bounds wave-split-K's ``head_dim >= 128`` from above: without
    it the gate would read as open-ended. If this bound ever widens, the gate
    inherits head dims it has never run on, so the two must move together.
    """
    args = _paged_inputs(1, 1, 8192, 256)
    with pytest.raises(NotImplementedError, match="head_dim=64 or 128"):
        _run(*args)


@pytest.mark.parametrize("num_kv_splits", [0, 1])
def test_default_split_counts_are_auto_sized(monkeypatch, num_kv_splits):
    """Both 0 and the historical default 1 reach the paged split sizing."""
    seen = spy(monkeypatch, fai, "_paged_num_kv_splits").results
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
    """Power-of-two rounding must never exceed _PAGED_MAX_SPLITS."""
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


@pytest.mark.parametrize("window_left", [-1, 2048])
def test_auto_splits_keep_block_table_window(window_left):
    """The wide-tile cap lowers the split count after chain sizing; the final
    count must still give every split at most the block-table window's pages."""
    ctx = 3 << 20
    for Sq in (64, 128):
        splits = fai._paged_auto_num_kv_splits(
            1, H, Sq, Sq, ctx, 128, PAGE, window_left, False, 256
        )
        assert math.ceil(ctx / PAGE / splits) <= fai._PAGED_BT_LDS_SIZE


@pytest.mark.parametrize("ctx", [128, 512])
@pytest.mark.parametrize("Sq,D", [(1, 64), (4, 128)])
def test_single_split_gqa_packed_matches_reference(ctx, Sq, D):
    """Short contexts run one split with GQA stride packing.

    The packed M tile has padding rows past the real (group, token) rows; they
    sit HEAD_DIM apart in the caller's buffer, so an unmasked store lands on
    the next KV heads' output. Split-K masked them, the single pass did not.
    """
    q, k, v, block_table, seqlen_k = _paged_inputs(1, Sq, ctx, D)
    got = _run(q, k, v, block_table, seqlen_k)
    ref = paged_reference(q, k, v, block_table, seqlen_k, causal=True)
    torch.testing.assert_close(got[0].float(), ref[0], atol=2e-2, rtol=2e-2)


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

    got = _run(q, k, v, block_table, seqlen_k, out=out)
    single_pass = _run_single_pass(monkeypatch, q, k, v, block_table, seqlen_k)
    reference = paged_reference(q, k, v, block_table, seqlen_k, causal=True)

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

    got = _run(
        q,
        k,
        v,
        block_table,
        seqlen_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=384,
        max_seqlen_kv=ctx,
    )
    ref = _paged_varlen_reference(q, cu_q, k, v, block_table, seqlen_k, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_stride_packed_varlen_gqa_uses_token_row_lengths():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q_dense, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q, cu_q = _varlen_q(q_dense)
    cu_kv = torch.tensor([0, ctx, 2 * ctx], device="cuda", dtype=torch.int32)

    got = _run(
        q,
        k,
        v,
        block_table,
        seqlen_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=ctx,
    )
    ref = _paged_varlen_reference(q, cu_q, k, v, block_table, seqlen_k, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_return_lse_still_works_under_auto():
    """return_lse needs the generic light kernel, which requires splits <= 1."""
    args = _paged_inputs(1, 1, 32768, 128)
    out, lse = _run(*args, return_lse=True)
    assert out.shape == args[0].shape
    assert lse.shape == (1, H, 1)


# ── Head-packed paged decode fast path (Sq == 1) ──────────────────────────────
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
    from flydsl.runtime.device import get_rocm_arch
    from mslk.attention.flydsl.decode import pa_decode_dense

    if not get_rocm_arch().startswith("gfx950"):
        pytest.skip("the head-packed decode route is gfx950-only")
    monkeypatch.setattr(fai, "_PAGED_GQA_PACK", False)
    return spy(monkeypatch, pa_decode_dense, "pa_decode_paged_launch")


def test_gqa_packing_takes_precedence(monkeypatch):
    """With both available, GQA packing owns the shape and the decode kernel idles.

    Packing folds query heads into M inside the generic paged kernel, so it
    fixes the same fan-out with a far wider reach, so it must win wherever it
    applies.
    """
    from mslk.attention.flydsl.decode import pa_decode_dense

    calls = spy(monkeypatch, pa_decode_dense, "pa_decode_paged_launch")
    _run(*_paged_inputs(1, 1, 32768, 64))
    assert calls == []


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
        return _run(q, k, v, block_table, seqlen_k, max_seqlen_kv=max_seqlen_kv)

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
    got = _run(
        q, k, v, block_table, seqlen_k, num_kv_splits, causal=False, max_seqlen_kv=ctx
    )
    ref = paged_reference(q, k, v, block_table, seqlen_k, causal=False)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


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
    got = _run(q, k, v, block_table, seqlen_k, num_kv_splits, causal=causal)
    bounded = torch.tensor([ctx, 0, ctx], device="cuda", dtype=torch.int32)
    ref = paged_reference(q, k, v, block_table, bounded, causal)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal,Sq", [(True, 13), (False, 512)])
def test_int64_dense_lengths_are_clamped_before_narrowing(causal, Sq):
    ctx, head_dim = 512, 64
    q, k, v, block_table, _ = _paged_inputs(2, Sq, ctx, head_dim)
    backing = torch.tensor([2**32 + 128, 0, 2**31, 0], device="cuda", dtype=torch.int64)
    seqlen_k = backing[::2]
    got = _run(q, k, v, block_table, seqlen_k, causal=causal, max_seqlen_kv=ctx)
    bounded = torch.full((2,), ctx, device="cuda", dtype=torch.int32)
    ref = paged_reference(q, k, v, block_table, bounded, causal)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_int64_cumulative_lengths_are_clamped_before_narrowing():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q_dense, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q, cu_q = _varlen_q(q_dense)
    huge = 2**32 + 128
    cu_kv = torch.tensor([0, huge, 2 * huge], device="cuda", dtype=torch.int64)
    got = _run(
        q,
        k,
        v,
        block_table,
        seqlen_k,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=ctx,
    )
    bounded = torch.full((batch,), ctx, device="cuda", dtype=torch.int32)
    ref = _paged_varlen_reference(q, cu_q, k, v, block_table, bounded, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_int64_gappy_starts_are_clamped_before_narrowing():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q, k, v, block_table, _ = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q_packed, cu_q = _varlen_q(q)
    huge = 2**32 + 128
    cu_kv = torch.tensor([0, huge, 2 * huge], device="cuda", dtype=torch.int64)
    starts = torch.tensor([huge, 0], device="cuda", dtype=torch.int64)
    got = _run(
        q_packed,
        k,
        v,
        block_table,
        None,
        causal=False,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        kv_seqstart=starts,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=ctx,
        cross_seqlen=True,
    ).view_as(q)
    normalized_starts = torch.tensor([ctx, 0], device="cuda", dtype=torch.int32)
    bounded = torch.tensor([0, ctx], device="cuda", dtype=torch.int32)
    ref = paged_reference(
        q, k, v, block_table, bounded, causal=False, starts=normalized_starts
    )
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal,Sq", [(True, 13), (False, 512)])
def test_explicit_max_seqlen_kv_bounds_each_request(causal, Sq):
    ctx, bound, D = 512, 384, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, D)
    got = _run(q, k, v, block_table, seqlen_k, causal=causal, max_seqlen_kv=bound)
    bounded = torch.full_like(seqlen_k, bound)
    ref = paged_reference(q, k, v, block_table, bounded, causal)
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
        ref = paged_reference(q, k, v, block_table, bounded, causal=causal)
    assert len(calls) == 1
    assert not got.isnan().any()
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


def test_zero_length_has_empty_attention_result():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 512, 64)
    seqlen_k.fill_(-1)
    out, lse = _run(q, k, v, block_table, seqlen_k, return_lse=True)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    assert torch.isneginf(lse).all()


def test_gappy_lengths_are_bounded_by_remaining_block_table_capacity():
    batch, seqlen_q, ctx, head_dim = 3, 4, 512, 64
    q, k, v, block_table, _ = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    starts = torch.tensor([-64, 480, 576], device="cuda", dtype=torch.int32)
    raw_lengths = torch.tensor([128, 128, 128], device="cuda", dtype=torch.int32)
    q_packed, cu_q = _varlen_q(q)
    cu_kv = torch.nn.functional.pad(raw_lengths.cumsum(0), (1, 0))
    got = _run(
        q_packed,
        k,
        v,
        block_table,
        None,
        causal=False,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        kv_seqstart=starts,
        max_seqlen_q=seqlen_q,
        cross_seqlen=True,
    ).view_as(q)
    normalized_starts = torch.tensor([0, 480, 512], device="cuda", dtype=torch.int32)
    bounded = torch.tensor([128, 32, 0], device="cuda", dtype=torch.int32)
    ref = paged_reference(
        q, k, v, block_table, bounded, causal=False, starts=normalized_starts
    )
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("max_seqlen_kv", [-1, 513])
def test_rejects_max_seqlen_kv_outside_block_table_capacity(max_seqlen_kv):
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 512, 64)
    with pytest.raises(ValueError, match="max_seqlen_kv must be between"):
        _run(q, k, v, block_table, seqlen_k, max_seqlen_kv=max_seqlen_kv)


@pytest.mark.parametrize("mode", ["dense", "varlen", "gappy"])
def test_rejects_zero_width_block_table_before_build(mode):
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    kwargs = {}
    q_arg = q
    if mode != "dense":
        q_arg, cu_q = _varlen_q(q)
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
        _run(q_arg, k, v, block_table[:, :0], seqlen_k, **kwargs)


@pytest.mark.parametrize("causal,Sq", [(True, 1), (True, 4), (False, 512)])
def test_cuda_graph_sees_updated_seqlen_k(causal, Sq):
    """Capture without a scalar bound must not read ``seqlen_k`` on the host."""
    ctx = 4096
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, 64)
    out = torch.empty_like(q)

    def step():
        _run(q, k, v, block_table, seqlen_k, causal=causal, out=out)

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    seqlen_k.copy_(torch.tensor([ctx, 1000], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    ref = _run(q, k, v, block_table, seqlen_k.clone(), causal=causal)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


# ── Paged sliding window ──────────────────────────────────────────────────────
# The window mask lives in the generic kernel's N32 path, which the paged light
# route already builds. Every other paged route masks by its own rules and has
# no window term, so it must decline a windowed shape rather than drop the
# window silently.


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
    got = _run(
        q,
        k,
        v,
        block_table,
        seqlen_k,
        num_kv_splits,
        max_seqlen_kv=ctx,
        window_left=window,
    )
    ref = paged_reference(q, k, v, block_table, seqlen_k, True, window)[0]
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
    got = _run(q, k, v, block_table, seqlen_k, max_seqlen_kv=ctx, window_left=window)
    ref = paged_reference(q, k, v, block_table, seqlen_k, True, window)[0]
    torch.testing.assert_close(got[0].float(), ref, atol=2e-2, rtol=2e-2)


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
    reference = paged_reference(
        q, k, v, block_table, seqlen_k, causal=True, window_left=window
    )
    torch.testing.assert_close(got.float(), reference, atol=2e-2, rtol=2e-2)


def test_paged_window_keeps_gqa_packing(monkeypatch):
    """A causal window stays on the light kernel under GQA packing: the packed
    rows carry their query position, so the window bound still applies per row."""

    seen = spy(monkeypatch, fai, "_build_paged_light")
    args = _paged_inputs(1, 4, 2048, 64)

    _run(*args, num_kv_splits=1)
    assert seen[-1]["window_left"] == -1
    assert seen[-1]["q_pack_qlen"] == 4, "unwindowed shapes should still pack"

    _run(*args, num_kv_splits=1, max_seqlen_kv=2048, window_left=255)
    # The builder takes the exclusive bound, window_left + 1.
    assert seen[-1]["window_left"] == 256, "window must reach the light builder"
    assert seen[-1]["q_pack_qlen"] == 4, "causal windowed shapes should pack"


@pytest.mark.parametrize("Sq,causal", [(1, True), (4, True)], ids=["sq1", "sq4"])
def test_paged_window_is_never_silently_dropped(Sq, causal):
    """The window must change the answer on every route a shape can take.

    This is the sharp form of the routing guard: a route with no window term
    returns a full-context answer rather than failing, so identical output is
    the bug. Non-causal windows are rejected outright, so only causal shapes
    are checked.
    """
    ctx = 2048
    q, k, v, block_table, seqlen_k = _paged_inputs(1, Sq, ctx, 64)

    def attend(window_left):
        return _run(
            q,
            k,
            v,
            block_table,
            seqlen_k,
            1,
            causal=causal,
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
        _run(
            q,
            k,
            v,
            block_table,
            seqlen_k,
            1,
            kv_seqstart=torch.zeros(B + 1, device="cuda", dtype=torch.int32),
            max_seqlen_kv=ctx,
            window_left=255,
        )


def calls_groups(calls):
    """Query-group count the routing layer asked for on the first call."""
    return calls[0].get("query_groups", 1)


def test_gqa_packing_infers_num_kv_heads(monkeypatch):
    """Omitting num_kv_heads must not change the route: it is read from K."""
    seen = spy(monkeypatch, fai, "_build_paged_light")
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 4, 2048, 64)
    got = _run(q, k, v, block_table, seqlen_k, num_kv_heads=None)
    assert seen[-1]["q_pack_qlen"] == 4
    ref = _run(q, k, v, block_table, seqlen_k)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("Sq", [2])
def test_head_packed_selected_for_short_query_blocks(monkeypatch, Sq):
    """M holds ratio*Sq pairs over max_m_tiles(D) tiles; Sq=2/4 fit at any D.

    Sq=4 at batch 1 is test_head_packed_floor_does_not_reject_shallow_tiling.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, Sq, 32768, 64))
    assert len(calls) == 1


@pytest.mark.parametrize("Sq", [8, 13])
def test_head_packed_selected_for_deep_tiling_at_d64(monkeypatch, Sq):
    """D=64 compiles clean to 8 M-tiles, so Sq up to 16 is in range (Sq=16
    is test_query_grouping_not_used_when_single_pass_fits)."""
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, Sq, 32768, 64))
    assert len(calls) == 1


@pytest.mark.parametrize("Sq", [13, 15])
def test_head_packed_declined_when_d128_would_spill(monkeypatch, Sq):
    """D=128 spills past 6 M-tiles.

    Spilling a bandwidth-bound decode kernel is self-defeating, so these fall
    through to the existing path. 13 and 15 are odd, so equal-span query
    grouping cannot rescue them either -- see the Sq=16 case below.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, Sq, 32768, 128))
    assert calls == []


def test_head_packed_uses_query_groups_when_single_pass_would_spill(monkeypatch):
    """D=128 / Sq=16 needs 8 tiles in one pass, which spills.

    Two passes of 8 query tokens need 4 tiles each -- under the budget -- at the
    cost of reading KV twice.
    """

    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, 16, 32768, 128))
    assert len(calls) == 1
    assert calls_groups(calls) == 2


def test_query_grouping_kill_switch(monkeypatch):
    """With grouping disabled, a shape that only fits via groups declines."""

    monkeypatch.setattr(fai, "_DISABLE_PAGED_QGROUPS", True)
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, 16, 32768, 128))
    assert calls == []


def test_query_grouping_not_used_when_single_pass_fits(monkeypatch):
    """Grouping is only for rescuing shapes the register budget would reject.

    D=64 fits Sq=16 in one pass, so the extra KV pass must not be spent.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(8, 16, 32768, 64))
    assert len(calls) == 1
    assert calls_groups(calls) == 1


def test_head_packed_declined_when_too_few_ctas(monkeypatch):
    """Deep tiling costs occupancy, so it needs enough CTAs to stay resident.

    About one CTA per CU cannot hide latency; the floor rejects it.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, 16, 32000, 64))  # 8 tiles, batch 1 -> ~1 CTA/CU
    assert calls == []


def test_head_packed_floor_does_not_reject_shallow_tiling(monkeypatch):
    """The CTA floor applies to deep tiling only.

    Sq=4 is two tiles and still keeps enough waves resident, so the floor must
    not take it away at batch 1.
    """
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, 4, 32000, 64))
    assert len(calls) == 1


@pytest.mark.parametrize("Sq,D", [(1, 64), (2, 64), (4, 64), (4, 128)])
def test_head_packed_matches_generic_multi_token(monkeypatch, Sq, D):
    """Packed (qtok, head) rows must agree with the generic kernel, and the
    kill switch must keep the generic reference off the decode kernel.

    For Sq > 1 this is the check on the per-query-token causal bound: the
    kernel applies `min(t_end, t_full - Sq + qtok + 1)` itself, so a wrong
    bound shows up here as a mismatch on the earlier query rows only.
    """

    calls = _count_hp_calls(monkeypatch)
    args = _paged_inputs(2, Sq, 32768, D)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
    ref = _run(*args, num_kv_splits=1)
    assert calls == []
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", False)
    got = _run(*args)
    assert len(calls) == 1
    assert got.shape == ref.shape == (2, Sq, H, D)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_head_packed_declined_for_wide_gqa_ratio(monkeypatch):
    """ratio = H // HKV must be <= MFMA_M (16); here it is 32."""
    calls = _count_hp_calls(monkeypatch)
    _run(*_paged_inputs(1, 1, 32768, 64, hkv=1))
    assert calls == []


def test_head_packed_declined_for_non_causal_multi_token(monkeypatch):
    """The decode kernel always masks causally, so Sq > 1 non-causal must not
    reach it; Sq=1 is unaffected by the mask and still may."""
    calls = _count_hp_calls(monkeypatch)
    q, k, v, block_table, seqlen_k = _paged_inputs(2, 4, 4096, 128)
    _run(q, k, v, block_table, seqlen_k, causal=False)
    assert calls == []
    _run(q[:, :1].contiguous(), k, v, block_table, seqlen_k, causal=False)
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
    got = _run(q, k, v, block_table, seqlen_k)
    ref = _run(q.contiguous(), k, v, block_table, seqlen_k)
    assert len(calls) == 2
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("split_k", [1, 0])
def test_head_packed_empty_request(monkeypatch, split_k):
    """seqlen_k == 0 (e.g. batch padding) attends to nothing, not to kv_max."""
    from mslk.attention.flydsl.decode import pa_decode_dense

    if split_k:
        monkeypatch.setattr(pa_decode_dense, "auto_split_k_hp", lambda *a: split_k)
    calls = _count_hp_calls(monkeypatch)
    q, k, v, block_table, _ = _paged_inputs(2, 1, 4096, 64)
    seqlen_k = torch.tensor([4096, 0], device="cuda", dtype=torch.int32)
    out = _run(q, k, v, block_table, seqlen_k, max_seqlen_kv=4096)
    assert len(calls) == 1
    assert torch.equal(out[1], torch.zeros_like(out[1]))


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
    block_table = ids.view(1, n)
    seqlen_k = torch.tensor([ctx], device="cuda", dtype=torch.int32)
    got = _run(q, k, v, block_table, seqlen_k, max_seqlen_kv=ctx)
    assert len(calls) == 1
    ref = paged_reference(q, k, v, block_table, seqlen_k, causal=True)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)


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


def test_head_packed_exceeds_dualwave_page_table_cap(monkeypatch):
    """The dualwave path caps at 2048 pages/split (131072 tokens at page 64).

    The decode kernel reads the block table straight from memory, so it has no
    such window and runs this context.
    """
    calls = _count_hp_calls(monkeypatch)
    ctx = 262144
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 1, ctx, 64)
    out = _run(q, k, v, block_table, seqlen_k)
    assert len(calls) == 1
    ref = paged_reference(q, k, v, block_table, seqlen_k, causal=True)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_target_requires_gfx950_and_flydsl():
    assert torch.version.hip, "requires a ROCm build"
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch == "gfx950", f"requires gfx950, got {arch}"
    assert is_flydsl_available(), "requires FlyDSL"


def test_cuda_graph_clamps_int64_lengths_before_narrowing():
    ctx, seqlen_q, head_dim = 512, 4, 64
    q, k, v, block_table, _ = _paged_inputs(2, seqlen_q, ctx, head_dim)
    seqlen_k = torch.full((2,), ctx, device="cuda", dtype=torch.int64)
    out = torch.empty_like(q)

    def step():
        _run(q, k, v, block_table, seqlen_k, max_seqlen_kv=ctx, out=out)

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    seqlen_k.copy_(torch.tensor([2**32 + 128, 2**31], device="cuda", dtype=torch.int64))
    graph.replay()
    torch.cuda.synchronize()

    bounded = torch.full((2,), ctx, device="cuda", dtype=torch.int32)
    ref = paged_reference(q, k, v, block_table, bounded, causal=True)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_cuda_graph_normalizes_gappy_metadata_without_dense_lengths():
    batch, seqlen_q, ctx, head_dim = 2, 4, 512, 64
    q, k, v, block_table, _ = _paged_inputs(batch, seqlen_q, ctx, head_dim)
    q_packed, cu_q = _varlen_q(q)
    raw_lengths = torch.tensor([128, 128], device="cuda", dtype=torch.int32)
    cu_kv = torch.nn.functional.pad(raw_lengths.cumsum(0), (1, 0))
    starts = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q_packed)

    def step():
        _run(
            q_packed,
            k,
            v,
            block_table,
            None,
            causal=False,
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
    ref = paged_reference(q, k, v, block_table, bounded, causal=False, starts=starts)
    torch.testing.assert_close(out.view_as(q).float(), ref, atol=2e-2, rtol=2e-2)


def test_updater_only_graph_invalidates_dualwave_lengths():
    ctx, Sq, D = 4096, 512, 64
    q, k, v, block_table, seqlen_k = _paged_inputs(2, Sq, ctx, D)
    staging = seqlen_k.clone()

    def attend(lengths):
        return _run(q, k, v, block_table, lengths, causal=False, max_seqlen_kv=ctx)

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
        got = _run(q, k, v, block_table, seqlen_k, causal=False, max_seqlen_kv=ctx)
    assert torch.isfinite(got).all()


def test_rejects_wrong_device_seqlen_k():
    q, k, v, block_table, _ = _paged_inputs(2, 4, 512, 64)
    seqlen_k = torch.tensor([512, 511], device="cpu", dtype=torch.int32)
    with pytest.raises(ValueError, match="seqlen_k must be on"):
        _run(q, k, v, block_table, seqlen_k)


def test_rejects_wrong_device_block_table():
    q, k, v, block_table, seqlen_k = _paged_inputs(2, 4, 512, 64)
    with pytest.raises(ValueError, match="block_table must be on"):
        _run(q, k, v, block_table.cpu(), seqlen_k)


def test_rejects_negative_num_kv_splits():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 384, 512, 64)
    with pytest.raises(ValueError, match="num_kv_splits must be nonnegative"):
        _run(q, k, v, block_table, seqlen_k, -1)


def test_explicit_stream_launch_matches_default_stream():
    q, k, v, block_table, seqlen_k = _paged_inputs(1, 384, 512, 64)
    launch_stream = torch.cuda.Stream()
    with torch.cuda.stream(launch_stream):
        got = _run(
            q, k, v, block_table, seqlen_k, 2, max_seqlen_kv=512, stream=launch_stream
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
        return _run(
            q_arg, k, v, block_table, seqlen_k, out=out, stream=stream, **kwargs
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
