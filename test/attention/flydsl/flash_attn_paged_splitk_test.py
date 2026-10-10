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
