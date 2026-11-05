# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Paged decodes routed to pa_decode_swa: windowed and full attention, in-kernel
and reduce-kernel split merges, against an fp32 reference."""

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
H, HKV = 32, 4


def _gfx950() -> bool:
    from flydsl.runtime.device import get_rocm_arch

    return get_rocm_arch().startswith("gfx950")


@pytest.mark.parametrize(
    "B,Sq,D,ctx,window_left",
    [
        (1, 1, 128, 32768, -1),  # full, reduce-kernel merge
        (1, 4, 64, 32768, -1),
        (4, 1, 128, 4096, 2047),  # windowed, reduce-kernel merge
        (16, 1, 128, 4096, 2047),  # windowed, in-kernel merge
        (16, 4, 128, 4096, 2047),
        (8, 4, 64, 4096, 2047),
        (64, 4, 128, 4096, 2047),  # windowed, no split
        (64, 1, 64, 4096, 1023),
    ],
)
def test_pa_decode_swa_routes(monkeypatch, B, Sq, D, ctx, window_left):
    if not _gfx950():
        pytest.skip("pa_decode_swa is gfx950-only")
    from mslk.attention.flydsl.decode import pa_decode_swa as mod

    used = spy(monkeypatch, mod, "pa_decode_swa_launch")
    torch.manual_seed(0)
    pages = ctx // PAGE
    # A peaky softmax, so a key read from the wrong page changes the output.
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16) * 4
    k = torch.randn(B * pages, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.randperm(B * pages, device="cuda", dtype=torch.int32).view(
        B, pages
    )
    # Ragged lengths, including ones shorter than the window and than a page.
    lens = [ctx, Sq, 31, 65, ctx - 17, ctx // 2 + 3]
    seqlen_k = torch.tensor(
        [max(Sq, lens[b % len(lens)]) for b in range(B)],
        device="cuda",
        dtype=torch.int32,
    )

    def run():
        return flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            max_seqlen_kv=ctx,
            window_left=window_left,
        )

    out = run().clone()
    assert used, "expected the pa_decode_swa route"
    ref = paged_reference(q, k, v, block_table, seqlen_k, True, window_left)
    err = (out.float().reshape(ref.shape) - ref).abs().max().item()
    assert err < 2e-2 * max(1.0, ref.abs().max().item()), err
    # Split merges reset their arrival counters: a second call matches.
    assert torch.equal(run(), out)


@pytest.mark.parametrize("route", ["pa_decode_swa", "head_packed", "generic"])
def test_paged_decode_launches_on_the_given_stream(monkeypatch, route):
    """Each paged decode route launches on the caller's ``stream`` even when it
    is not the current stream: with the current stream held busy, the result
    must be ready once ``stream`` alone has finished."""
    if not _gfx950():
        pytest.skip("pa_decode_swa and the head-packed route are gfx950-only")
    from mslk.attention.flydsl import flash_attn_interface as fai

    if route != "pa_decode_swa":
        monkeypatch.setattr(fai, "_PAGED_SWA_KERNEL", False)
    if route == "head_packed":
        monkeypatch.setattr(fai, "_PAGED_GQA_PACK", False)
    if route == "generic":
        monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
    torch.manual_seed(0)
    B, D, ctx, window_left = 16, 128, 4096, 2047
    pages = ctx // PAGE
    q = torch.randn(B, 1, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B * pages, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.arange(B * pages, device="cuda", dtype=torch.int32).view(
        B, pages
    )
    seqlen_k = torch.full((B,), ctx, device="cuda", dtype=torch.int32)

    def run(stream=None):
        return flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            num_kv_heads=HKV,
            block_table=block_table,
            seqlen_k=seqlen_k,
            kv_cache_layout="linear",
            max_seqlen_kv=ctx,
            window_left=window_left if route == "pa_decode_swa" else -1,
            stream=stream,
        )

    ref = run()  # also compiles, so the timed call below only launches
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    torch.cuda._sleep(1 << 31)  # hold the current stream busy
    got = run(stream=side)
    with torch.cuda.stream(side):
        # Copied on ``side`` right behind the launch: a kernel queued on the
        # busy current stream instead has not run yet.
        snapshot = got.clone()
    side.synchronize()
    still_busy = not torch.cuda.current_stream().query()
    torch.cuda.synchronize()
    assert still_busy, "the current stream finished first; the test proves nothing"
    torch.testing.assert_close(snapshot, ref, atol=0, rtol=0)


@pytest.mark.parametrize("window_left", [2047, -1])
def test_paged_decode_ignores_block_table_padding(window_left):
    """Slots past a request's pages may hold anything (here -1); waves left
    without keys must not read through them."""
    if not _gfx950():
        pytest.skip("pa_decode_swa is gfx950-only")
    torch.manual_seed(0)
    B, D, ctx = 4, 128, 4096
    pages = ctx // PAGE
    q = torch.randn(B, 1, H, D, device="cuda", dtype=torch.bfloat16) * 4
    k = torch.randn(B * pages, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.arange(B * pages, device="cuda", dtype=torch.int32).view(
        B, pages
    )
    # Page-aligned lengths put the empty waves on the first padding slot.
    seqlen_k = torch.tensor([64, 128, 1024, 2048], device="cuda", dtype=torch.int32)
    for b in range(B):
        block_table[b, -(-int(seqlen_k[b]) // PAGE) :] = -1
    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_heads=HKV,
        block_table=block_table,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
        max_seqlen_kv=ctx,
        window_left=window_left,
    )
    ref = paged_reference(q, k, v, block_table, seqlen_k, True, window_left)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2 * max(1.0, ref.abs().max().item()), err
