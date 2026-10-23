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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flydsl_available(),
    reason="requires a ROCm GPU with FlyDSL",
)

PAGE = 64
H, HKV = 32, 4


def _gfx950() -> bool:
    from flydsl.runtime.device import get_rocm_arch

    return get_rocm_arch().startswith("gfx950")


def _reference(q, k, v, block_table, seqlen_k, window_left):
    B, Sq, _, D = q.shape
    g = H // HKV
    out = torch.zeros(B, Sq, H, D, device=q.device)
    for b in range(B):
        n = int(seqlen_k[b])
        pages = -(-n // PAGE)
        kk = k[block_table[b, :pages].long()].reshape(-1, HKV, D)[:n].float()
        vv = v[block_table[b, :pages].long()].reshape(-1, HKV, D)[:n].float()
        kk = kk.repeat_interleave(g, dim=1)
        vv = vv.repeat_interleave(g, dim=1)
        pos = torch.arange(n, device=q.device)
        qpos = n - Sq + torch.arange(Sq, device=q.device)
        mask = pos[None, :] <= qpos[:, None]
        if window_left >= 0:
            mask &= pos[None, :] > qpos[:, None] - window_left
        s = torch.einsum("lhd,khd->hlk", q[b].float(), kk) / D**0.5
        s = s.masked_fill(~mask[None], float("-inf"))
        out[b] = torch.einsum("hlk,khd->lhd", s.softmax(-1), vv)
    return out


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

    used = []
    real = mod.pa_decode_swa_launch
    monkeypatch.setattr(
        mod,
        "pa_decode_swa_launch",
        lambda *a, **kw: (used.append(kw), real(*a, **kw))[1],
    )
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
    ref = _reference(q, k, v, block_table, seqlen_k, window_left)
    err = (out.float().reshape(ref.shape) - ref).abs().max().item()
    assert err < 2e-2 * max(1.0, ref.abs().max().item()), err
    # Split merges reset their arrival counters: a second call matches.
    assert torch.equal(run(), out)
