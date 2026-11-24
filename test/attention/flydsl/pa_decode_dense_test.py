# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dense decodes routed to the gfx950 coop-DMA kernel (pa_decode_gfx950_coop),
which serves GQA ratios above 16, against an fp32 reference."""

import math

import pytest
import torch
from mslk.flydsl.common import is_flydsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flydsl_available(),
    reason="requires a ROCm GPU with FlyDSL",
)

TILE = 64  # keys per KV tile of the coop kernel
PEAK = 16.0  # score of a peaked row's target key
CALLS = 8


def _gfx950() -> bool:
    from flydsl.runtime.device import get_rocm_arch

    return get_rocm_arch().startswith("gfx950")


def _inputs(gen, B, H_q, H_kv, D, kv_len):
    """New bf16 inputs for one call, in pa_decode_launch's 5D layout.

    Lengths cycle through full, more than half, any, and at most 3 tiles. K is
    +-1. Three quarters of the query rows are peaked: q = c * k[t] for a random
    valid key t, so key t scores PEAK and the row's output is about v[t]; a
    stale V row there moves the output by O(1). V rows past a request's length
    hold large finite values, which must get zero weight."""
    dev = "cuda"
    u = torch.rand(B, generator=gen, device=dev).tolist()
    lengths = []
    for b in range(B):
        n = [kv_len, kv_len // 2 + 1 + int(u[b] * (kv_len // 2))]
        n += [1 + int(u[b] * kv_len), 1 + int(u[b] * 3 * TILE)]
        lengths.append(min(n[b % 4], kv_len))
    seq = torch.tensor(lengths, device=dev, dtype=torch.int32)
    k = torch.randint(0, 2, (B, kv_len, 1, H_kv, D), generator=gen, device=dev)
    k = k.float() * 2 - 1
    v = torch.randn(B, kv_len, 1, H_kv, D, generator=gen, device=dev)
    past = torch.arange(kv_len, device=dev)[None, :] >= seq[:, None]
    v = torch.where(past[:, :, None, None, None], 30.0 * v, v)
    t = (torch.rand(B, H_q, generator=gen, device=dev) * seq[:, None]).long()
    t = torch.minimum(t, seq[:, None].long() - 1)
    kv_head = torch.arange(H_q, device=dev) // (H_q // H_kv)
    batch = torch.arange(B, device=dev)[:, None]
    q = (PEAK / math.sqrt(D)) * k[batch, t, 0, kv_head[None, :]]
    peaked = torch.rand(B, H_q, 1, generator=gen, device=dev) < 0.75
    q = torch.where(peaked, q, torch.randn(B, H_q, D, generator=gen, device=dev))
    dt = torch.bfloat16
    return q.to(dt).view(B, 1, 1, H_q, D), k.to(dt), v.to(dt), seq


def _reference(q, k, v, seq):
    """fp32 softmax(q k^T / sqrt(D)) v over each request's first seq[b] keys."""
    B, _, _, H_q, D = q.shape
    kv_len, H_kv = k.shape[1], k.shape[3]
    qf = q.float().view(B, H_kv, H_q // H_kv, D)
    kf, vf = k[:, :, 0].float(), v[:, :, 0].float()
    scores = torch.einsum("bgrd,bkgd->bgrk", qf, kf) / math.sqrt(D)
    past = torch.arange(kv_len, device=q.device)[None, :] >= seq[:, None]
    scores = scores.masked_fill(past[:, None, None, :], -math.inf)
    out = torch.einsum("bgrk,bkgd->bgrd", scores.softmax(-1), vf)
    return out.reshape(B, H_q, D)


@pytest.mark.parametrize(
    "B,H_q,H_kv,D,kv_len",
    [
        (64, 32, 1, 128, 1024),  # MQA with 32 query heads
        (64, 64, 2, 128, 1024),  # GQA ratio 32 with 2 KV heads
    ],
)
def test_coop_decode_matches_reference_on_new_inputs(
    monkeypatch, B, H_q, H_kv, D, kv_len
):
    """All 4 waves of a workgroup read every V row of a KV tile, and the next
    tile's copy overwrites those rows. A wave that starts that copy before the
    others finish reading corrupts their outputs. With one split, each
    workgroup walks up to kv_len / 64 tiles, and the ragged lengths stagger
    the workgroups that share a CU."""
    if not _gfx950():
        pytest.skip("pa_decode_gfx950_coop is gfx950-only")
    from mslk.attention.flydsl.decode import pa_decode_gfx950_coop as coop
    from mslk.attention.flydsl.decode.pa_decode_dense import pa_decode_launch

    splits = []
    real = coop._make_gfx950_coop_jit_launcher

    def spy(*args):
        splits.append(args[3])
        return real(*args)

    monkeypatch.setattr(coop, "_make_gfx950_coop_jit_launcher", spy)
    gen = torch.Generator(device="cuda").manual_seed(B * kv_len + H_q)
    for call in range(CALLS):
        q, k, v, seq = _inputs(gen, B, H_q, H_kv, D, kv_len)
        out = pa_decode_launch(q, k, v, seq, 1.0 / math.sqrt(D), split_k=1)
        ref = _reference(q, k, v, seq)
        err = (out.float().view(B, H_q, D) - ref).abs().amax(-1)
        bad = ~(err <= 3e-2 + 2e-2 * ref.abs().amax(-1))
        assert not bad.any(), (
            f"call {call}: {int(bad.sum())} of {bad.numel()} (batch, head) rows "
            f"differ from the reference, max error {err.max().item():.3g}"
        )
    assert splits == [1] * CALLS, f"expected coop launches with one split: {splits}"
