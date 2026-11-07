# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Non-causal sliding windows on dense and varlen attention: flash-attention's
window_size=(window_left, -1), i.e. keep keys >= q_pos - window_left with no
right bound, rows bottom-right aligned. Checked against fp32."""

import math

import pytest
import torch
from mslk.attention.flydsl import flydsl_flash_attn_func
from mslk.flydsl.common import is_flydsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flydsl_available(),
    reason="requires a ROCm GPU with FlyDSL",
)


def _reference(q, k, v, window_left, scale=None):
    """fp32 [B, Sq, H, D] attention keeping keys >= pos - window_left, where
    pos = i + Skv - Sq; returns (out, natural-log LSE [B, H, Sq])."""
    group = q.shape[2] // k.shape[2]
    kf = k.float().repeat_interleave(group, dim=2)
    vf = v.float().repeat_interleave(group, dim=2)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kf) * scale
    sq, skv = q.shape[1], k.shape[1]
    pos = torch.arange(sq, device=q.device)[:, None] + (skv - sq)
    keep = torch.arange(skv, device=q.device)[None, :] >= pos - window_left
    s = s.masked_fill(~keep, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), vf), torch.logsumexp(
        s, -1
    )


def _close(out, ref):
    err = (out.float() - ref).abs().max().item()
    assert math.isfinite(err) and err < 2e-2 * max(1.0, ref.abs().max().item()), err


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize(
    "Sq,Skv,window_left",
    [
        (256, 256, 0),  # each row keeps itself and every later key
        (256, 256, 31),
        (384, 384, 100),
        (128, 1000, 17),  # cross, Sq < Skv
        (300, 100, 20),  # Sq > Skv: early rows sit before the KV start
        (200, 200, 5000),  # wider than the sequence: full attention
    ],
)
def test_dense_noncausal_window(dtype, D, Sq, Skv, window_left):
    torch.manual_seed(0)
    q = torch.randn(2, Sq, 8, D, device="cuda", dtype=dtype)
    k = torch.randn(2, Skv, 2, D, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    out = flydsl_flash_attn_func(
        q, k, v, causal=False, num_kv_heads=2, window_left=window_left
    )
    _close(out, _reference(q, k, v, window_left)[0])


def test_dense_noncausal_window_lse_scale_bias():
    torch.manual_seed(0)
    q = torch.randn(2, 384, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 384, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    bias = torch.randn(2, 8, 384, 384, device="cuda", dtype=torch.bfloat16)
    out, lse = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=False,
        window_left=64,
        sm_scale=0.1,
        bias=bias,
        return_lse=True,
    )
    kf, vf = k.float(), v.float()
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kf) * 0.1 + bias.float()
    pos = torch.arange(384, device="cuda")[:, None]
    s = s.masked_fill(
        ~(torch.arange(384, device="cuda")[None, :] >= pos - 64), -math.inf
    )
    _close(out, torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), vf))
    assert (lse - torch.logsumexp(s, -1)).abs().max().item() < 1e-2


def test_varlen_noncausal_window():
    torch.manual_seed(0)
    q_lens, kv_lens = [7, 64, 130, 1], [1000, 64, 2000, 50]
    q = torch.randn(sum(q_lens), 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(kv_lens), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    def cu(lens):
        return torch.tensor([0] + lens, device="cuda").cumsum(0).to(torch.int32)

    cq, ck = cu(q_lens), cu(kv_lens)
    out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=False,
        num_kv_heads=4,
        cu_seqlens_q=cq,
        cu_seqlens_kv=ck,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        cross_seqlen=True,
        window_left=33,
    )
    for b in range(len(q_lens)):
        sq = slice(int(cq[b]), int(cq[b + 1]))
        sk = slice(int(ck[b]), int(ck[b + 1]))
        ref = _reference(q[None, sq], k[None, sk], v[None, sk], 33)[0][0]
        _close(out[sq], ref)
