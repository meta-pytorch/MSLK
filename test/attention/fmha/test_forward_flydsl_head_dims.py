# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from mslk.attention import fmha
from mslk.attention.fmha import flydsl
from mslk.flydsl.common import is_flydsl_available

from .utils import assert_allclose, ref_attention_bmhk_for_test, rocm_only


def _head_dims_case(route, K, Kv, dtype):
    ab = fmha.attn_bias
    H = 4

    def rand(M, D):
        return torch.randn(
            1 if route in ("varlen", "padded") else 2,
            M,
            H,
            D,
            device="cuda",
            dtype=dtype,
        )

    if route in ("dense", "causal", "splitk"):
        # A 64-row query underfills the GPU, so dense routes to generic split-K.
        M = 64 if route == "splitk" else 300
        bias = ab.LowerTriangularMask() if route == "causal" else None
        return rand(M, K), rand(M, K), rand(M, Kv), bias
    if route == "varlen":
        lens = [100, 300, 37]
        bias = ab.BlockDiagonalCausalMask.from_seqlens(lens)
        return rand(sum(lens), K), rand(sum(lens), K), rand(sum(lens), Kv), bias
    assert route == "padded"
    bias = ab.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[30, 64], kv_padding=256, kv_seqlen=[200, 256]
    )
    return rand(94, K), rand(512, K), rand(512, Kv), bias


@rocm_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("route", ["dense", "causal", "splitk", "varlen", "padded"])
@pytest.mark.parametrize(
    "K,Kv", [(96, 96), (40, 40), (64, 128), (128, 64), (200, 160), (2, 6), (3, 5)]
)
def test_forward_flydsl_head_dims(K, Kv, route, dtype):
    # Non-native head dims and value head_dim != query head_dim run without
    # padded copies: the kernel masks the columns up to its compute width.
    if not is_flydsl_available():
        pytest.skip("FlyDSL is not available")
    torch.manual_seed(0)
    q, k, v, bias = _head_dims_case(route, K, Kv, dtype)
    assert flydsl.FwOp.not_supported_reasons(fmha.Inputs(q, k, v, bias)) == []
    out = fmha.memory_efficient_attention_forward(q, k, v, bias, op=flydsl.FwOp)
    assert out.shape == (*q.shape[:-1], Kv)
    ref = ref_attention_bmhk_for_test(q, k, v, bias)
    assert_allclose(
        out.float(),
        ref,
        atol=flydsl.FwOp.ERROR_ATOL[dtype],
        rtol=flydsl.FwOp.ERROR_RTOL[dtype],
    )


@rocm_only
@pytest.mark.parametrize("route", ["dense", "causal", "varlen"])
@pytest.mark.parametrize("K", [40, 96, 160])
def test_forward_flydsl_head_dims_match_padding(K, route):
    # Masking the missing columns in-kernel is bit-identical to zero-padding
    # them with a copy (and pinning the scale to the unpadded head dim).
    if not is_flydsl_available():
        pytest.skip("FlyDSL is not available")
    torch.manual_seed(0)
    q, k, v, bias = _head_dims_case(route, K, K, torch.bfloat16)
    out = fmha.memory_efficient_attention_forward(q, k, v, bias, op=flydsl.FwOp)
    Dpad = 64 if K <= 64 else (128 if K <= 128 else 256)
    qp, kp, vp = (torch.nn.functional.pad(t, (0, Dpad - K)) for t in (q, k, v))
    padded = fmha.memory_efficient_attention_forward(
        qp, kp, vp, bias, scale=K**-0.5, op=flydsl.FwOp
    )
    assert torch.equal(out, padded[..., :K])


@rocm_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("K", [72, 96, 160])
def test_forward_flydsl_paged_gappy_head_dims(monkeypatch, K, dtype):
    # Paged-gappy at a non-native head dim reads the cache in place: the light
    # kernel is built at the next native width and masks the rest, rather than
    # the op copying the cache out to a padded one.
    if not is_flydsl_available():
        pytest.skip("FlyDSL is not available")
    from mslk.attention.flydsl import flash_attn_interface as fai

    builds = []
    real = fai._build_paged_light

    def spy(*args, **kwargs):
        builds.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(fai, "_build_paged_light", spy)
    bias_type = fmha.attn_bias.PagedBlockDiagonalGappyKeysMask
    page_size = 64
    block_tables = torch.tensor(
        [[3, 1, 5, 0], [4, 6, 2, 7]], device="cuda", dtype=torch.int32
    )
    unpaged = fmha.attn_bias.BlockDiagonalGappyKeysMask.from_seqlens(
        q_seqlen=[11, 21],
        kv_seqstarts=[5, 256 + 17, 512],
        kv_seqlen=[100, 190],
    )
    bias = unpaged.make_paged(
        block_tables=block_tables,
        page_size=page_size,
        notional_padding=256,
        paged_type=bias_type,
    )
    q = torch.randn(1, 32, 1, K, device="cuda", dtype=dtype)
    k = torch.randn(1, 8 * page_size, 1, K, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    out = fmha.memory_efficient_attention_forward(q, k, v, bias, op=flydsl.FwOp)
    ref = ref_attention_bmhk_for_test(q, k, v, bias)
    assert_allclose(
        out.float(),
        ref,
        atol=flydsl.FwOp.ERROR_ATOL[dtype],
        rtol=flydsl.FwOp.ERROR_RTOL[dtype],
    )
    D = 128 if K <= 128 else 256
    assert builds and all(
        c["head_dim"] == D and c["head_dim_qk"] == K for c in builds
    ), builds
