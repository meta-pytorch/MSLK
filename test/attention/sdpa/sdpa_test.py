# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for mslk.attention.sdpa — SDPA ops matching aten contracts."""

import pytest
import torch
from mslk.attention.sdpa import sdpa_efficient_fwd, sdpa_flash_fwd

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA/ROCm",
)


def _ref_sdpa(Q, K, V, scale=None, is_causal=False):
    """Reference SDPA via PyTorch (BHLD layout, fp32)."""
    return torch.nn.functional.scaled_dot_product_attention(
        Q.float(),
        K.float(),
        V.float(),
        scale=scale,
        is_causal=is_causal,
    )


@skip_no_cuda
class TestSdpaFlashForward:
    """Flash-attention forward: aten-contract output shape and correctness."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize(
        "B,H,Lq,Lkv,D",
        [
            (1, 1, 32, 32, 64),
            (2, 4, 16, 64, 128),
            (1, 2, 32, 32, 256),
        ],
    )
    @pytest.mark.parametrize("is_causal", [False, True])
    def test_flash_fwd_shape_and_correctness(self, dtype, B, H, Lq, Lkv, D, is_causal):
        if is_causal and Lq > Lkv:
            pytest.skip("causal requires Lq <= Lkv")

        Q = torch.randn(B, H, Lq, D, device="cuda", dtype=dtype)
        K = torch.randn(B, H, Lkv, D, device="cuda", dtype=dtype)
        V = torch.randn(B, H, Lkv, D, device="cuda", dtype=dtype)

        result = sdpa_flash_fwd(Q, K, V, is_causal=is_causal)
        out, lse = result[0], result[1]

        assert out.shape == (B, H, Lq, D)
        assert out.dtype == dtype

        ref = _ref_sdpa(Q, K, V, is_causal=is_causal)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)

    def test_flash_fwd_rejects_dropout(self):
        Q = torch.randn(1, 1, 16, 64, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(NotImplementedError, match="dropout"):
            sdpa_flash_fwd(Q, Q, Q, dropout_p=0.1)


@skip_no_cuda
class TestSdpaEfficientForward:
    """Efficient-attention forward: aten-contract output shape and correctness."""

    @pytest.mark.parametrize(
        "B,H,Lq,Lkv,D",
        [
            (1, 1, 32, 32, 64),
            (2, 4, 16, 64, 128),
        ],
    )
    def test_efficient_fwd(self, B, H, Lq, Lkv, D):
        Q = torch.randn(B, H, Lq, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, H, Lkv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, H, Lkv, D, device="cuda", dtype=torch.bfloat16)

        out, lse, _, _ = sdpa_efficient_fwd(
            Q, K, V, attn_bias=None, compute_log_sumexp=True
        )

        assert out.shape == (B, H, Lq, D)
        ref = _ref_sdpa(Q, K, V)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)


@skip_no_cuda
class TestSdpaCrossAttention:
    """Lq != Lkv cross-attention."""

    def test_cross_attn_noncausal(self):
        Q = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)

        out, _, _, _, _, _, _, _, _ = sdpa_flash_fwd(Q, K, V, is_causal=False)
        ref = _ref_sdpa(Q, K, V, is_causal=False)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)

    def test_cross_attn_causal_bottom_right(self):
        """Causal with Lq < Lkv uses bottom-right alignment."""
        Q = torch.randn(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(1, 2, 64, 128, device="cuda", dtype=torch.bfloat16)

        out, _, _, _, _, _, _, _, _ = sdpa_flash_fwd(Q, K, V, is_causal=True)
        ref = _ref_sdpa(Q, K, V, is_causal=True)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)


@skip_no_cuda
class TestSdpaKisV:
    """K=V shared tensor case."""

    def test_flash_k_is_v(self):
        Q = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)
        KV = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.bfloat16)

        out, _, _, _, _, _, _, _, _ = sdpa_flash_fwd(Q, KV, KV)
        ref = _ref_sdpa(Q, KV, KV)
        torch.testing.assert_close(out.float(), ref, atol=0.03, rtol=0.02)


@skip_no_cuda
class TestSdpaAtenOverride:
    """Test enable/disable of aten override."""

    def test_override_api_exists(self):
        from mslk.attention.sdpa import disable_aten_override, enable_aten_override

        assert callable(enable_aten_override)
        assert callable(disable_aten_override)
