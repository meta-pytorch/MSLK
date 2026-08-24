#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Test f4f4bf16_grouped_mm with mxfp4_block_size=16 (offset-based API).

Reproduces the crash seen in MetaShuffling MoE with MXFP4_16:
- Some workers crash with 'illegal memory access'
- Pattern suggests certain M_group_size values trigger the issue
"""

import unittest

import torch

try:
    import mslk.gemm  # noqa: F401

    HAS_MSLK = True
except ImportError:
    HAS_MSLK = False

try:
    from mslk.quantize.triton.fp4_quantize import triton_quantize_mx4_unpack

    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False

try:
    # vLLM owns the fused activation quantizer; mslk does not depend on vllm,
    # so the tests exercising it skip rather than forcing a layering-inverting
    # BUCK dep from mslk onto vllm.
    from vllm.fb.plugins.meta_shuffling_kernels.quantization import (
        mxfp4_quantize_stacked,
    )

    HAS_VLLM_QUANT = True
except ImportError:
    HAS_VLLM_QUANT = False

GROUP_SIZE = 16
# SQNR of the MXFP4 result against the unquantized BF16 GEMM. Real MXFP4
# lands near 16dB on this data; a kernel reading the wrong scales or segments
# collapses to ~0dB or below. 10dB sits between the two with margin.
MIN_SQNR_DB = 10.0


def _quantize_per_expert(data, counts, bs=GROUP_SIZE):
    """Quantize per-expert with correct padding (matching fused kernel)."""
    xq_parts, xs_parts = [], []
    offset = 0
    for c in counts:
        if c > 0:
            end = offset + c
            xq_i, xs_i = triton_quantize_mx4_unpack(data[offset:end], group_size=bs)
            xq_parts.append(xq_i)
            xs_parts.append(xs_i)
        offset += c
    return torch.cat(xq_parts, dim=0), torch.cat(xs_parts, dim=0)


def _sqnr(ref, got):
    """Signal-to-quantization-noise ratio in dB; inf when bit-identical."""
    r, g = ref.float(), got.float()
    noise = (r - g).pow(2).mean()
    if noise == 0:
        return float("inf")
    return float(10.0 * torch.log10(r.pow(2).mean() / noise))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not HAS_MSLK, "mslk not available")
@unittest.skipIf(not HAS_QUANT, "mslk.quantize not available")
class TestMXFP4_16GroupedMM(unittest.TestCase):
    """Test f4f4bf16_grouped_mm with mxfp4_block_size=16."""

    def _run_test(self, E, tokens_per_expert, N, K):
        device = torch.device("cuda:0")
        total_M = sum(tokens_per_expert)
        counts = list(tokens_per_expert)

        # Create data
        x_bf16 = torch.randn(total_M, K, dtype=torch.bfloat16, device=device) * 0.01
        W_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.01

        # Quantize activations per-expert
        xq, x_scale = _quantize_per_expert(x_bf16, counts, GROUP_SIZE)

        # Quantize weights per-expert
        wq_parts, ws_parts = [], []
        for e in range(E):
            wq_i, ws_i = triton_quantize_mx4_unpack(W_bf16[e], group_size=GROUP_SIZE)
            wq_parts.append(wq_i)
            ws_parts.append(ws_i)
        wq = torch.stack(wq_parts).contiguous()
        w_scale = torch.stack(ws_parts).contiguous()

        # Compute offsets (cumulative sum of counts), must be int32
        offsets = (
            torch.tensor(counts, dtype=torch.int64, device=device)
            .cumsum(0)
            .to(torch.int32)
        )

        print(f"  E={E}, tokens={counts}, N={N}, K={K}")
        print(f"  xq={xq.shape}, x_scale={x_scale.shape}")
        print(f"  wq={wq.shape}, w_scale={w_scale.shape}")
        print(f"  offsets={offsets.tolist()}")

        torch.cuda.synchronize()
        out = torch.ops.mslk.f4f4bf16_grouped_mm(
            xq.view(torch.float4_e2m1fn_x2),
            wq.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
            x_scale.view(torch.float8_e8m0fnu),
            w_scale.view(torch.float8_e8m0fnu),
            offsets,
            global_scale=None,
            mxfp4_block_size=16,
        )
        torch.cuda.synchronize()

        self.assertTrue(out.isfinite().all(), "Output has non-finite values")

        # Accuracy: compare against the unquantized BF16 GEMM. MXFP4 is
        # 4-bit, so some error is expected -- the point is that it lands in
        # the band real quantization produces. A kernel that mis-indexes
        # scales or segments still returns finite output, so finiteness alone
        # says nothing; those failures show up here as ~0dB or negative.
        ref = torch.empty(out.shape, dtype=torch.float32, device=out.device)
        start = 0
        for e, c in enumerate(counts):
            if c == 0:
                continue
            stop = start + c
            ref[start:stop] = x_bf16[start:stop].float() @ W_bf16[e].float().T
            start += c

        sqnr = _sqnr(ref, out)
        print(
            f"  OK: out={out.shape}, mean={out.float().mean():.6f}, sqnr={sqnr:.1f}dB"
        )
        self.assertGreater(
            sqnr,
            MIN_SQNR_DB,
            f"MXFP4_16 grouped MM disagrees with its dequantized reference "
            f"(sqnr={sqnr:.1f}dB); scales or segment indexing are likely wrong",
        )

    def test_uniform_small(self):
        """E=4, 64 tokens per expert."""
        print("\ntest_uniform_small:")
        self._run_test(4, [64, 64, 64, 64], 4096, 4096)

    def test_uniform_large(self):
        """E=4, 256 tokens per expert."""
        print("\ntest_uniform_large:")
        self._run_test(4, [256, 256, 256, 256], 4096, 4096)

    def test_single_token(self):
        """E=4, 1 token to one expert, 0 to others."""
        print("\ntest_single_token:")
        self._run_test(4, [1, 0, 0, 0], 4096, 4096)

    def test_sparse_like_warmup(self):
        """E=32, 1 token routed to 1 expert (mimics warmup)."""
        print("\ntest_sparse_like_warmup:")
        counts = [0] * 32
        counts[5] = 1  # one expert gets 1 token
        self._run_test(32, counts, 4096, 4096)

    def test_sparse_multi_expert(self):
        """E=32, tokens spread across 8 experts."""
        print("\ntest_sparse_multi_expert:")
        counts = [0] * 32
        for i in [0, 4, 8, 12, 16, 20, 24, 28]:
            counts[i] = 1
        self._run_test(32, counts, 4096, 4096)

    def test_n8192(self):
        """Test with N=8192 (w13 GEMM dimension)."""
        print("\ntest_n8192:")
        self._run_test(4, [64, 64, 64, 64], 8192, 4096)

    def test_mixed_counts(self):
        """E=32 with varying token counts."""
        print("\ntest_mixed_counts:")
        counts = [0] * 32
        counts[0] = 100
        counts[3] = 50
        counts[7] = 200
        counts[15] = 10
        counts[31] = 1
        self._run_test(32, counts, 4096, 4096)

    def test_m128_per_expert(self):
        """E=4, 128 tokens per expert (border of heuristic)."""
        print("\ntest_m128_per_expert:")
        self._run_test(4, [128, 128, 128, 128], 4096, 4096)

    def test_m192_per_expert(self):
        """E=4, 192 tokens per expert."""
        print("\ntest_m192_per_expert:")
        self._run_test(4, [192, 192, 192, 192], 4096, 4096)

    def test_m256_e1(self):
        """E=1, 256 tokens (pure GEMM, no expert split)."""
        print("\ntest_m256_e1:")
        self._run_test(1, [256], 4096, 4096)

    def test_m512_per_expert(self):
        """E=4, 512 tokens per expert."""
        print("\ntest_m512_per_expert:")
        self._run_test(4, [512, 512, 512, 512], 4096, 4096)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not HAS_MSLK, "mslk not available")
@unittest.skipIf(not HAS_QUANT, "mslk.quantize not available")
@unittest.skipIf(not HAS_VLLM_QUANT, "vllm meta_shuffling quantization not available")
class TestFusedMXFP4QuantizeStacked(unittest.TestCase):
    """Test fused mxfp4_quantize_stacked kernel (the cudagraph-compatible
    activation quantization) matches per-expert triton_quantize_mx4_unpack."""

    def _compare_fused_vs_per_expert(self, E, tokens_per_expert, K, bs=16):
        """Compare fused stacked kernel vs per-expert by running grouped_mm
        with both and checking that results are close to BF16 reference."""
        device = torch.device("cuda:0")
        total_M = sum(tokens_per_expert)
        counts = list(tokens_per_expert)
        m_sizes = torch.tensor(counts, dtype=torch.int64, device=device)
        N = 4096

        x_bf16 = torch.randn(total_M, K, dtype=torch.bfloat16, device=device) * 0.01
        W_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.01

        # Quantize weights (shared between both paths)
        wq_parts, ws_parts = [], []
        for e in range(E):
            wq_i, ws_i = triton_quantize_mx4_unpack(W_bf16[e], group_size=bs)
            wq_parts.append(wq_i)
            ws_parts.append(ws_i)
        wq = torch.stack(wq_parts).contiguous()
        w_scale = torch.stack(ws_parts).contiguous()
        offsets = m_sizes.cumsum(0).to(torch.int32)

        # Path A: per-expert quant (reference)
        ref_xq, ref_xs = _quantize_per_expert(x_bf16, counts, bs)
        torch.cuda.synchronize()
        out_ref = torch.ops.mslk.f4f4bf16_grouped_mm(
            ref_xq.view(torch.float4_e2m1fn_x2),
            wq.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
            ref_xs.view(torch.float8_e8m0fnu),
            w_scale.view(torch.float8_e8m0fnu),
            offsets,
            global_scale=None,
            mxfp4_block_size=bs,
        )
        torch.cuda.synchronize()

        # Path B: fused stacked kernel
        fused_xq, fused_xs = mxfp4_quantize_stacked(m_sizes, x_bf16, bs)
        torch.cuda.synchronize()
        out_fused = torch.ops.mslk.f4f4bf16_grouped_mm(
            fused_xq.view(torch.float4_e2m1fn_x2),
            wq.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
            fused_xs.view(torch.float8_e8m0fnu),
            w_scale.view(torch.float8_e8m0fnu),
            offsets,
            global_scale=None,
            mxfp4_block_size=bs,
        )
        torch.cuda.synchronize()

        # BF16 reference
        out_bf16 = torch._grouped_mm(
            x_bf16,
            W_bf16.transpose(-2, -1),
            offs=offsets,
            out_dtype=torch.bfloat16,
        )

        # Both quantized paths should be close to BF16
        self.assertTrue(out_ref.isfinite().all(), "ref output non-finite")
        self.assertTrue(out_fused.isfinite().all(), "fused output non-finite")

        # Check fused vs ref are very close (both approximate BF16)
        torch.testing.assert_close(out_fused, out_ref, atol=1e-1, rtol=1e-1)
        # Check both approximate BF16
        torch.testing.assert_close(out_ref, out_bf16, atol=1.0, rtol=0.5)

        ref_err = (out_ref - out_bf16).float().abs().mean()
        fused_err = (out_fused - out_bf16).float().abs().mean()
        print(
            f"  E={E}, tokens={counts}, K={K}, bs={bs}: "
            f"ref_err={ref_err:.4f}, fused_err={fused_err:.4f} -- OK"
        )

    def test_fused_uniform_e4(self):
        print("\ntest_fused_uniform_e4:")
        self._compare_fused_vs_per_expert(4, [64, 64, 64, 64], 4096, bs=16)

    def test_fused_sparse_e32(self):
        print("\ntest_fused_sparse_e32:")
        counts = [0] * 32
        counts[0] = 10
        counts[15] = 5
        counts[31] = 20
        self._compare_fused_vs_per_expert(32, counts, 4096, bs=16)

    def test_fused_large_m(self):
        print("\ntest_fused_large_m:")
        self._compare_fused_vs_per_expert(4, [256, 256, 256, 256], 4096, bs=16)

    def test_fused_bs32(self):
        """Also verify bs32 still works after buffer allocation change."""
        print("\ntest_fused_bs32:")
        self._compare_fused_vs_per_expert(4, [64, 64, 64, 64], 4096, bs=32)

    def test_fused_single_token(self):
        print("\ntest_fused_single_token:")
        self._compare_fused_vs_per_expert(4, [1, 0, 0, 0], 4096, bs=16)

    def test_fused_then_grouped_mm(self):
        """End-to-end: fused quant -> grouped_mm with bs16."""
        print("\ntest_fused_then_grouped_mm:")
        device = torch.device("cuda:0")
        E, M_per, N, K, bs = 4, 64, 4096, 4096, 16

        x_bf16 = torch.randn(E * M_per, K, dtype=torch.bfloat16, device=device) * 0.01
        W_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.01

        m_sizes = torch.full((E,), M_per, dtype=torch.int64, device=device)

        # Fused activation quant
        xq, x_scale = mxfp4_quantize_stacked(m_sizes, x_bf16, bs)

        # Weight quant per-expert
        wq_parts, ws_parts = [], []
        for e in range(E):
            wq_i, ws_i = triton_quantize_mx4_unpack(W_bf16[e], group_size=bs)
            wq_parts.append(wq_i)
            ws_parts.append(ws_i)
        wq = torch.stack(wq_parts).contiguous()
        w_scale = torch.stack(ws_parts).contiguous()

        offsets = m_sizes.cumsum(0).to(torch.int32)

        torch.cuda.synchronize()
        out = torch.ops.mslk.f4f4bf16_grouped_mm(
            xq.view(torch.float4_e2m1fn_x2),
            wq.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
            x_scale.view(torch.float8_e8m0fnu),
            w_scale.view(torch.float8_e8m0fnu),
            offsets,
            global_scale=None,
            mxfp4_block_size=bs,
        )
        torch.cuda.synchronize()
        self.assertTrue(out.isfinite().all())
        print(f"  out={out.shape}, mean={out.float().mean():.6f} -- OK")
