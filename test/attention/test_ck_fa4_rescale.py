# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

"""
Focused test for FA4 Eq. 6 conditional rescaling in CK FMHA.

Tests that the conditional rescaling optimization (threshold tau = log2(256) = 8.0)
produces correct results by comparing against PyTorch reference attention.

The key insight: when the running max changes by <= tau between blocks, we can skip
rescaling O_acc and instead correct P. This test exercises both branches:
- Rescale branch: large max change between blocks (e.g., first block)
- Skip branch: small max change between blocks (uniform-ish attention scores)
"""

import math
import unittest

import torch


def ref_attention(q, k, v, scale=None):
    """Reference attention implementation using PyTorch.

    Input format: BMHK (batch, seqlen, heads, head_dim).
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    # Transpose to BHMK for matmul: (B, M, H, K) -> (B, H, M, K)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # (B, H, M, M)
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_t)  # (B, H, M, K)
    return out.transpose(1, 2)  # Back to BMHK


class CkFa4RescaleTest(unittest.TestCase):
    """Test CK FMHA forward pass correctness with FA4 conditional rescaling."""

    def _run_ck_fmha(self, q, k, v):
        """Run CK FMHA forward pass."""
        from mslk.attention import fmha

        out = fmha.memory_efficient_attention_forward(
            q, k, v, attn_bias=None, op=fmha.ck.FwOp
        )
        return out

    def _compare(self, q, k, v, atol, rtol, msg=""):
        """Compare CK FMHA output against reference."""
        ref = ref_attention(q.float(), k.float(), v.float()).to(q.dtype)
        out = self._run_ck_fmha(q, k, v)

        self.assertFalse(out.isnan().any(), f"Output has NaNs {msg}")

        max_diff = (out.float() - ref.float()).abs().max().item()
        mean_diff = (out.float() - ref.float()).abs().mean().item()
        print(f"  {msg}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        torch.testing.assert_close(
            out.float(),
            ref.float(),
            atol=atol,
            rtol=rtol,
            msg=lambda m: f"{msg}: {m}",
        )

    def test_uniform_scores_bf16(self):
        """Uniform Q/K: all blocks have similar max -> skip branch dominant."""
        torch.manual_seed(42)
        B, H, M, K = 2, 8, 1024, 128
        # Small uniform values -> max changes slowly between blocks
        q = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
        k = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
        v = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        self._compare(q, k, v, atol=2e-2, rtol=1e-2, msg="uniform_bf16")

    def test_large_seqlen_bf16(self):
        """Long sequence: exercises many KV blocks, both branches."""
        torch.manual_seed(123)
        B, H, M, K = 1, 4, 4096, 64
        q = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        self._compare(q, k, v, atol=3e-2, rtol=1e-2, msg="large_seqlen_bf16")

    def test_spike_pattern_bf16(self):
        """Spike pattern: one KV position has much larger dot product.

        Forces rescale branch on the block containing the spike,
        and skip branch on surrounding blocks.
        """
        torch.manual_seed(99)
        B, H, M, K = 2, 4, 2048, 128
        q = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
        k = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16) * 0.1
        v = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        # Insert spike at position 512 -> forces rescale when that block is hit
        k[:, 512, :, :] = 10.0
        self._compare(q, k, v, atol=3e-2, rtol=1e-2, msg="spike_bf16")

    def test_multiple_spikes_bf16(self):
        """Multiple spikes at different KV positions.

        Creates a pattern where some blocks trigger rescale and others don't,
        exercising the interleaving of both branches.
        """
        torch.manual_seed(77)
        B, H, M, K = 2, 4, 4096, 128
        q = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16) * 0.05
        k = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16) * 0.05
        v = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        # Spikes at various positions
        for pos in [256, 1024, 2048, 3072]:
            k[:, pos, :, :] = 8.0
        self._compare(q, k, v, atol=3e-2, rtol=1e-2, msg="multi_spike_bf16")

    def test_fp16(self):
        """FP16 precision test."""
        torch.manual_seed(42)
        B, H, M, K = 2, 8, 2048, 64
        q = torch.randn(B, M, H, K, device="cuda", dtype=torch.float16)
        k = torch.randn(B, M, H, K, device="cuda", dtype=torch.float16)
        v = torch.randn(B, M, H, K, device="cuda", dtype=torch.float16)
        self._compare(q, k, v, atol=2e-2, rtol=1e-2, msg="fp16")

    def test_deterministic(self):
        """Verify deterministic output across multiple runs."""
        torch.manual_seed(42)
        B, H, M, K = 1, 4, 2048, 128
        q = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        out1 = self._run_ck_fmha(q, k, v)
        out2 = self._run_ck_fmha(q, k, v)
        self.assertTrue(
            torch.equal(out1, out2),
            f"Non-deterministic: max_diff={((out1 - out2).abs().max().item())}",
        )

    def test_perf_benefit(self):
        """Measure performance of CK FMHA to verify no regression.

        The FA4 conditional rescaling should improve or maintain perf
        by skipping unnecessary exp2+multiply operations.
        """
        torch.manual_seed(42)
        B, H, M, K = 4, 32, 4096, 128
        q = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, M, H, K, device="cuda", dtype=torch.bfloat16)

        # Warmup
        for _ in range(5):
            self._run_ck_fmha(q, k, v)
        torch.cuda.synchronize()

        # Benchmark
        n_iters = 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iters):
            self._run_ck_fmha(q, k, v)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / n_iters
        # FLOPs: 2 * B * H * M * M * K (QK^T) + 2 * B * H * M * M * K (PV)
        flops = 4 * B * H * M * M * K
        tflops = flops / (elapsed_ms * 1e-3) / 1e12
        print(f"\n  CK FMHA Perf: {elapsed_ms:.2f} ms/iter, {tflops:.1f} TFLOPS")
        print(f"  Shape: B={B}, H={H}, M={M}, K={K}, dtype=bf16")
