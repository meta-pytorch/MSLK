# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Unit test for fused_mxfp8_quant: verify output matches to_mxfp8 + _to_blocked."""

import unittest

import torch
from mslk.test.gemm.fused_mxfp8_quant import fused_mxfp8_quant


def _reference_to_mxfp8_blocked(x: torch.Tensor):
    """Pure PyTorch reference: BF16 → FP8 E4M3 + blocked E8M0 scales.

    No external deps — just torch ops. Implements the same algorithm as
    to_mxfp8 + _to_blocked but in one function.
    """
    M, K = x.shape
    BLOCK_SIZE = 32
    FP8_MAX = 448.0
    E8M0_BIAS = 127

    # Reshape into blocks of 32
    n_blocks_k = K // BLOCK_SIZE
    x_blocks = x.float().reshape(M, n_blocks_k, BLOCK_SIZE)

    # Per-block amax
    amax = x_blocks.abs().amax(dim=-1, keepdim=True)  # [M, n_blocks_k, 1]
    amax = amax.clamp(min=1e-12)

    # E8M0 biased exponent: ceil(log2(amax / FP8_MAX)) + 127
    log2_scale = torch.log2(amax / FP8_MAX)
    biased_exp = torch.ceil(log2_scale).clamp(-127, 127) + E8M0_BIAS
    e8m0_scale = biased_exp.to(torch.uint8).squeeze(-1)  # [M, n_blocks_k]

    # Descale and quantize to FP8
    descale = torch.exp2(E8M0_BIAS - biased_exp.float())  # 2^(127 - biased_exp)
    scaled = (x_blocks * descale).clamp(-FP8_MAX, FP8_MAX)
    fp8_data = scaled.reshape(M, K).to(torch.float8_e4m3fn)

    # _to_blocked layout transformation (pure reshape+permute)
    n_row_blocks = (M + 127) // 128
    n_col_blocks = (n_blocks_k + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    # Pad scales if needed
    scales_padded = torch.zeros(
        padded_rows, padded_cols, dtype=torch.uint8, device=x.device
    )
    scales_padded[:M, :n_blocks_k] = e8m0_scale

    # Blocked layout: view(n_row_blocks, 4, 32, n_col_blocks, 4)
    #                 .permute(0, 3, 2, 1, 4).flatten()
    blocked = (
        scales_padded.view(n_row_blocks, 4, 32, n_col_blocks, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )

    return fp8_data, blocked


class FusedMxfp8QuantTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = "cuda"

    def _reference(self, x):
        """Reference: pure PyTorch BF16 → FP8 E4M3 + blocked E8M0 scales."""
        return _reference_to_mxfp8_blocked(x)

    def test_basic_shape(self):
        M, K = 16, 1024
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.1
        fp8, scales = fused_mxfp8_quant(x)
        self.assertEqual(fp8.shape, (M, K))
        self.assertEqual(fp8.dtype, torch.float8_e4m3fn)

    def test_scale_layout_matches_reference(self):
        """Verify blocked scale layout matches _to_blocked output."""
        for M in [1, 16, 64, 128, 256]:
            with self.subTest(M=M):
                K = 1024
                x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.1

                ref_fp8, ref_scales = self._reference(x)
                fused_fp8, fused_scales = fused_mxfp8_quant(x)

                # Scales should match exactly (both E8M0 integers)
                ref_flat = ref_scales.flatten()
                fused_flat = fused_scales[: ref_flat.numel()]
                self.assertTrue(
                    torch.equal(ref_flat, fused_flat),
                    f"Scale mismatch at M={M}: "
                    f"first diff at {(ref_flat != fused_flat).nonzero(as_tuple=True)[0][:5]}",
                )

    def test_fp8_values_match_reference(self):
        """Verify FP8 quantized values match reference."""
        for M in [1, 16, 128]:
            with self.subTest(M=M):
                K = 1024
                x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.1

                ref_fp8, _ = self._reference(x)
                fused_fp8, _ = fused_mxfp8_quant(x)

                # Compare as uint8 (exact FP8 bit patterns)
                ref_bits = ref_fp8.view(torch.uint8)
                fused_bits = fused_fp8.view(torch.uint8)
                self.assertTrue(
                    torch.equal(ref_bits, fused_bits),
                    f"FP8 data mismatch at M={M}",
                )

    def test_large_k(self):
        """Test with K=16384 (production size)."""
        M, K = 16, 16384
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.1

        ref_fp8, ref_scales = self._reference(x)
        fused_fp8, fused_scales = fused_mxfp8_quant(x)

        ref_flat = ref_scales.flatten()
        fused_flat = fused_scales[: ref_flat.numel()]
        self.assertTrue(torch.equal(ref_flat, fused_flat), "Scale mismatch at K=16384")
        self.assertTrue(
            torch.equal(ref_fp8.view(torch.uint8), fused_fp8.view(torch.uint8)),
            "FP8 mismatch at K=16384",
        )


if __name__ == "__main__":
    unittest.main()
