# @nolint # fbcode
"""Tests for CuteDSL Blockscale GEMM (MXFP8) kernel."""

import pytest
import torch

from mslk.gemm.cutedsl import mxfp8_gemm
from mslk.gemm.triton.fp8_gemm import to_mxfp8
from mslk.quantize.triton.fp4_quantize import _to_blocked
from mslk.quantize.triton.mxfp8_quantize import triton_quantize_mxfp8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell (SM100+)"
)
class TestBlockscaleGemmMXFP8:
    """Tests for CuteDSL blockscale GEMM with MXFP8."""

    def _run_mxfp8_gemm(
        self,
        M: int,
        N: int,
        K: int,
        atol: float = 8e-2,
        rtol: float = 8e-2,
    ):
        """Helper to run MXFP8 GEMM and validate against dequantized reference."""
        torch.manual_seed(42)
        X = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
        W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1

        # Quantize to MXFP8
        x_scale, xq = to_mxfp8(X)
        w_scale, wq = to_mxfp8(W)

        # Run CuteDSL kernel
        result = mxfp8_gemm(
            A=xq,
            B=wq,
            A_scale=x_scale,
            B_scale=w_scale,
        )

        # Dequantized reference: use quantized inputs to isolate kernel correctness
        # from quantization error
        def _dequantize_mxfp8(data_fp8, scale_e8m0):
            """Dequantize MXFP8 data using E8M0 scales."""
            data_bf16 = data_fp8.to(torch.bfloat16)
            scale_f32 = scale_e8m0.view(torch.uint8).to(torch.float32)
            scale_f32 = torch.pow(2.0, scale_f32 - 127.0)
            scale_bf16 = scale_f32.to(torch.bfloat16)
            # Expand scales from (MN, K//32) -> (MN, K)
            scale_expanded = scale_bf16.repeat_interleave(32, dim=1)
            return data_bf16 * scale_expanded

        X_deq = _dequantize_mxfp8(xq, x_scale)
        W_deq = _dequantize_mxfp8(wq, w_scale)
        ref = torch.mm(X_deq, W_deq.t())

        # Validate
        torch.testing.assert_close(result, ref, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "M,N,K",
        [
            pytest.param(128, 128, 256, id="small_square"),
            pytest.param(256, 256, 512, id="medium_square"),
            pytest.param(512, 512, 1024, id="large_square"),
            pytest.param(128, 256, 512, id="rectangular_1"),
            pytest.param(256, 128, 1024, id="rectangular_2"),
            pytest.param(512, 1024, 2048, id="large_rectangular"),
        ],
    )
    def test_basic_shapes(self, M, N, K):
        """Test basic shapes with auto-selected tile settings."""
        self._run_mxfp8_gemm(M, N, K)

    @pytest.mark.parametrize(
        "M,N,K",
        [
            # Llama4 decoder shapes
            pytest.param(128, 7168, 8192, id="llama4_qkv_proj_128"),
            pytest.param(256, 7168, 8192, id="llama4_qkv_proj_256"),
            pytest.param(128, 8192, 7168, id="llama4_out_proj_128"),
            pytest.param(128, 16384, 8192, id="llama4_ffn_up_128"),
            pytest.param(128, 8192, 16384, id="llama4_ffn_down_128"),
        ],
    )
    def test_llama4_shapes(self, M, N, K):
        """Test Llama4-relevant shapes."""
        self._run_mxfp8_gemm(M, N, K)

    @pytest.mark.parametrize(
        "M,N,K",
        [
            pytest.param(128, 128, 256, id="small_square"),
            pytest.param(256, 256, 512, id="medium_square"),
            pytest.param(128, 256, 512, id="rectangular"),
        ],
    )
    def test_triton_quantize_mxfp8_matches_reference(self, M, N, K):
        """Verify triton_quantize_mxfp8 produces same results as to_mxfp8 + _to_blocked."""
        torch.manual_seed(42)
        X = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1

        # Reference: to_mxfp8 then _to_blocked
        ref_scale, ref_data = to_mxfp8(X)
        ref_scale_blocked = _to_blocked(ref_scale)

        # Fused: triton_quantize_mxfp8
        fused_scale, fused_data = triton_quantize_mxfp8(X)

        # FP8 data should be identical
        torch.testing.assert_close(
            fused_data.to(torch.float32), ref_data.to(torch.float32),
            atol=0, rtol=0,
        )
        # Blocked scales should be identical
        torch.testing.assert_close(
            fused_scale.view(torch.uint8).to(torch.int32),
            ref_scale_blocked.view(torch.uint8).to(torch.int32),
            atol=0, rtol=0,
        )

    @pytest.mark.parametrize(
        "M,N,K",
        [
            pytest.param(128, 128, 256, id="small_square"),
            pytest.param(256, 256, 512, id="medium_square"),
            pytest.param(512, 512, 1024, id="large_square"),
            pytest.param(128, 256, 512, id="rectangular_1"),
        ],
    )
    def test_gemm_with_fused_quantize(self, M, N, K):
        """Test GEMM using triton_quantize_mxfp8 (fused quantization path)."""
        torch.manual_seed(42)
        X = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
        W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1

        # Quantize using fused path
        x_scale, xq = triton_quantize_mxfp8(X)
        w_scale, wq = triton_quantize_mxfp8(W)

        # Run CuteDSL kernel with pre-blocked scales
        result = mxfp8_gemm(A=xq, B=wq, A_scale=x_scale, B_scale=w_scale)

        # Reference: quantize with original to_mxfp8
        ref_x_scale, ref_xq = to_mxfp8(X)
        ref_w_scale, ref_wq = to_mxfp8(W)
        ref_result = mxfp8_gemm(
            A=ref_xq, B=ref_wq, A_scale=ref_x_scale, B_scale=ref_w_scale,
        )

        # Both paths should produce identical GEMM output
        torch.testing.assert_close(result, ref_result, atol=0, rtol=0)
