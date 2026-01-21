# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import unittest
from typing import Optional, Tuple

import mslk.quantize  # noqa: F401
import torch
from hypothesis import given, settings, strategies as st
from mslk.quantize.triton.fp4_quantize import (
    _from_blocked,
    _to_blocked,
    cal_global_scale_mx4_as_nvfp4,
    FP4_E2M1_MAX,
    FP4_EBITS,
    FP4_MBITS,
    FP8_E4M3_MAX,
    RoundingMode,
    triton_quantize_mx4_unpack,
    triton_quantize_nvfp4,
    triton_rms_quantize_mx4_unpack,
    triton_scale_nvfp4_quant_rms,
    triton_scale_nvfp4_quant_silu,
    triton_silu_quantize_mx4_unpack,
)
from mslk.quantize.triton.fp4_utils import (
    dequantize_nvfp4,
    fp4_to_float,
    global_scale_nvfp4,
)
from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked, pack_uint4

# pyre-fixme [16]
open_source: bool = getattr(mslk, "open_source", False)

if not open_source:
    from gen_ai.llm_inference.fb.llm.kernel.rms_norm import rms_norm
    from gen_ai.llm_inference.fb.llm.kernel.silu_mul import silu_mul


def scale_mx4(x: torch.Tensor, exp: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    # Input should be dequantized to float, just need to apply shared exponent.
    FP32_EXP_BIAS = 127
    scale = torch.exp2(exp.view(torch.uint8).to(torch.int32) - FP32_EXP_BIAS)
    # View input as chunks of group size.
    orig_shape = x.shape
    num_groups = orig_shape[-1] // group_size
    scaled_x = (
        x.view(-1, num_groups, group_size)
        * scale.view(x.shape[0], -1)[:, :num_groups, None]
    )
    return scaled_x.view(orig_shape)


def sample_scales() -> st.SearchStrategy[Optional[torch.Tensor]]:
    return st.sampled_from(
        [
            None,
            torch.tensor(
                [1.0],
                dtype=torch.float,
                device=torch.accelerator.current_accelerator(),
            ),
        ]
        if torch.cuda.is_available()
        else [None]
    )


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10,
    "Skip when B200 is not available",
)
class TestFp4Quantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_quantize_fp4(self) -> None:
        def _test_quantize_fp4(
            shape: Tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 32
            rounding_mode = RoundingMode.nearest
            num_groups = math.ceil(N / group_size)
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            xq, x_scale = triton_quantize_mx4_unpack(
                x, group_size=group_size, rounding_mode=rounding_mode
            )
            # Convert blocked x_scale format back to (M, num_groups) layout.
            x_scale = _from_blocked(x_scale, (M, num_groups))
            # Dequantize and check that results are similar.
            xq_float = fp4_to_float(xq)
            xq_dequant = scale_mx4(xq_float, x_scale, group_size).to(torch.bfloat16)
            torch.testing.assert_close(xq_dequant, x, atol=1, rtol=1)

        _test_quantize_fp4((1, 128))
        _test_quantize_fp4((3, 512))
        _test_quantize_fp4((128, 1024))
        _test_quantize_fp4((4096, 10240))


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10,
    "Skip when B200 is not available",
)
class TestFp4RmsQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_rms_quantize_fp4(self) -> None:
        def _test_rms_quantize_fp4(
            shape: Tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 32
            rounding_mode = RoundingMode.even
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            xq, x_scale = triton_rms_quantize_mx4_unpack(
                x, w, EPS=1e-5, group_size=group_size, rounding_mode=rounding_mode
            )

            intermediate = (
                x.to(torch.float32).reshape(-1, group_size)
                * torch.rsqrt(
                    torch.pow(x.to(torch.float32).reshape(-1, group_size), 2).mean(
                        dim=1
                    )
                    + 1e-5
                ).unsqueeze(1)
            ) * w.reshape(-1, group_size).to(torch.float32)

            intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
            # Dequantize and check that results are similar.
            x_scale = _from_blocked(x_scale, (M, math.ceil(N / group_size)))
            xq_float = fp4_to_float(xq)
            xq_dequant = scale_mx4(xq_float, x_scale, group_size).to(torch.bfloat16)
            torch.testing.assert_close(xq_dequant, intermediate, atol=1, rtol=1)

        _test_rms_quantize_fp4((1, 32))
        _test_rms_quantize_fp4((1, 128))
        _test_rms_quantize_fp4((3, 512))
        _test_rms_quantize_fp4((128, 1024))
        # TODO: fix potential bug with large tensors
        # _test_rms_quantize_fp4((4096, 10240))


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10,
    "Skip when B200 is not available",
)
class TestFp4SiluQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_silu_quantize_fp4(self) -> None:
        def _test_silu_quantize_fp4(
            shape: Tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 32
            rounding_mode = RoundingMode.even
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            xq, x_scale = triton_silu_quantize_mx4_unpack(
                x, w, group_size=group_size, rounding_mode=rounding_mode
            )
            intermediate = torch.nn.functional.silu(x.to(torch.float32)) * w.to(
                torch.float32
            )
            intermediate = intermediate.to(torch.bfloat16)
            # Dequantize and check that results are similar.
            x_scale = _from_blocked(x_scale, (M, math.ceil(N / group_size)))
            xq_float = fp4_to_float(xq)
            xq_dequant = scale_mx4(xq_float, x_scale, group_size).to(torch.bfloat16)
            torch.testing.assert_close(xq_dequant, intermediate, atol=1, rtol=1)

        _test_silu_quantize_fp4((1, 128))
        _test_silu_quantize_fp4((3, 512))
        _test_silu_quantize_fp4((128, 1024))
        _test_silu_quantize_fp4((10240, 10240))


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10,
    "Skip when B200 is not available",
)
class TestNVFp4Quantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.accelerator.current_accelerator()

    @unittest.skipIf(open_source, "fp4 quantize is not available")
    def test_silu_quantize_nvfp4(self) -> None:
        def _test_silu_quantize_nvfp4(
            shape: Tuple[int, int],
            device: str = "cuda",
            mimic_mx4_as_nvfp4: bool = False,
        ) -> None:
            M, N = shape
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            if mimic_mx4_as_nvfp4:
                x_global_scale = cal_global_scale_mx4_as_nvfp4(x)
            else:
                x_global_scale = global_scale_nvfp4(x)
            xq, x_scale = triton_quantize_nvfp4(
                x,
                x_global_scale,
                use_e8m0_scale=mimic_mx4_as_nvfp4,
            )

            xq = xq.view(torch.uint8)
            xq_dequant = dequantize_nvfp4(xq, x_scale, x_global_scale)
            torch.testing.assert_close(xq_dequant, x, atol=1, rtol=1)

        _test_silu_quantize_nvfp4((1, 128))
        _test_silu_quantize_nvfp4((4, 512))
        _test_silu_quantize_nvfp4((128, 1024))
        _test_silu_quantize_nvfp4((10240, 10240))
        _test_silu_quantize_nvfp4((1, 128), mimic_mx4_as_nvfp4=True)
        _test_silu_quantize_nvfp4((4, 512), mimic_mx4_as_nvfp4=True)
        _test_silu_quantize_nvfp4((128, 1024), mimic_mx4_as_nvfp4=True)
        _test_silu_quantize_nvfp4((10240, 10240), mimic_mx4_as_nvfp4=True)

    @settings(deadline=None)
    @given(
        B_T=st.sampled_from([2048, 4096]),
        D=st.sampled_from([128, 256]),
        HD_L=st.sampled_from([256, 512]),
        static_scale=sample_scales(),
        scale_ub=sample_scales(),
    )
    def test_fake_quantize_nvfp4_per_tensor(
        self,
        B_T: int,
        D: int,
        HD_L: int,
        static_scale: Optional[torch.Tensor],
        scale_ub: Optional[torch.Tensor],
    ) -> None:
        x = (
            torch.randn(
                size=(B_T, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(HD_L, D),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

        xq, _ = torch.ops.mslk.fake_quantize_nvfp4_per_tensor(
            x, static_scales=static_scale, scale_ub=scale_ub
        )
        wq, _ = torch.ops.mslk.fake_quantize_nvfp4_per_tensor(
            w, static_scales=static_scale, scale_ub=scale_ub
        )
        fake_quant_y = xq @ wq.T
        fake_quant_y = fake_quant_y.to(torch.bfloat16)

        y_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(fake_quant_y, y_ref, atol=0.1, rtol=0.1)

    @settings(deadline=None)
    @given(
        problem_shape=st.sampled_from(
            [
                (128, 64),  # canonical layout
                (5, 64),  # canonical layout with M padding
                (128, 16),  # canonical layout with N padding
                (5, 16),  # canonical layout with M and N padding
                (256, 64),  # two canonical layouts over M
                (128, 128),  # two canonical layouts over N
                (150, 64),  # two canonical layouts over M with padding
                (128, 144),  # two canonical layouts over N with padding
                (4096, 4096),  # large square matrix
                (4000, 4096),  # large matrix with m padding
                (4096, 4080),  # large square matrix with n padding
                (4000, 4080),  # large square matrix with m and n padding
            ]
        ),
    )
    def test_numerical_correctness(self, problem_shape):
        """
        Quantize against torch NVFP4 reference and compare for numerical correctness.
        """
        torch.manual_seed(0)
        M, N = problem_shape
        x = torch.randn(M, N, dtype=torch.bfloat16, device=self.device)

        x_global_scale = global_scale_nvfp4(x)
        x_fp4, x_scales = triton_quantize_nvfp4(x, x_global_scale)
        x_fp4_ref, x_scales_ref, x_global_scale_ref = data_to_nvfp4_with_global_scale(
            x, 16
        )
        # Compare the scales with padding to ensure the padding zero'd out.
        x_scales_ref = _to_blocked(x_scales_ref).view(x_scales.shape)

        torch.testing.assert_close(x_global_scale, x_global_scale_ref.reciprocal())
        torch.testing.assert_close(x_scales, x_scales_ref)
        torch.testing.assert_close(x_fp4, x_fp4_ref)

    @settings(deadline=None)
    @given(
        test_case=st.sampled_from(
            [
                ((1024, 5), "N must be divisible by 16 for NVFP4 quantization"),
            ]
        ),
    )
    def test_invalid_inputs(self, test_case):
        torch.manual_seed(0)
        problem_shape, error_msg = test_case
        M, N = problem_shape
        x = torch.randn(M, N, dtype=torch.bfloat16, device=self.device)

        x_global_scale = global_scale_nvfp4(x)
        with self.assertRaisesRegex(AssertionError, error_msg):
            triton_quantize_nvfp4(x, x_global_scale)


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10,
    "Skip when B200 is not available",
)
class TestNVFp4SiluQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    @unittest.skipIf(open_source, "silu_mul is not available")
    def test_silu_quantize_nvfp4(self) -> None:
        def _test_silu_quantize_nvfp4(
            shape: Tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 16
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            x_global_scale = global_scale_nvfp4(x)
            xq, x_scale = triton_scale_nvfp4_quant_silu(
                x,
                w,
                x_global_scale,
                group_size=group_size,
            )

            intermediate = silu_mul(x.reshape(-1, 16), w.reshape(-1, 16))
            intermediate = intermediate.to(torch.bfloat16).reshape(M, N)

            xq_dequant = dequantize_nvfp4(xq, x_scale, x_global_scale)
            torch.testing.assert_close(xq_dequant, intermediate, atol=1, rtol=1)

        _test_silu_quantize_nvfp4((1, 128))
        _test_silu_quantize_nvfp4((4, 512))
        _test_silu_quantize_nvfp4((128, 1024))
        _test_silu_quantize_nvfp4((10240, 10240))


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 10,
    "Skip when B200 is not available",
)
class TestNVFp4RmsQuantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    @unittest.skipIf(open_source, "rms_norm is not available")
    def test_rms_quantize_nvfp4(self) -> None:
        def _test_rms_quantize_nvfp4(
            shape: Tuple[int, int],
            device: str = "cuda",
        ) -> None:
            M, N = shape
            group_size = 16
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            w = torch.randn(group_size, dtype=torch.bfloat16, device=device)
            x_global_scale = global_scale_nvfp4(x)
            xq, x_scale = triton_scale_nvfp4_quant_rms(
                x,
                w.repeat(M * N // group_size),
                x_global_scale,
                group_size=group_size,
                EPS=1e-5,
            )

            intermediate = rms_norm(x.reshape(-1, 16), w, eps=1e-5)
            intermediate = intermediate.to(torch.bfloat16).reshape(M, N)

            xq_dequant = dequantize_nvfp4(xq, x_scale, x_global_scale)
            torch.testing.assert_close(xq_dequant, intermediate, atol=1, rtol=1)

        _test_rms_quantize_nvfp4((1, 128))
        _test_rms_quantize_nvfp4((4, 512))
        _test_rms_quantize_nvfp4((128, 1024))
        _test_rms_quantize_nvfp4((1024, 10240))
        # Note, large testing tensors may lead to slight numerical differences


# When the below functions is added to Torch core, remove it and use it there instead.
def data_to_nvfp4_with_global_scale(x, block_size):
    # Simple (slow) reference implementation of NVFP4 two-level-scaling
    orig_shape = x.shape
    x = x.reshape(-1, block_size).to(torch.float32)

    # Per-block-amax
    block_max = torch.amax(torch.abs(x), 1) + 1e-12

    # Per-tensor max
    global_max = x.abs().max()
    assert global_max.dtype == torch.float32

    # Constants
    # Global encoding scale for block-scales
    S_enc = (FP8_E4M3_MAX * FP4_E2M1_MAX) / global_max
    S_dec = 1.0 / S_enc

    # Per-block decode-scale
    S_dec_b = block_max / FP4_E2M1_MAX

    # Stored scaled-e4m3 per-block decode scales
    S_dec_b_e4m3 = (S_dec_b * S_enc).to(torch.float8_e4m3fn)

    # Actual per-block encoding scale
    S_enc_b = S_enc / S_dec_b_e4m3.float()

    # scale & reshape input, reshape scales
    x = (S_enc_b.unsqueeze(1) * x).reshape(orig_shape)
    S_dec_b_e4m3 = S_dec_b_e4m3.reshape(orig_shape[0], -1)

    # cast input
    x_fp4 = _float32_to_float4_e2m1fn_x2(x)

    # fp4x2, fp8_e4m3, float respectively
    return x_fp4, S_dec_b_e4m3, S_dec.float()


def _float32_to_float4_e2m1fn_x2(x):
    assert x.dtype == torch.float
    x = _f32_to_floatx_unpacked(x, FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x
