# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import unittest
from typing import Optional, Tuple

import mslk
import torch
from hypothesis import given, settings, strategies as st

from mslk.quantize.triton.fp4_quantize import (
    _from_blocked,
    cal_global_scale_mx4_as_nvfp4,
    RoundingMode,
    triton_quantize_mx4_unpack,
    triton_rms_quantize_mx4_unpack,
    triton_scale_nvfp4_quant,
    triton_scale_nvfp4_quant_rms,
    triton_scale_nvfp4_quant_silu,
    triton_silu_quantize_mx4_unpack,
)

# pyre-fixme [16]
open_source: bool = getattr(mslk, "open_source", False)

if not open_source:
    from gen_ai.llm_inference.fb.llm.kernel.rms_norm import rms_norm
    from gen_ai.llm_inference.fb.llm.kernel.silu_mul import silu_mul


def fp4_to_float(x: torch.Tensor) -> torch.Tensor:
    # Start by unpacking the FP4 values into separate integers.
    low_mx4 = torch.bitwise_and(x, 0xF)
    high_mx4 = torch.bitwise_and(x >> 4, 0xF)
    comb_shape = x.shape[:-1] + (x.shape[-1] * 2,)
    x_comb = (
        torch.stack([low_mx4, high_mx4], dim=0)
        .view(2, -1)
        .t()
        .contiguous()
        .to(torch.int32)
    )
    # Map to float with a lookup table.
    E2M1_LUT = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=x.device,
    )
    return torch.index_select(E2M1_LUT, 0, x_comb.view(-1)).view(comb_shape)


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


def scale_nvfp4(
    x: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    # NVFP4 uses a trick where global scales are folded into the scales
    # but not x itself. Those scales are normally removed in the epilogue.
    # Here, we manually get the scaling for x by removing global components.
    true_scale = scale.view(torch.float8_e4m3fn).to(torch.float) / global_scale
    # Now we can reverse scaling of x.
    num_groups = x.shape[-1] // group_size
    scaled_x = (
        x.view(-1, num_groups, group_size)
        * true_scale.view(x.shape[0], -1)[:, :num_groups, None]
    )
    return scaled_x.view(x.shape)


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
            group_size = 16
            x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            if mimic_mx4_as_nvfp4:
                x_global_scale = cal_global_scale_mx4_as_nvfp4(x)
            else:
                x_global_scale = torch.tensor([448.0 * 6.0]).to(
                    device=x.device
                ) / torch.amax(x.flatten(), dim=-1)
            xq, x_scale = triton_scale_nvfp4_quant(
                x,
                x_global_scale,
                group_size=group_size,
                use_e8m0_scale=mimic_mx4_as_nvfp4,
            )
            # Convert blocked x_scale format back to (M, num_groups) layout.
            x_scale = _from_blocked(x_scale, (M, math.ceil(N / group_size)))
            # Dequantize and check that results are similar.
            xq_float = fp4_to_float(xq)
            xq_dequant = scale_nvfp4(xq_float, x_scale, x_global_scale, group_size).to(
                torch.bfloat16
            )
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
            x_global_scale = torch.tensor([448.0 * 6.0]).to(
                device=x.device
            ) / torch.amax(x.flatten(), dim=-1)
            xq, x_scale = triton_scale_nvfp4_quant_silu(
                x,
                w,
                x_global_scale,
                group_size=group_size,
            )

            intermediate = silu_mul(x.reshape(-1, 16), w.reshape(-1, 16))
            intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
            # Convert blocked x_scale format back to (M, num_groups) layout.
            x_scale = _from_blocked(x_scale, (M, math.ceil(N / group_size)))
            # Dequantize and check that results are similar.
            xq_float = fp4_to_float(xq)
            xq_dequant = scale_nvfp4(xq_float, x_scale, x_global_scale, group_size).to(
                torch.bfloat16
            )
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
            x_global_scale = torch.tensor([448.0 * 6.0]).to(
                device=x.device
            ) / torch.amax(x.flatten(), dim=-1)
            xq, x_scale = triton_scale_nvfp4_quant_rms(
                x,
                w.repeat(M * N // group_size),
                x_global_scale,
                group_size=group_size,
                EPS=1e-5,
            )

            intermediate = rms_norm(x.reshape(-1, 16), w, eps=1e-5)
            intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
            # Convert blocked x_scale format back to (M, num_groups) layout.
            x_scale = _from_blocked(x_scale, (M, math.ceil(N / group_size)))
            # Dequantize and check that results are similar.
            xq_float = fp4_to_float(xq)
            xq_dequant = scale_nvfp4(xq_float, x_scale, x_global_scale, group_size).to(
                torch.bfloat16
            )
            torch.testing.assert_close(xq_dequant, intermediate, atol=1, rtol=1)

        _test_rms_quantize_nvfp4((1, 128))
        _test_rms_quantize_nvfp4((4, 512))
        _test_rms_quantize_nvfp4((128, 1024))
        _test_rms_quantize_nvfp4((1024, 10240))
        # Note, large testing tensors may lead to slight numerical differences
