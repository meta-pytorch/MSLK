# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Optional

import torch

if torch.cuda.is_available():
    from mslk.gemm.triton.fp8_gemm import matmul_fp8_block, matmul_fp8_row
    from mslk.quantize.triton.fp8_quantize import quantize_fp8_block, quantize_fp8_row


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp8Matmul(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_matmul_fp8_row(self) -> None:
        def _test_matmul_fp8_row(
            shape: tuple[int, int, int],
            device: torch.device,
            fp8_fast_accum: bool,
            use_bias: bool = False,
            transpose_input: bool = False,
            compile: bool = False,
        ) -> None:
            M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            # Make a non-contiguous tensor and check that we still get proper results.
            if transpose_input:
                a = a.t()
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
            bias = (
                torch.randn(N, dtype=torch.float32, device=device) if use_bias else None
            )

            # Test that we can compile the full fp8 matmul operation.
            if compile:

                @torch.compile(fullgraph=True)
                def _quantize_matmul_fp8(
                    a: torch.Tensor,
                    b: torch.Tensor,
                    bias: Optional[torch.Tensor],
                    fp8_fast_accum: bool,
                ) -> torch.Tensor:
                    a_fp8, a_scale = quantize_fp8_row(a)
                    b_fp8, b_scale = quantize_fp8_row(b)
                    return matmul_fp8_row(
                        a_fp8,
                        b_fp8,
                        a_scale,
                        b_scale,
                        bias=bias,
                        fp8_fast_accum=fp8_fast_accum,
                    )

                result = _quantize_matmul_fp8(a, b, bias, fp8_fast_accum)
            # Otherwise run normally.
            else:
                # Quantize inputs.
                a_fp8, a_scale = quantize_fp8_row(a)
                b_fp8, b_scale = quantize_fp8_row(b)

                result = matmul_fp8_row(
                    a_fp8,
                    b_fp8,
                    a_scale,
                    b_scale,
                    bias=bias,
                    fp8_fast_accum=fp8_fast_accum,
                )
            self.assertTrue(result.shape == (M, N))

            expected_result = a @ b.T
            if use_bias:
                # pyre-fixme[6]: For 1st argument expected `Union[bool, complex,
                #  float, int, Tensor]` but got `Optional[Tensor]`.
                expected_result += bias
            self.assertTrue(
                torch.allclose(result, expected_result, atol=2e-1, rtol=5e-2)
            )

        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True, compile=True)
        _test_matmul_fp8_row(
            (5, 4, 5), torch.device("cuda"), True, transpose_input=True
        )
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), True, True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), False)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cuda"), False, True)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cpu"), False)
        _test_matmul_fp8_row((3, 4, 5), torch.device("cpu"), False, True)

    def test_matmul_fp8_row_skip_scaling(self) -> None:
        def _fp8_clamp(x: torch.Tensor) -> torch.Tensor:
            fp8_dtype = torch.float8_e4m3fn
            fp8_max = torch.finfo(fp8_dtype).max
            xq = torch.clamp(x, min=-1 * fp8_max, max=fp8_max).to(fp8_dtype)
            return xq

        def _test_matmul_fp8_row_skip_scaling(
            shape: tuple[int, int, int],
            device: torch.device,
            use_bias: bool = True,
            transpose_input: bool = False,
            compile: bool = False,
        ) -> None:
            M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            # Make a non-contiguous tensor and check that we still get proper results.
            if transpose_input:
                a = a.t()
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
            bias = (
                torch.randn(N, dtype=torch.float32, device=device) if use_bias else None
            )

            # Test that we can compile the full fp8 matmul operation.
            if compile:

                @torch.compile(fullgraph=True)
                def _quantize_matmul_fp8(
                    a: torch.Tensor,
                    b: torch.Tensor,
                    bias: Optional[torch.Tensor],
                ) -> torch.Tensor:
                    a_fp8, a_scale = _fp8_clamp(a), None
                    b_fp8, b_scale = quantize_fp8_row(b)
                    return matmul_fp8_row(
                        a_fp8,
                        b_fp8,
                        a_scale,
                        b_scale,
                        bias=bias,
                        fp8_fast_accum=True,
                        imprecise_acc=False,
                        tma_persistent=False,
                        no_use_persistent=False,
                        use_warp_specialization=False,
                    )

                result = _quantize_matmul_fp8(a, b, bias)
            # Otherwise run normally.
            else:
                # Quantize inputs.
                a_fp8, a_scale = _fp8_clamp(a), None
                b_fp8, b_scale = quantize_fp8_row(b)

                result = matmul_fp8_row(
                    a_fp8,
                    b_fp8,
                    a_scale,
                    b_scale,
                    bias=bias,
                    fp8_fast_accum=True,
                    imprecise_acc=False,
                    tma_persistent=False,
                    no_use_persistent=False,
                    use_warp_specialization=False,
                )
            self.assertTrue(result.shape == (M, N))

            expected_result = a @ b.T
            if use_bias:
                # pyre-fixme[6]: For 1st argument expected `Union[bool, complex,
                #  float, int, Tensor]` but got `Optional[Tensor]`.
                expected_result += bias
            self.assertTrue(
                torch.allclose(result, expected_result, atol=2e-1, rtol=5e-2)
            )

        _test_matmul_fp8_row_skip_scaling((3, 4, 5), torch.device("cuda"))
        _test_matmul_fp8_row_skip_scaling((3, 4, 5), torch.device("cuda"), compile=True)
        _test_matmul_fp8_row_skip_scaling(
            (5, 4, 5), torch.device("cuda"), transpose_input=True
        )
        _test_matmul_fp8_row_skip_scaling(
            (3, 4, 5), torch.device("cuda"), use_bias=False
        )

    def test_matmul_fp8_block(self) -> None:
        def _test_matmul_fp8_block(
            shape: tuple[int, int, int],
            block_shape: tuple[int, int, int],
            fp8_fast_accum: bool,
            transpose_input: bool = False,
            device: str = "cuda",
        ) -> None:
            M, N, K = shape
            BLOCK_M, BLOCK_N, BLOCK_K = block_shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            # Make a non-contiguous tensor and check that we still get proper results.
            if transpose_input:
                a = a.t()
            b = torch.randn(N, K, dtype=torch.bfloat16, device=device)

            # Quantize inputs.
            a_fp8, a_scale = quantize_fp8_block(
                a, BLOCK_M, BLOCK_K, output_device=torch.device("cuda")
            )
            b_fp8, b_scale = quantize_fp8_block(
                b, BLOCK_N, BLOCK_K, output_device=torch.device("cuda")
            )

            result = matmul_fp8_block(
                a_fp8,
                b_fp8,
                a_scale,
                b_scale,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                fp8_fast_accum=fp8_fast_accum,
            )
            self.assertTrue(result.shape == (M, N))

            expected_result = (a @ b.T).to("cuda")

            self.assertTrue(
                torch.allclose(result, expected_result, atol=1e2, rtol=5e-2)
            )

        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), True)
        _test_matmul_fp8_block((5, 4, 5), (256, 256, 256), True, transpose_input=True)
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), True)
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), False)
        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), False)
        _test_matmul_fp8_block((3, 4, 5), (256, 256, 256), True, device="cpu")
        _test_matmul_fp8_block((1024, 2048, 4096), (256, 512, 1024), True, device="cpu")
