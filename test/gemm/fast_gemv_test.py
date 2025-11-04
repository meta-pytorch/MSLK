# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import os
import unittest

from typing import Any, Callable, List, Tuple

import mslk.gemm.fast_gemv  # noqa: F401

import torch
import triton  # noqa: F401

if torch.cuda.is_available():
    from mslk.quantize.triton.fp8_quantize import quantize_fp8_row

running_on_github: bool = os.getenv("GITHUB_ENV") is not None


def evaluate_cuda_platform_version(major: int) -> bool:
    if torch.version.cuda:
        return torch.cuda.get_device_capability() >= (major, 0)
    return False


SM90_OR_LATER: bool = evaluate_cuda_platform_version(9)

# pyre-fixme[16]: Module `mslk` has no attribute `open_source`.
open_source: bool = getattr(mslk, "open_source", False)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when GPU is not available")
@unittest.skipIf(not SM90_OR_LATER, "Skip when not SM90+")
class FastGemvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.accelerator.current_accelerator()

    def run_gemv(
        self,
        test_cases: List[Tuple[int, int, int]] | List[Tuple[int, int, int, int]],
        gemv_op: Callable[..., Any],
        atol: float,
        rtol: float,
        quantize_w: bool = False,
        quantize_x: bool = False,
    ) -> None:
        for M, N, K in test_cases:
            x = (
                torch.randn(
                    size=(M, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.1
            )
            w = (
                torch.randn(
                    size=(N, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.01
            )
            if quantize_w and not quantize_x:
                wq, w_scale = torch.ops.mslk.quantize_fp8_per_tensor(w)
                z = gemv_op(x, wq, w_scale)
            elif quantize_w and quantize_x:
                # row-wise scaling
                xq, x_scale = torch.ops.mslk.quantize_fp8_per_row(x)
                wq, w_scale = torch.ops.mslk.quantize_fp8_per_row(w)
                z = gemv_op(xq, wq, x_scale, w_scale)
            else:
                z = gemv_op(x, w)
            z_ref = (x @ w.T).to(torch.bfloat16).to(self.device)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)

    def run_gemv_batched(
        self,
        test_cases: List[Tuple[int, int, int]] | List[Tuple[int, int, int, int]],
        gemv_op: Callable[..., Any],
        atol: float,
        rtol: float,
    ) -> None:
        for B, M, N, K in test_cases:
            x = (
                torch.randn(
                    size=(B, M, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.1
            )
            w = (
                torch.randn(
                    size=(B, N, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.01
            )
            xq, x_scale = quantize_fp8_row(x)
            x_scale = x_scale.view(B, -1)
            assert x_scale.shape == (B, M)
            wq, w_scale = quantize_fp8_row(w)
            w_scale = w_scale.view(B, -1)
            assert w_scale.shape == (B, N)
            z = gemv_op(xq, wq, x_scale, w_scale, is_batched=True)
            z_ref = torch.bmm(x, w.transpose(1, 2)).to(torch.bfloat16).to(self.device)
            torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)

    def test_bf16_gemv(self) -> None:
        test_cases = [
            (1, 128, 256),
            (1, 256, 256),
            (1, 1280, 8192),
            (1, 8192, 1024),
            (1, 7168, 8192),
            (1, 8192, 3584),
            (2, 128, 256),
            (2, 256, 256),
            (2, 1280, 8192),
            (2, 8192, 1024),
            (2, 7168, 8192),
            (2, 8192, 3584),
            (4, 128, 256),
            (4, 256, 256),
            (4, 1280, 8192),
            (4, 8192, 1024),
            (4, 7168, 8192),
            (4, 8192, 3584),
        ]
        self.run_gemv(test_cases, torch.ops.mslk.bf16_fast_gemv, 9.0e-3, 9.0e-3)

    def test_bf16_fp8_gemv(self) -> None:
        test_cases = [
            (1, 1280, 8192),
            (1, 8192, 1024),
            (1, 7168, 8192),
            (1, 8192, 3584),
            (2, 1280, 8192),
            (2, 8192, 1024),
            (2, 7168, 8192),
            (2, 8192, 3584),
            (4, 1280, 8192),
            (4, 8192, 1024),
            (4, 7168, 8192),
            (4, 8192, 3584),
        ]
        self.run_gemv(
            test_cases,
            torch.ops.mslk.bf16fp8bf16_fast_gemv,
            9.0e-2,
            9.0e-2,
            quantize_w=True,
        )

    def test_fp8_fp8_gemv(self) -> None:
        test_cases = [
            (1, 1280, 8192),
            (1, 8192, 1024),
            (1, 7168, 8192),
            (1, 8192, 3584),
            (2, 1280, 8192),
            (2, 8192, 1024),
            (2, 7168, 8192),
            (2, 8192, 3584),
            (3, 1280, 8192),
            (3, 8192, 1024),
            (3, 7168, 8192),
            (3, 8192, 3584),
            (4, 1280, 8192),
            (4, 8192, 1024),
            (4, 7168, 8192),
            (4, 8192, 3584),
            (1, 4096, 5120),  # below are l4_17B_128E dense model shapes
            (1, 5120, 2048),
            (1, 896, 5120),
            (1, 5120, 640),
            (2, 4096, 5120),
            (2, 5120, 2048),
            (2, 896, 5120),
            (2, 5120, 640),
        ]
        self.run_gemv(
            test_cases,
            torch.ops.mslk.fp8fp8bf16_fast_gemv,
            9.0e-2,
            9.0e-2,
            quantize_w=True,
            quantize_x=True,
        )

    def test_fp8_gemv_batched(self) -> None:
        test_cases = [
            (2, 1, 4096, 5120),
            (2, 1, 5120, 2048),
            (2, 1, 896, 5120),
            (2, 1, 5120, 640),
            (2, 1, 8192, 1024),
            (2, 1, 7168, 8192),
            (2, 1, 8192, 3584),
            (2, 1, 1280, 8192),
            (2, 2, 8192, 1024),
            (2, 2, 7168, 8192),
            (2, 2, 8192, 3584),
            (2, 2, 1280, 8192),
            (32, 1, 1280, 8192),
            (128, 1, 1280, 8192),
        ]
        self.run_gemv_batched(
            test_cases,
            torch.ops.mslk.fp8fp8bf16_fast_gemv,
            1.0e-1,
            1.0e-1,
        )


if __name__ == "__main__":
    unittest.main()
