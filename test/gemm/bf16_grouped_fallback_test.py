# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest

import mslk.gemm  # noqa: F401
import torch
from mslk.testing.device import skipUnlessGfxArch, skipUnlessRocm
from mslk.utils.device import is_gfx950
from parameterized import parameterized


@skipUnlessRocm()
class BF16GroupedFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNone(
            importlib.util.find_spec("flydsl"),
            "fallback-only test target unexpectedly packages FlyDSL",
        )

    def test_falls_back_to_ck_without_flydsl(self) -> None:
        device = torch.accelerator.current_accelerator()
        m, n, k = 16, 512, 512
        x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        w = torch.randn((1, n, k), dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([m], dtype=torch.int64, device=device)
        out = torch.empty(m * n, dtype=torch.bfloat16, device=device)

        actual = torch.ops.mslk.bf16bf16bf16_grouped_stacked(x, w, m_sizes, out=out)
        expected = x.float().matmul(w[0].float().T).to(torch.bfloat16)

        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertEqual(actual.shape, (m, n))
        torch.testing.assert_close(actual, expected, atol=8.0e-3, rtol=8.0e-3)

    def test_rejects_unsupported_k_tail(self) -> None:
        device = torch.accelerator.current_accelerator()
        m, n, k = 16, 512, 520
        x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        w = torch.randn((1, n, k), dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([m], dtype=torch.int64, device=device)

        with self.assertRaisesRegex(RuntimeError, "K must be divisible by 64"):
            torch.ops.mslk.bf16bf16bf16_grouped_stacked(x, w, m_sizes)


@skipUnlessRocm()
@skipUnlessGfxArch("gfx942", "gfx950")
class RowwisePreshuffleFallbackTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("f8f8bf16_rowwise_preshuffle", torch.bfloat16),
            ("f8f8f16_rowwise_preshuffle", torch.float16),
        ]
    )
    def test_falls_back_to_ck_without_flydsl(self, op: str, dtype: torch.dtype) -> None:
        self.assertIsNone(importlib.util.find_spec("flydsl"))
        device = torch.accelerator.current_accelerator()
        fp8_dtype = torch.float8_e4m3fn if is_gfx950() else torch.float8_e4m3fnuz
        xq = torch.ones((16, 1024), dtype=torch.bfloat16, device=device).to(fp8_dtype)
        wq = torch.ones((512, 1024), dtype=torch.bfloat16, device=device).to(fp8_dtype)
        x_scale = torch.full((16,), 0.25, dtype=torch.float32, device=device)
        w_scale = torch.full((512,), 0.125, dtype=torch.float32, device=device)

        actual = getattr(torch.ops.mslk, op)(xq, wq, x_scale, w_scale)

        expected = torch.full((16, 512), 32.0, dtype=dtype, device=device)
        self.assertEqual(actual.dtype, dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
