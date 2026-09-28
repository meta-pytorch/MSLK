# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest

import mslk.gemm  # noqa: F401
import torch
from mslk.testing.device import skipUnlessRocm


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
