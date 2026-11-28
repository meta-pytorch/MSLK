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

    def _inputs_with_empty_group(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.accelerator.current_accelerator()
        x = torch.ones((32, 512), dtype=torch.bfloat16, device=device)
        w = torch.stack(
            [
                torch.full((512, 512), value, dtype=torch.bfloat16, device=device)
                for value in (1, 2, 3)
            ]
        )
        m_sizes = torch.tensor([16, 0, 16], dtype=torch.int64, device=device)
        out = torch.full((32 * 512,), -17, dtype=torch.bfloat16, device=device)
        return x, w, m_sizes, out

    def test_empty_group_and_preallocated_output(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU unavailable")
        x, w, m_sizes, out = self._inputs_with_empty_group()
        actual = torch.ops.mslk.bf16bf16bf16_grouped_stacked(x, w, m_sizes, out=out)
        expected = torch.cat(
            [
                torch.full((16, 512), value, dtype=torch.bfloat16, device=x.device)
                for value in (512, 1536)
            ]
        )
        self.assertEqual(actual.data_ptr(), out.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_graph_replay_reads_updated_group_sizes(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("GPU unavailable")
        x, w, m_sizes, out = self._inputs_with_empty_group()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.ops.mslk.bf16bf16bf16_grouped_stacked(x, w, m_sizes, out=out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = torch.ops.mslk.bf16bf16bf16_grouped_stacked(x, w, m_sizes, out=out)
        m_sizes.copy_(
            torch.tensor([0, 16, 16], dtype=torch.int64, device=m_sizes.device)
        )
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.cat(
            [
                torch.full((16, 512), value, dtype=torch.bfloat16, device=x.device)
                for value in (1024, 1536)
            ]
        )
        self.assertEqual(actual.data_ptr(), out.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
