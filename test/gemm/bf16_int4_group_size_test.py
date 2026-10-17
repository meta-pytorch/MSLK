# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import mslk.gemm  # noqa: F401
import torch
from mslk.testing.device import skipUnlessCuda, skipUnlessCudaCapability
from parameterized import parameterized


@skipUnlessCuda()
@skipUnlessCudaCapability(9, 9)
class BF16Int4GroupSizeTests(unittest.TestCase):
    @parameterized.expand(
        [
            (group_size, shuffled, scale_dtype)
            for group_size in (32, 96)
            for shuffled in (False, True)
            for scale_dtype in (torch.bfloat16, torch.float32)
        ]
    )
    def test_rejects_groups_that_split_k_tiles(
        self, group_size: int, shuffled: bool, scale_dtype: torch.dtype
    ) -> None:
        K = 4 * group_size
        x = torch.ones((40, K), dtype=torch.bfloat16, device="cuda")
        # Every packed weight is 1, so shuffling leaves the data unchanged.
        wq = torch.full((64, K // 2), 0x11, dtype=torch.int8, device="cuda")
        scale = (
            torch.arange(1, 5, device="cuda", dtype=scale_dtype)
            .unsqueeze(1)
            .repeat(1, 64)
        )
        zero = torch.zeros_like(scale)
        op = (
            torch.ops.mslk.bf16i4bf16_shuffled
            if shuffled
            else torch.ops.mslk.bf16i4bf16_rowwise
        )

        with self.assertRaisesRegex(RuntimeError, "group size"):
            op(x, wq, scale, zero)

    @parameterized.expand(
        [
            (K, group_size, shuffled, scale_dtype)
            for K, group_size, shuffled in (
                (128, 64, False),
                (128, 128, False),
                (32, 32, False),
                (96, 96, False),
                (128, 64, True),
                (128, 128, True),
            )
            for scale_dtype in (torch.bfloat16, torch.float32)
        ]
    )
    def test_supported_groups_use_each_scale_and_zero(
        self, K: int, group_size: int, shuffled: bool, scale_dtype: torch.dtype
    ) -> None:
        x = torch.ones((40, K), dtype=torch.bfloat16, device="cuda")
        wq = torch.full((64, K // 2), 0x11, dtype=torch.int8, device="cuda")
        scale = (
            torch.arange(1, K // group_size + 1, device="cuda", dtype=scale_dtype)
            .unsqueeze(1)
            .repeat(1, 64)
        )
        zero = torch.full_like(scale, 0.25)
        op = (
            torch.ops.mslk.bf16i4bf16_shuffled
            if shuffled
            else torch.ops.mslk.bf16i4bf16_rowwise
        )

        actual = op(x, wq, scale, zero)
        expected = ((scale + zero).sum(dim=0) * group_size).to(torch.bfloat16)

        torch.testing.assert_close(actual, expected.expand(40, -1), rtol=0, atol=0)
