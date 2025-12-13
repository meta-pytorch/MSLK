# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest
from typing import Any, Callable, List

import mslk.quantize  # noqa: F401
import torch
from hypothesis import given, settings, strategies as st
from mslk.quantize.triton.fp8_quantize import (
    quantize_fp8_block,
    quantize_fp8_row,
    quantize_fp8_tensor,
)


def undo_tensorwise_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Undo FP8 tensorwise quantization.
    return x.to(torch.float) * scale


def undo_rowwise_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Undo FP8 rowwise quantization.
    return x.to(torch.float) * scale.unsqueeze(1)


def undo_colwise_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Undo FP8 colwise quantization.
    return x.to(torch.float) * scale.unsqueeze(0)


def undo_blockwise_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
) -> torch.Tensor:
    orig_shape = x.shape
    # View input in blocks:
    # We limit block size to x's shape as a way to skip padding.
    block_m = min(block_m, x.shape[-2])
    block_k = min(block_k, x.shape[-1])
    x = x.to(torch.float).view(-1, block_m, block_k)
    # Apply scaling.
    x = x * scale
    return x.view(orig_shape)


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Operators are only available on CUDA enabled machines",
)
class FP8CorrectnessTest(unittest.TestCase):
    """Check that FP8 ops produce correct results."""

    @unittest.skipIf(
        torch.cuda.get_device_capability("cuda") < (9, 0), "Only support sm90+"
    )
    @given(
        input_shape=st.sampled_from([[32, 32]]),
        quantize_op_pair=st.sampled_from(
            [
                [quantize_fp8_row, undo_rowwise_quant],
                [quantize_fp8_block, undo_blockwise_quant],
                [quantize_fp8_tensor, undo_tensorwise_quant],
                [torch.ops.mslk.quantize_fp8_per_row, undo_rowwise_quant],
                [torch.ops.mslk.quantize_fp8_per_col, undo_colwise_quant],
                [torch.ops.mslk.quantize_fp8_per_tensor, undo_tensorwise_quant],
            ]
        ),
    )
    @settings(deadline=None)
    def test_correctness(
        self,
        quantize_op_pair: List[Callable[..., Any]],
        input_shape: List[int],
    ) -> None:
        quantize_op, dequantize_op = quantize_op_pair
        test_input = torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)
        quant_out = quantize_op(test_input)
        reconstructed_input = dequantize_op(*quant_out).to(torch.bfloat16)
        torch.testing.assert_close(
            test_input, reconstructed_input, atol=2e-1, rtol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
