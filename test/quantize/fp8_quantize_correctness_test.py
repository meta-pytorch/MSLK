# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Callable, List, Tuple

import mslk.quantize  # noqa: F401
import torch
from mslk.quantize.triton.fp8_quantize import (
    quantize_fp8_block,
    quantize_fp8_row,
    quantize_fp8_tensor,
)
from parameterized import parameterized


def undo_tensorwise_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float) * scale


def undo_rowwise_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float) * scale.unsqueeze(1)


def undo_colwise_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float) * scale.unsqueeze(0)


def undo_blockwise_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
) -> torch.Tensor:
    M, K = x.shape[-2], x.shape[-1]
    block_m = min(block_m, M)
    block_k = min(block_k, K)

    # Pad to be divisible by block sizes
    pad_m = (block_m - M % block_m) % block_m
    pad_k = (block_k - K % block_k) % block_k

    x = x.to(torch.float)
    if pad_m > 0 or pad_k > 0:
        x = torch.nn.functional.pad(x, (0, pad_k, 0, pad_m))

    M_padded, K_padded = x.shape[-2], x.shape[-1]
    grid_m = M_padded // block_m
    grid_k = K_padded // block_k

    # Reshape into blocks and apply per-block scale
    x = x.reshape(grid_m, block_m, grid_k, block_k)
    x = x * scale[:, None, :, None]
    x = x.reshape(M_padded, K_padded)

    # Remove padding
    return x[:M, :K]


INPUT_SHAPES: List[List[int]] = [
    # Small shape
    [16, 16],
    # Non-square shape
    [32, 64],
    # Medium shape
    [1024, 1024],
    # Large shape
    [8192, 4096],
    # Non-power-of-2 shapes
    [333, 777],
    # Small dimensions
    [1, 128],
    [128, 1],
]

TRITON_QUANTIZE_OPS: List[Tuple[Callable[..., Any], Callable[..., Any]]] = [
    (quantize_fp8_row, undo_rowwise_quant),
    (quantize_fp8_block, undo_blockwise_quant),
    (quantize_fp8_tensor, undo_tensorwise_quant),
]

MSLK_QUANTIZE_OPS: List[Tuple[Callable[..., Any], Callable[..., Any]]] = [
    (torch.ops.mslk.quantize_fp8_per_row, undo_rowwise_quant),
    (torch.ops.mslk.quantize_fp8_per_col, undo_colwise_quant),
    (torch.ops.mslk.quantize_fp8_per_tensor, undo_tensorwise_quant),
]

ALL_QUANTIZE_OPS: List[Tuple[Callable[..., Any], Callable[..., Any]]] = (
    TRITON_QUANTIZE_OPS + MSLK_QUANTIZE_OPS
)


def _generate_test_cases() -> List[
    Tuple[str, List[int], Callable[..., Any], Callable[..., Any]]
]:
    """Generate test cases for parameterized tests."""
    test_cases = []
    for shape in INPUT_SHAPES:
        for quantize_op, dequantize_op in ALL_QUANTIZE_OPS:
            name = f"{quantize_op}_{shape[0]}x{shape[1]}"
            test_cases.append((name, shape, quantize_op, dequantize_op))
    return test_cases


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Operators are only available on CUDA enabled machines",
)
@unittest.skipIf(
    torch.version.cuda is None or torch.cuda.get_device_capability("cuda") < (9, 0),
    "Only support sm90+",
)
class FP8CorrectnessTest(unittest.TestCase):
    """Check that FP8 ops produce correct results."""

    # pyre-ignore[56]: Pyre can't track decorator-modified signatures.
    @parameterized.expand(_generate_test_cases())
    def test_quantize(
        self,
        _name: str,
        input_shape: List[int],
        quantize_op: Callable[..., Any],
        dequantize_op: Callable[..., Any],
    ) -> None:
        test_input = torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)
        quant_out = quantize_op(test_input)
        reconstructed_input = dequantize_op(*quant_out).to(torch.bfloat16)
        torch.testing.assert_close(
            test_input, reconstructed_input, atol=2.5e-1, rtol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
