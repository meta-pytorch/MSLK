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


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Operators are only available on CUDA enabled machines",
)
class Fp8QuantizeExportTest(unittest.TestCase):
    """Check that FP8 ops can be compiled and exported properly."""

    def compile_quantize_op(self, op: Callable[..., Any], input: torch.Tensor) -> None:
        class WrapperModule(torch.nn.Module):
            def forward(self, *args) -> Any:
                return op(*args)

        # Create a module wrapper around input operator.
        torch_mod = WrapperModule()
        eager_out = torch_mod(input)
        # Compile and check vs eager output.
        compiled_mod = torch.compile(torch_mod)
        compiled_out = compiled_mod(input)
        # Also check that the operators can be exported.
        exported_mod = torch.export.export(torch_mod, args=(input,))
        exported_out = exported_mod.module()(input)
        for e_o, c_o, x_o in zip(eager_out, compiled_out, exported_out):
            self.assertTrue(torch.allclose(e_o.to(torch.float), c_o.to(torch.float)))
            self.assertTrue(torch.allclose(e_o.to(torch.float), x_o.to(torch.float)))

    @unittest.skipIf(
        torch.cuda.get_device_capability("cuda") < (9, 0), "Only support sm90+"
    )
    @given(
        input_shape=st.sampled_from([[32, 32]]),
        quantize_op=st.sampled_from(
            [
                quantize_fp8_row,
                quantize_fp8_block,
                quantize_fp8_tensor,
                torch.ops.mslk.quantize_fp8_per_row,
                torch.ops.mslk.quantize_fp8_per_col,
                torch.ops.mslk.quantize_fp8_per_tensor,
            ]
        ),
    )
    @settings(deadline=None)
    def test_compile_and_export(
        self, quantize_op: Callable[..., Any], input_shape: List[int]
    ) -> None:
        test_input = torch.randn(input_shape, device="cuda", dtype=torch.bfloat16)
        self.compile_quantize_op(quantize_op, test_input)


if __name__ == "__main__":
    unittest.main()
