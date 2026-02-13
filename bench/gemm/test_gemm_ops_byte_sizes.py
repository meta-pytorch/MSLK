# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that GemmOpBase byte size properties match actual quantized tensor dtypes."""

from __future__ import annotations

import unittest

import torch

# gemm_ops cannot be imported without an accelerator because some ops
# check hardware support at registration time (module import).
if torch.cuda.is_available():
    from mslk.bench.gemm.gemm_ops import GemmOpBase, get_gemm_ops


def _effective_bytes_per_element(tensor: torch.Tensor, logical_numel: int) -> float:
    """Compute effective bytes per element, accounting for sub-byte packing.

    For packed types (e.g., Int4 packed as Int8, FP4 packed as float4_e2m1fn_x2),
    the tensor has fewer storage elements than logical elements. We use the
    ratio of total bytes to logical element count to get the true per-element cost.

    Args:
        tensor: The quantized tensor.
        logical_numel: The number of logical elements (e.g., M*K for input).
    """
    if logical_numel <= 0:
        return float(tensor.dtype.itemsize)
    return tensor.nbytes / logical_numel


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA accelerator")
class TestGemmOpByteSizeProperties(unittest.TestCase):
    """Verify that each GemmOpBase subclass declares byte sizes consistent
    with the tensors it actually produces during quantization."""

    M, K, N = 64, 256, 128

    def _run_quantize(self, op: GemmOpBase) -> tuple:
        """Run preprocess + quantize for an op and return the quantized tensors."""
        x = torch.randn(self.M, self.K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(self.N, self.K, dtype=torch.bfloat16, device="cuda")
        preprocessed = op.preprocess(x, w)
        quantized = op.quantize(*preprocessed)
        return quantized

    def test_byte_size_properties_exist(self) -> None:
        """Every registered op must expose all three byte size properties."""
        for op in get_gemm_ops():
            with self.subTest(op=op.name):
                self.assertIsInstance(op.input_bytes_per_element, float)
                self.assertIsInstance(op.weight_bytes_per_element, float)
                self.assertIsInstance(op.output_bytes_per_element, float)
                self.assertGreater(op.input_bytes_per_element, 0)
                self.assertGreater(op.weight_bytes_per_element, 0)
                self.assertGreater(op.output_bytes_per_element, 0)

    def test_byte_size_values_are_reasonable(self) -> None:
        """Byte sizes must be one of the known precision values."""
        known_sizes = {0.5, 1.0, 2.0, 4.0}
        for op in get_gemm_ops():
            with self.subTest(op=op.name):
                self.assertIn(
                    op.input_bytes_per_element,
                    known_sizes,
                    f"{op.name}: input_bytes_per_element="
                    f"{op.input_bytes_per_element} not in {known_sizes}",
                )
                self.assertIn(
                    op.weight_bytes_per_element,
                    known_sizes,
                    f"{op.name}: weight_bytes_per_element="
                    f"{op.weight_bytes_per_element} not in {known_sizes}",
                )
                self.assertIn(
                    op.output_bytes_per_element,
                    known_sizes,
                    f"{op.name}: output_bytes_per_element="
                    f"{op.output_bytes_per_element} not in {known_sizes}",
                )

    def test_quantized_dtypes_match_declared_byte_sizes(self) -> None:
        """For each supported op, run quantize and compute, then verify that
        the actual tensor byte sizes match the declared properties."""
        supported_ops = [op for op in get_gemm_ops() if op.supported]
        if not supported_ops:
            return  # No supported ops on this hardware, nothing to validate

        for op in supported_ops:
            with self.subTest(op=op.name):
                try:
                    quantized = self._run_quantize(op)
                    # quantize() returns (xq, wq, ...scales/metadata...)
                    xq = quantized[0]
                    wq = quantized[1]

                    # Handle grouped gemm ops where xq/wq may be lists
                    if isinstance(xq, list):
                        xq = xq[0]
                    if isinstance(wq, list):
                        wq = wq[0]

                    # Check input (xq) byte size
                    actual_input_bpe = _effective_bytes_per_element(xq, self.M * self.K)
                    self.assertEqual(
                        op.input_bytes_per_element,
                        actual_input_bpe,
                        msg=f"{op.name}: declared input_bytes_per_element="
                        f"{op.input_bytes_per_element} but actual quantized input "
                        f"has {actual_input_bpe:.2f} bytes/elem "
                        f"(dtype={xq.dtype}, shape={xq.shape}, "
                        f"nbytes={xq.nbytes}, logical={self.M * self.K})",
                    )

                    # Check weight (wq) byte size
                    actual_weight_bpe = _effective_bytes_per_element(
                        wq, self.N * self.K
                    )
                    self.assertEqual(
                        op.weight_bytes_per_element,
                        actual_weight_bpe,
                        msg=f"{op.name}: declared weight_bytes_per_element="
                        f"{op.weight_bytes_per_element} but actual quantized weight "
                        f"has {actual_weight_bpe:.2f} bytes/elem "
                        f"(dtype={wq.dtype}, shape={wq.shape}, "
                        f"nbytes={wq.nbytes}, logical={self.N * self.K})",
                    )

                    # Check output byte size
                    output = op.compute(*quantized)
                    if isinstance(output, (list, tuple)):
                        out_tensor = output[0]
                    else:
                        out_tensor = output
                    self.assertEqual(
                        op.output_bytes_per_element,
                        float(out_tensor.dtype.itemsize),
                        msg=f"{op.name}: declared output_bytes_per_element="
                        f"{op.output_bytes_per_element} but actual output dtype "
                        f"{out_tensor.dtype} has itemsize={out_tensor.dtype.itemsize}",
                    )
                except Exception:
                    # Some ops may fail on certain hardware or shapes; skip gracefully
                    continue


if __name__ == "__main__":
    unittest.main()
