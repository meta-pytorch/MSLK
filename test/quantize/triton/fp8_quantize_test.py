# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import unittest
from typing import Optional

import torch
import triton

if torch.cuda.is_available():
    from mslk.quantize.triton.fp8_quantize import (
        dequantize_fp8_block,
        dequantize_fp8_packed_row,
        dequantize_fp8_row,
        quantize_fp8_block,
        quantize_fp8_group,
        # packed_row unpacks the values, packed_row_raw returns just the packed tensor
        quantize_fp8_packed_row,
        quantize_fp8_packed_row_raw,
        quantize_fp8_row,
        quantize_fp8_row_meta,
        scale_fp8_row,
    )


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "Skip when H100 is not available",
)
class TestFp8Quantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_quantize_fp8_row(self) -> None:
        def _test_quantize_fp8_row(
            shape: tuple[int, ...],
            use_triton: bool,
            device: torch.device,
            output_device: Optional[torch.device] = None,
            use_jagged: bool = False,
            use_scale_ub: bool = False,
            transpose_inputs: bool = False,
            align_rows_to: Optional[int] = None,
            expected_padded_size: Optional[int] = None,  # only set with align_rows_to
        ) -> None:
            a = torch.randn(shape, dtype=torch.bfloat16, device=device)
            inputs = [a]
            # if transpose_inputs is true, get all possible dimension combinations
            # of the input tensor and transposes each pair
            if transpose_inputs:
                dims = range(a.ndim)
                for dim1, dim2 in itertools.combinations(dims, 2):
                    dims_list = list(dims)
                    dims_list[dim1], dims_list[dim2] = dims_list[dim2], dims_list[dim1]
                    inputs.append(a.clone().permute(dims_list))
            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device=device)
                if use_scale_ub
                else None
            )
            for input_a in inputs:
                # Apply sparsification if specified.
                zero_start_index_M = None
                if use_jagged:
                    # View input as [G, M, K] where G is the number of groups.
                    grouped_input = input_a.view(
                        -1, input_a.shape[-2], input_a.shape[-1]
                    )
                    m_vals = torch.randint(
                        0, grouped_input.shape[1] + 1, (grouped_input.shape[0],)
                    )
                    mask = torch.arange(grouped_input.shape[-2]).expand(
                        (grouped_input.shape[0], grouped_input.shape[1])
                    ) >= m_vals.unsqueeze(-1)
                    # Set corresponding values to 0.
                    grouped_input[mask] = 0.0
                    # Generate nonzero tensor in same layout as input.
                    zero_start_index_M = torch.count_nonzero(
                        torch.sum(grouped_input, dim=-1), dim=-1
                    )

                a_fp8, a_scale = quantize_fp8_row(
                    input_a,
                    scale_ub=scale_ub,
                    zero_start_index_M=zero_start_index_M,
                    use_triton=use_triton,
                    output_device=output_device,
                    align_rows_to=align_rows_to,
                )

                a_fp8_meta, a_scale_meta = quantize_fp8_row_meta(
                    input_a,
                    scale_ub=scale_ub,
                    zero_start_index_M=zero_start_index_M,
                    use_triton=use_triton,
                    output_device=output_device,
                    align_rows_to=align_rows_to,
                )

                self.assertEqual(a_fp8.dtype, a_fp8_meta.dtype)
                self.assertEqual(a_fp8.shape, a_fp8_meta.shape)
                self.assertEqual(a_scale.dtype, a_scale_meta.dtype)
                self.assertEqual(a_scale.shape, a_scale_meta.shape)

                # Undo scaling.
                a_torch = a_fp8.to(torch.bfloat16)
                broadcast_shape = list(a_torch.shape[:-1]) + [-1]

                assert a_scale.shape == a_torch.shape[:-1]

                a_torch *= a_scale.view(broadcast_shape)

                if align_rows_to is not None:
                    # Pad input_a's row dimension to expected_padded_size if specified.
                    assert expected_padded_size is not None
                    pad_rows = expected_padded_size - input_a.shape[-1]
                    if pad_rows > 0:
                        pad_shape = list(input_a.shape)
                        pad_shape[-1] = pad_rows
                        pad_tensor = torch.zeros(
                            pad_shape,
                            dtype=input_a.dtype,
                            device=input_a.device,
                        )
                        input_a = torch.cat([input_a, pad_tensor], dim=-1)

                self.assertTrue(
                    torch.allclose(
                        input_a.to(device=output_device),
                        a_torch,
                        atol=2e-1,
                        rtol=1e-1,
                    )
                )

        for n_col in range(1, 9000, 100):
            _test_quantize_fp8_row((2, n_col), True, torch.device("cuda"))

            # Test with padding. These go up to 9000 (larger than max BLOCK_SIZE)

            # Calculate expected_padded_size from align_rows_to=8.
            # Using a different math here, just to make tests different from implementation.
            align_rows_to = 8
            trailing_beyond_alignment = n_col % align_rows_to
            padding_size = (
                align_rows_to - trailing_beyond_alignment
                if trailing_beyond_alignment > 0
                else 0
            )
            expected_padded_size = n_col + padding_size
            _test_quantize_fp8_row(
                (2, n_col),
                True,
                torch.device("cuda"),
                align_rows_to=align_rows_to,
                expected_padded_size=expected_padded_size,
            )

        # Test with batched input.
        _test_quantize_fp8_row((4, 2, 3), True, torch.device("cuda"))
        _test_quantize_fp8_row(  # simple padding case
            (4, 2, 3),
            True,
            torch.device("cuda"),
            align_rows_to=8,
            expected_padded_size=8,
        )
        _test_quantize_fp8_row(  # multiple padding case
            (4, 2, 13),
            True,
            torch.device("cuda"),
            align_rows_to=8,
            expected_padded_size=16,
        )
        _test_quantize_fp8_row(  # 0 padding case
            (4, 2, 8),
            True,
            torch.device("cuda"),
            align_rows_to=8,
            expected_padded_size=8,
        )
        _test_quantize_fp8_row((6, 4, 2, 3), True, torch.device("cuda"))
        # Test with non-contiguous input
        _test_quantize_fp8_row(
            (4, 2, 3), True, torch.device("cuda"), transpose_inputs=True
        )
        _test_quantize_fp8_row(
            (6, 4, 2, 3), True, torch.device("cuda"), transpose_inputs=True
        )
        _test_quantize_fp8_row((2, 3), True, torch.device("cuda"), use_scale_ub=True)
        # Test with cpu
        _test_quantize_fp8_row((2, 3), False, torch.device("cpu"), torch.device("cuda"))
        _test_quantize_fp8_row(
            (2, 3), False, torch.device("cpu"), torch.device("cuda"), use_scale_ub=True
        )
        _test_quantize_fp8_row((4, 2, 3), True, torch.device("cpu"))
        _test_quantize_fp8_row((6, 4, 2, 3), True, torch.device("cpu"))
        # Test with zero_start_index_M
        _test_quantize_fp8_row((20, 30), True, torch.device("cuda"), use_jagged=True)
        _test_quantize_fp8_row(
            (20, 30),
            True,
            torch.device("cuda"),
            use_jagged=True,
            align_rows_to=16,
            expected_padded_size=32,
        )
        _test_quantize_fp8_row(
            (6, 4, 2, 3), True, torch.device("cuda"), use_jagged=True
        )
        _test_quantize_fp8_row(
            (4, 2, 3),
            True,
            torch.device("cuda"),
            transpose_inputs=True,
            use_jagged=True,
        )

    def test_quantize_fp8_packed_row(self) -> None:
        def _test_quantize_fp8_packed_row(
            shape: tuple[int, ...],
            use_triton: bool,
            device: torch.device,
            output_device: Optional[torch.device] = None,
            use_jagged: bool = False,
            use_scale_ub: bool = False,
            transpose_inputs: bool = False,
        ) -> None:
            a = torch.randn(shape, dtype=torch.bfloat16, device=device)
            inputs = [a]
            # if transpose_inputs is true, get all possible dimension combinations
            # of the input tensor and transposes each pair
            if transpose_inputs:
                dims = range(a.ndim)
                for dim1, dim2 in itertools.combinations(dims, 2):
                    dims_list = list(dims)
                    dims_list[dim1], dims_list[dim2] = dims_list[dim2], dims_list[dim1]
                    inputs.append(a.clone().permute(dims_list))
            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device=device)
                if use_scale_ub
                else None
            )
            for input_a in inputs:
                # Apply sparsification if specified.
                zero_start_index_M = None
                if use_jagged:
                    # View input as [G, M, K] where G is the number of groups.
                    grouped_input = input_a.view(
                        -1, input_a.shape[-2], input_a.shape[-1]
                    )
                    m_vals = torch.randint(
                        0, grouped_input.shape[1] + 1, (grouped_input.shape[0],)
                    )
                    mask = torch.arange(grouped_input.shape[-2]).expand(
                        (grouped_input.shape[0], grouped_input.shape[1])
                    ) >= m_vals.unsqueeze(-1)
                    # Set corresponding values to 0.
                    grouped_input[mask] = 0.0
                    # Generate nonzero tensor in same layout as input.
                    zero_start_index_M = torch.count_nonzero(
                        torch.sum(grouped_input, dim=-1), dim=-1
                    )

                a_fp8, a_scale = quantize_fp8_packed_row(
                    input_a,
                    scale_ub=scale_ub,
                    zero_start_index_M=zero_start_index_M,
                    use_triton=use_triton,
                    output_device=output_device,
                )

                # Undo scaling.
                a_torch = a_fp8.to(torch.bfloat16)
                broadcast_shape = list(a_torch.shape[:-1]) + [-1]

                assert a_scale.shape == a_torch.shape[:-1]

                a_torch *= a_scale.view(broadcast_shape)

                self.assertTrue(
                    torch.allclose(
                        input_a.to(device=output_device),
                        a_torch,
                        atol=2e-1,
                        rtol=1e-1,
                    )
                )

        for n_col in range(1, 9000, 100):
            _test_quantize_fp8_packed_row((2, n_col), True, torch.device("cuda"))
        # Test with batched input.
        _test_quantize_fp8_packed_row((4, 2, 3), True, torch.device("cuda"))
        _test_quantize_fp8_packed_row((6, 4, 2, 3), True, torch.device("cuda"))
        # Test with non-contiguous input
        _test_quantize_fp8_packed_row(
            (4, 2, 3), True, torch.device("cuda"), transpose_inputs=True
        )
        _test_quantize_fp8_packed_row(
            (6, 4, 2, 3), True, torch.device("cuda"), transpose_inputs=True
        )
        _test_quantize_fp8_packed_row(
            (2, 3), True, torch.device("cuda"), use_scale_ub=True
        )
        # Test with cpu
        _test_quantize_fp8_packed_row(
            (2, 3), False, torch.device("cpu"), torch.device("cuda")
        )
        _test_quantize_fp8_packed_row(
            (2, 3), False, torch.device("cpu"), torch.device("cuda"), use_scale_ub=True
        )
        _test_quantize_fp8_packed_row((4, 2, 3), True, torch.device("cpu"))
        _test_quantize_fp8_packed_row((6, 4, 2, 3), True, torch.device("cpu"))
        # Test with zero_start_index_M
        _test_quantize_fp8_packed_row(
            (20, 30), True, torch.device("cuda"), use_jagged=True
        )
        _test_quantize_fp8_packed_row(
            (6, 4, 2, 3), True, torch.device("cuda"), use_jagged=True
        )
        _test_quantize_fp8_packed_row(
            (4, 2, 3),
            True,
            torch.device("cuda"),
            transpose_inputs=True,
            use_jagged=True,
        )

    def test_dequantize_fp8_row(self) -> None:
        def _test_dequantize_fp8_row(
            shape: tuple[int, ...],
        ) -> None:
            a = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
            a_fp8, a_scale = quantize_fp8_row(
                a,
                use_triton=True,
            )

            # Undo scaling.
            a_bf16 = dequantize_fp8_row(a_fp8, a_scale)

            ms = triton.testing.do_bench(
                lambda: dequantize_fp8_row(a_fp8, a_scale),
            )
            print(f"Shape: {a.shape} MS: {ms}")
            torch.testing.assert_close(a_bf16, a, atol=2e-1, rtol=1e-1)
            self.assertTrue(
                torch.allclose(
                    a,
                    a_bf16,
                    atol=2e-1,
                    rtol=1e-1,
                )
            )

        for n_col in [1, 100, 1000]:
            _test_dequantize_fp8_row((2, n_col))
        # Test with batched input.
        _test_dequantize_fp8_row((4, 2, 3))
        shapes = [(4, 2, 3), (6, 4, 2, 3), (2, 3), (20, 30)]
        for shape in shapes:
            _test_dequantize_fp8_row(shape)

    def test_dequantize_fp8_packed_row(self) -> None:
        def _test_dequantize_fp8_packed_row(
            shape: tuple[int, ...],
        ) -> None:
            a = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

            packed_values = quantize_fp8_packed_row_raw(
                a,
                use_triton=True,
            )

            # Undo scaling.
            a_bf16 = dequantize_fp8_packed_row(packed_values)

            ms = triton.testing.do_bench(
                lambda: dequantize_fp8_packed_row(packed_values),
            )
            print(f"Shape: {a.shape} MS: {ms}")

            torch.testing.assert_close(a_bf16, a, atol=2e-1, rtol=1e-1)

            self.assertTrue(
                torch.allclose(
                    a,
                    a_bf16,
                    atol=2e-1,
                    rtol=1e-1,
                )
            )

        for n_col in [1, 100, 1000]:
            _test_dequantize_fp8_packed_row((2, n_col))
        # Test with batched input.
        _test_dequantize_fp8_packed_row((4, 2, 3))
        shapes = [(4, 2, 3), (6, 4, 2, 3), (2, 3), (20, 30)]
        for shape in shapes:
            _test_dequantize_fp8_packed_row(shape)

    def test_scale_fp8_row(self) -> None:
        def _test_scale_fp8_row(
            shape: tuple[int, int],
            device: torch.device,
        ) -> None:
            M, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)

            x_scale = torch.randn(M, dtype=torch.bfloat16, device=device)
            w_scale = torch.randn(K, dtype=torch.bfloat16, device=device)

            scaled_out = scale_fp8_row(a, x_scale, w_scale)

            # Compare with reference value.
            scaled_out_torch = a * x_scale[:, None] * w_scale[None, :]

            self.assertTrue(
                torch.allclose(
                    scaled_out,
                    scaled_out_torch,
                    atol=2e-1,
                    rtol=1e-1,
                )
            )

        _test_scale_fp8_row((2, 3), torch.device("cuda"))
        _test_scale_fp8_row((2, 3), torch.device("cpu"))

    def test_quantize_fp8_group(self) -> None:
        def _test_quantize_fp8_group(
            shape: tuple[int, int],
            group_size: int,
            use_scale_ub: bool = False,
        ) -> None:
            M, K = shape
            a = torch.randn(M, K, dtype=torch.float, device="cuda")

            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device="cuda")
                if use_scale_ub
                else None
            )

            a_fp8, a_scale = quantize_fp8_group(a, group_size, scale_ub=scale_ub)

            a_torch = a_fp8.to(torch.float)

            # Undo scaling.
            a_torch = a_torch.view(-1, K // group_size, group_size) * a_scale.unsqueeze(
                -1
            )
            a_torch = a_torch.view(M, K)

            self.assertTrue(torch.allclose(a, a_torch, atol=2e-1, rtol=5e-2))

        _test_quantize_fp8_group((128, 128), 128)
        _test_quantize_fp8_group((1, 256), 64)
        _test_quantize_fp8_group((2, 384), 128, use_scale_ub=True)

    def test_quantize_fp8_block(self) -> None:
        def _test_quantize_fp8_block(
            shape: tuple[int, int],
            block_shape: tuple[int, int],
            use_scale_ub: bool = False,
        ) -> None:
            M, K = shape
            BLOCK_M, BLOCK_K = block_shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device="cuda")
                if use_scale_ub
                else None
            )

            a_fp8, a_scale = quantize_fp8_block(a, BLOCK_M, BLOCK_K, scale_ub=scale_ub)

            a_torch = a_fp8.to(torch.bfloat16)

            # Undo scaling.
            for i in range(0, M, BLOCK_M):
                for j in range(0, K, BLOCK_K):
                    block = a_torch[i : i + BLOCK_M, j : j + BLOCK_K]
                    scaling = a_scale[i // BLOCK_M, j // BLOCK_K]
                    scaled_block = block * scaling
                    a_torch[i : i + BLOCK_M, j : j + BLOCK_K] = scaled_block

            self.assertTrue(torch.allclose(a, a_torch, atol=2e-1, rtol=5e-2))

        _test_quantize_fp8_block((2, 4), (1, 2))
        _test_quantize_fp8_block((3, 6), (2, 8))
        _test_quantize_fp8_block((3, 6), (2, 8), use_scale_ub=True)

    def test_dequantize_fp8_block(self) -> None:
        def _test_dequantize_fp8_block(
            shape: tuple[int, int],
            block_shape: tuple[int, int],
            use_scale_ub: bool = False,
        ) -> None:
            M, K = shape
            BLOCK_M, BLOCK_K = block_shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

            scale_ub = (
                torch.tensor([1200], dtype=torch.float, device="cuda")
                if use_scale_ub
                else None
            )

            a_fp8, a_scale = quantize_fp8_block(
                a, block_m=BLOCK_M, block_k=BLOCK_K, scale_ub=scale_ub
            )
            a_dequant = dequantize_fp8_block(
                a_fp8, a_scale, block_m=BLOCK_M, block_k=BLOCK_K
            )
            self.assertTrue(torch.allclose(a, a_dequant, atol=2e-1, rtol=5e-2))

        _test_dequantize_fp8_block((3, 1024), (1, 256))
        _test_dequantize_fp8_block((11, 128), (1, 128))
        _test_dequantize_fp8_block((11, 256), (1, 256), use_scale_ub=True)
