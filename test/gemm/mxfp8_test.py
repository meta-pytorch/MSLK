# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""MXFP8 grouped GEMM correctness tests."""

import random
import unittest

import mslk.gemm  # noqa: F401
import torch
from mslk.gemm.triton.fp8_gemm import to_mxfp8
from mslk.quantize.triton.fp4_quantize import _to_blocked
from mslk.testing.device import skipUnlessCudaCapability, skipUnlessGfxArch
from parameterized import parameterized


# Source: pytorch/ao, prototype/moe_training/utils.py (568c1932a16a), line 310.
def generate_jagged_offs(E, M, multiple_of=16, dtype=torch.int32, device="cuda"):
    """
    Utility function for tests and benchmarks.

    Generates a tensor of length E, containing random values divisible by `multiple_of`,
    from 0 to M, in sorted order, and where the final value in the tensor is always M.
    Args:
        E (int): The length of the tensor.
        M (int): The maximum value in the tensor.
    Returns:
        torch.Tensor: A tensor of length E with the specified properties.
    """
    # Ensure M is divisible by 16
    if M % multiple_of != 0:
        raise ValueError(f"M must be divisible by {multiple_of}")

    # Generate a list of possible values
    possible_values = list(range(multiple_of, M + 1, multiple_of))

    # If E is larger than the number of possible values, raise an error
    if E > len(possible_values):
        raise ValueError("E cannot be larger than the number of possible values")

    # Randomly select E - 1 values from the possible values (excluding M)
    selected_values = random.sample(possible_values[:-1], E - 1)

    # Append M to the selected values
    selected_values.append(M)

    # Sort the selected values
    selected_values.sort()

    # Citrine C3: construct the sampled metadata directly on its target device.
    return torch.tensor(selected_values, dtype=dtype, device=device)


@skipUnlessGfxArch("gfx950")
@skipUnlessCudaCapability(10)
class MXFP8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (1, 2048, 256, 256),  # small
            (4, 2048, 1024, 512),  # medium
            (16, 3584, 6144, 3584),  # large
            (64, 3584, 6144, 3584),  # large G
        ]
    )
    def test_grouped_gemm_2d_2d(
        self,
        G: int,
        K: int,
        N: int,
        M: int,
    ) -> None:
        # Backward weight gradient: grad_output_t @ input, grouped along K.

        total_K = K
        input_group_end_offsets = generate_jagged_offs(
            G, total_K, multiple_of=32, device=self.device
        )
        X = torch.randn((M, total_K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((N, total_K), dtype=torch.bfloat16, device=self.device) * 0.01

        # Convert scales to blocked format.
        x_list = []
        w_list = []
        x_blocked_scale_list = []
        w_blocked_scale_list = []

        def round_up(x: int, y: int) -> int:
            return ((x + y - 1) // y) * y

        for group_idx in range(G):
            # to_mxfp8 per group
            prev_group_end_offset = (
                0 if group_idx == 0 else input_group_end_offsets[group_idx - 1]
            )
            curr_group_end_offset = input_group_end_offsets[group_idx]
            group_size = curr_group_end_offset - prev_group_end_offset
            if group_size > 0:
                x_slice = X[
                    :, prev_group_end_offset:curr_group_end_offset
                ].contiguous()  # (M, K_group)
                w_slice = W[
                    :, prev_group_end_offset:curr_group_end_offset
                ].contiguous()  # (N, K_group)
                x_scale_slice, xq_slice = to_mxfp8(
                    x_slice
                )  # scale shape -> (M, K_group // 32)
                w_scale_slice, wq_slice = to_mxfp8(
                    w_slice
                )  # scale shape -> (N, K_group // 32)
                x_list.append(xq_slice)
                w_list.append(wq_slice)

                # Convert scales to blocked format.
                x_scale_slice_blocked = _to_blocked(
                    x_scale_slice
                )  # (round_up(M, 128), round_up(K_group//32, 4))
                w_scale_slice_blocked = _to_blocked(
                    w_scale_slice
                )  # (round_up(N, 128), round_up(K_group//32, 4))
                x_blocked_scale_list.append(x_scale_slice_blocked)
                w_blocked_scale_list.append(w_scale_slice_blocked)

        # Assemble the full XQ and WQ
        xq = torch.cat(x_list, dim=1).contiguous()
        wq = torch.cat(w_list, dim=1).contiguous()

        # Combine all XQ groups blocked scales into one tensor.
        x_blocked_scales = torch.cat(x_blocked_scale_list, dim=0)
        M_rounded = round_up(M, 128)
        x_blocked_scales = x_blocked_scales.reshape(M_rounded, -1)

        # Combine all WQ groups blocked scales into one tensor.
        w_blocked_scales = torch.cat(w_blocked_scale_list, dim=0)
        N_rounded = round_up(N, 128)
        w_blocked_scales = w_blocked_scales.reshape(N_rounded, -1)

        # Compute mxfp8 grouped mm output
        out = torch.empty((G, M, N), dtype=torch.bfloat16, device=self.device)
        y_mxfp8 = torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq,  # (M, total_K)
            wq.transpose(-2, -1),  # (total_K, N)
            x_blocked_scales,  # to_blocked_per_group(M, total_K//32)
            w_blocked_scales,  # to_blocked_per_group(N, total_K//32)
            input_group_end_offsets,  # (G,)
            out,  # (G, M, N)
        )

        # bf16 reference output
        y_bf16 = torch._grouped_mm(
            X, W.t(), offs=input_group_end_offsets, out_dtype=torch.bfloat16
        )

        # Assert no NaNs
        assert not y_mxfp8.isnan().any(), "mxfp8 output contains NaN"

        # Assert outputs are close
        torch.testing.assert_close(y_mxfp8, y_bf16, atol=8.0e-2, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 256, 256, 256),  # small
            (4, 2048, 1024, 512),  # medium
            (16, 3584, 6144, 3584),  # large
            (64, 3584, 6144, 3584),  # large G
        ]
    )
    def test_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        # Simulate 2d-3d grouped gemm `out = input @ weight.t()`
        # 2D inputs with groups along M, 3D weights.
        block_size = 32
        total_M = M  # Alias for clarity that M dim contains groups.
        X = torch.randn((total_M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((G, N, K), dtype=torch.bfloat16, device=self.device) * 0.01
        input_group_end_offsets = generate_jagged_offs(
            G, total_M, multiple_of=32, device=self.device
        )

        # Quantize and block scales separately for each weight group.
        wq_list = []
        w_scale_list = []
        for i in range(G):
            w_scale, wq = to_mxfp8(W[i])
            w_scale = _to_blocked(w_scale)
            wq_list.append(wq)
            w_scale_list.append(w_scale)
        wq = torch.stack(wq_list, dim=0).contiguous()
        w_scale = torch.stack(w_scale_list, dim=0).contiguous()

        # Quantize and block scales separately for each input group.
        xq_list = []
        x_scale_list = []
        for i in range(G):
            prev_group_end = 0 if i == 0 else input_group_end_offsets[i - 1]
            curr_group_end = input_group_end_offsets[i]
            group_size = curr_group_end - prev_group_end
            if group_size > 0:
                x_slice = X[prev_group_end:curr_group_end, :]
                x_scale, xq = to_mxfp8(x_slice)
                x_scale = _to_blocked(x_scale)
                xq_list.append(xq)
                x_scale_list.append(x_scale)
        xq = torch.cat(xq_list, dim=0).contiguous()
        x_scale = torch.cat(x_scale_list, dim=0).contiguous()
        x_scale = x_scale.reshape(-1, K // block_size)
        xq = xq.view(-1, xq.shape[-1])

        # Compute mxfp8 grouped gemm.
        out = torch.empty((total_M, N), dtype=torch.bfloat16, device=self.device)
        y_mxfp8 = torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq,
            wq.transpose(-2, -1),
            x_scale,
            w_scale,
            input_group_end_offsets,
            out,
        )

        # Compute reference bf16 grouped gemm.
        y_bf16 = torch._grouped_mm(
            X,
            W.transpose(-2, -1),
            offs=input_group_end_offsets,
            out_dtype=torch.bfloat16,
        )

        # Assert outputs are close.
        torch.testing.assert_close(y_mxfp8, y_bf16, atol=8.0e-2, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 16, 2, 1024, 512),  # small G
            (4, 16, 2, 1024, 512),  # small
            (16, 32, 4, 1024, 512),  # medium
            (32, 64, 8, 2048, 3072),  # large
            (64, 64, 8, 2048, 3072),  # large G
        ]
    )
    def test_grouped_gemm_2d_3d_actual_num_tokens(
        self,
        G: int,
        M_per_group: int,
        pad_factor: int,
        N: int,
        K: int,
    ) -> None:
        """Test that actual_num_tokens hint produces correct results.

        XQ is padded to pad_factor * actual_total_M, but offsets only cover the
        actual tokens. Verifies that:
        1. Output with hint matches output without hint (same GEMM, different tile)
        2. Output matches bf16 reference
        """

        block_size = 32
        actual_total_M = M_per_group * G
        padded_total_M = actual_total_M * pad_factor

        # Create actual data
        X = (
            torch.randn((actual_total_M, K), dtype=torch.bfloat16, device=self.device)
            * 0.1
        )
        W = torch.randn((G, N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        # Uniform offsets for actual tokens
        offsets = torch.arange(
            M_per_group,
            actual_total_M + 1,
            M_per_group,
            dtype=torch.int32,
            device=self.device,
        )

        # Quantize weights
        wq_list = []
        w_scale_list = []
        for i in range(G):
            w_scale, wq = to_mxfp8(W[i])
            w_scale = _to_blocked(w_scale)
            wq_list.append(wq)
            w_scale_list.append(w_scale)
        wq = torch.stack(wq_list, dim=0).contiguous()
        w_scale = torch.stack(w_scale_list, dim=0).contiguous()

        # Quantize input per group
        xq_list = []
        x_scale_list = []
        for i in range(G):
            x_slice = X[i * M_per_group : (i + 1) * M_per_group]
            xs, xq = to_mxfp8(x_slice)
            xs = _to_blocked(xs)
            xq_list.append(xq)
            x_scale_list.append(xs)
        xq_actual = torch.cat(xq_list, dim=0).contiguous()
        x_scale_actual = torch.cat(x_scale_list, dim=0).contiguous()
        x_scale_actual = x_scale_actual.reshape(-1, K // block_size)

        # Pad XQ and x_scale to simulate static-shape padding
        xq_padded = torch.zeros(
            (padded_total_M, K), dtype=torch.float8_e4m3fn, device=self.device
        )
        xq_padded[:actual_total_M] = xq_actual

        actual_scale_rows = x_scale_actual.shape[0]
        padded_scale_rows = max(
            ((padded_total_M + 127) // 128) * 128, actual_scale_rows
        )
        x_scale_padded = torch.zeros(
            (padded_scale_rows, x_scale_actual.shape[1]),
            dtype=x_scale_actual.dtype,
            device=self.device,
        )
        x_scale_padded[:actual_scale_rows] = x_scale_actual

        # Run without hint (actual_num_tokens=None)
        out_no_hint = torch.empty(
            (padded_total_M, N), dtype=torch.bfloat16, device=self.device
        )
        y_no_hint = torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq_padded,
            wq.transpose(-2, -1),
            x_scale_padded,
            w_scale,
            offsets,
            out_no_hint,
        )

        # Run with hint (actual_num_tokens=actual_total_M)
        out_with_hint = torch.empty(
            (padded_total_M, N), dtype=torch.bfloat16, device=self.device
        )
        y_with_hint = torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq_padded,
            wq.transpose(-2, -1),
            x_scale_padded,
            w_scale,
            offsets,
            out_with_hint,
            actual_total_M,
        )

        # Outputs should be identical (only tile selection differs)
        torch.testing.assert_close(
            y_no_hint[:actual_total_M],
            y_with_hint[:actual_total_M],
            atol=0.0,
            rtol=0.0,
        )

        # Compare with bf16 reference
        y_bf16 = torch._grouped_mm(
            X,
            W.transpose(-2, -1),
            offs=offsets,
            out_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(
            y_with_hint[:actual_total_M], y_bf16, atol=8.0e-2, rtol=8.0e-2
        )
