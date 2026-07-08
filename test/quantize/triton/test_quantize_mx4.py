# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""MX4 quantize tests for the MSLK copy: bitwise-vs-torch-reference + E8M0 checks."""

import unittest

import torch
from mslk.quantize.triton.fp4_primitives import RoundingMode
from mslk.quantize.triton.quantize_kernels.mx4 import quantize_mx4
from mslk.test.quantize.triton._mx4_torch_reference import (
    _unblock_mx4_scales,
    swizzle_scales_to_blocked,
    torch_quantize_mx4_ref,
)
from mslk.testing.device import skipUnlessCudaCapability, skipUnlessGfxArch
from mslk.utils.device import is_rocm
from parameterized import parameterized  # @manual


@skipUnlessGfxArch("gfx950")
@skipUnlessCudaCapability(10)
class QuantizeMX4Test(unittest.TestCase):
    """Bitwise-vs-torch-reference + first-principles tests for quantize_mx4."""

    @parameterized.expand(
        [
            # (name, M, N, group_size): canonical Blackwell layout + M/N padding +
            # unaligned-K (num_scale_cols < NUM_GROUPS) + group_size=16.
            ("canonical_gs32", 128, 256, 32),
            ("m_pad_gs32", 5, 64, 32),
            ("n_pad_gs32", 128, 32, 32),
            ("n_unaligned_gs32", 128, 96, 32),
            ("single_row_gs32", 1, 64, 32),
            ("large_gs32", 512, 4096, 32),
            ("canonical_gs16", 128, 256, 16),
            ("n_pad_gs16", 128, 16, 16),
            ("n_unaligned_gs16_80", 128, 80, 16),
            ("n_unaligned_gs16_144", 128, 144, 16),
            ("single_row_gs16", 1, 64, 16),
        ]
    )
    def test_bitwise_torch_ref(self, _name, M, N, group_size):
        """quantize_mx4 must be byte-identical to the pure-Torch reference.

        Pinned to RoundingMode.ceil because the reference (to_mxfp) is RCEIL-only.
        """
        torch.manual_seed(42)
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

        b_xq, b_scales = quantize_mx4(
            x, group_size=group_size, rounding_mode=RoundingMode.ceil
        )
        ref_xq, ref_scales_2d = torch_quantize_mx4_ref(
            x, group_size=group_size, swizzle=False
        )

        self.assertTrue(
            torch.equal(
                b_xq.view(torch.uint8).flatten(),
                ref_xq.view(torch.uint8).flatten(),
            ),
            f"FP4 data mismatch for shape ({M}, {N}, gs={group_size})",
        )
        if is_rocm():
            # ROCm returns the plain [M, K//32] layout — compare directly.
            b_cmp = b_scales.view(torch.uint8).reshape(M, -1).flatten()
            ref_cmp = ref_scales_2d.view(torch.uint8).reshape(M, -1).flatten()
        else:
            b_cmp = b_scales.view(torch.uint8).flatten()
            ref_cmp = (
                swizzle_scales_to_blocked(
                    ref_scales_2d, b_scales.shape, convention="mslk"
                )
                .view(torch.uint8)
                .flatten()
            )
        self.assertTrue(
            torch.equal(b_cmp, ref_cmp),
            f"Scale mismatch for shape ({M}, {N}, gs={group_size})",
        )

    @parameterized.expand(
        [
            # (name, group_max, expected_biased_exponent):
            # byte = ceil(log2(group_max)) - EBITS(2) + 127, stored as int8.
            ("pow2_4", 4.0, 127),
            ("nonpow2_3", 3.0, 127),
            ("nonpow2_5", 5.0, -128),  # 128 wraps to int8 -128 (unsigned E8M0 byte)
            ("one", 1.0, 125),
        ]
    )
    def test_shared_exponent_e8m0_encoding(self, _name, group_max, expected_exp):
        """Shared E8M0 exponent matches the OCP MX v1.0 ceil-rounded formula."""
        x = torch.zeros(1, 32, dtype=torch.bfloat16, device="cuda")
        x[0, 0] = group_max
        _, scales = quantize_mx4(x)
        if is_rocm():
            # ROCm returns the plain [M, K//32] layout (uint8) — no un-blocking.
            # View as int8 so the byte matches the int8 expected encoding.
            s = scales.view(torch.int8).reshape(1, -1)
        else:
            s = _unblock_mx4_scales(scales.view(torch.int8), 1, 1)
        self.assertEqual(s[0, 0].item(), expected_exp)

    @parameterized.expand(
        [
            ("nan", float("nan")),
            ("pos_inf", float("inf")),
            ("neg_inf", float("-inf")),
        ]
    )
    def test_quantize_mx4_special_value_saturation(self, _name, bad_value):
        """A single NaN/±Inf is saturated; scales stay finite."""
        torch.manual_seed(701)
        M, N = 64, 128
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        x[0, 0] = bad_value
        xq, scales = quantize_mx4(x)
        self.assertEqual(xq.shape, (M, N // 2))
        self.assertFalse(
            torch.isnan(scales.to(torch.float32)).any(), "scales contain NaN"
        )
        self.assertFalse(
            torch.isinf(scales.to(torch.float32)).any(), "scales contain Inf"
        )
