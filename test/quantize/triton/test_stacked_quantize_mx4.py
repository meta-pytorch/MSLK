# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Stacked (MoE) MX4 quantize tests for the MSLK copy: bitwise-vs-torch-reference."""

import math
import unittest

import torch
from mslk.quantize.triton.fp4_primitives import RoundingMode
from mslk.quantize.triton.quantize_kernels.mx4 import quantize_mx4
from mslk.quantize.triton.quantize_kernels.mx4_stacked import quantize_mx4_stacked
from mslk.test.quantize.triton._mx4_torch_reference import (
    swizzle_scales_to_blocked,
    torch_quantize_mx4_ref,
)
from mslk.testing.device import skipUnlessCudaCapability, skipUnlessGfxArch
from mslk.utils.device import is_rocm
from parameterized import parameterized  # @manual


def _make_stacked_input(m_sizes_list, N, dtype=torch.bfloat16, device="cuda"):
    """Create a stacked input tensor and m_sizes from a list of segment sizes."""
    m_sizes = torch.tensor(m_sizes_list, dtype=torch.int64, device=device)
    M = sum(m_sizes_list)
    x = torch.randn(M, N, dtype=dtype, device=device)
    return x, m_sizes


def _per_segment_padded_offsets(m_sizes_list):
    """Yield (seg_idx, m_i, seg_row_start, padded_start, padded_rows) tuples."""
    seg_row_start = 0
    padded_start = 0
    for i, m_i in enumerate(m_sizes_list):
        padded_rows = math.ceil(m_i / 128) * 128 if m_i > 0 else 0
        yield i, m_i, seg_row_start, padded_start, padded_rows
        seg_row_start += m_i
        padded_start += padded_rows


def _slice_padded_scales(scales_flat, padded_start, padded_rows, padded_cols):
    """Slice the segment's padded sub-buffer out of the stacked flat scales."""
    start = padded_start * padded_cols
    end = (padded_start + padded_rows) * padded_cols
    return scales_flat[start:end]


@skipUnlessGfxArch("gfx950")
@skipUnlessCudaCapability(10)
class StackedQuantizeMX4Test(unittest.TestCase):
    """Bitwise-vs-torch-reference tests for stacked (MoE) MX4 quantization."""

    @parameterized.expand(
        [
            ("two_equal", [64, 64], 64, 32),
            ("three_unequal", [50, 30, 48], 64, 32),
            ("four_mixed", [16, 64, 128, 256], 128, 32),
            ("two_large", [256, 512], 512, 32),
            ("gs16_three_unequal", [50, 30, 48], 64, 16),
            ("zero_row_segment", [64, 0, 64], 64, 32),
            # Many small segments (PREFIX_NUM=64, BSEARCH_ITERS=6 path).
            ("many_segments", [16] * 64, 128, 32),
            # group_size=16 with N=80 (N%64≠0, (N/16)%4≠0): column-side swizzle-pad.
            ("gs16_unaligned_n", [128, 128], 80, 16),
            # Sub-block-height + unaligned segments at gs16 (gs16 avoids the gs32
            # torch.empty padding-leak in the per-segment scale comparison).
            ("single_row_segments_gs16", [1, 1, 1, 1], 64, 16),
            ("unaligned_segment_boundary_gs16", [5, 11, 19], 64, 16),
        ]
    )
    def test_bitwise_torch_ref_match_stacked(self, _name, m_sizes_list, N, group_size):
        """quantize_mx4_stacked must be byte-identical to per-segment
        torch_quantize_mx4_ref + concat (xq and per-segment swizzled scales)."""
        torch.manual_seed(42)
        x, m_sizes = _make_stacked_input(m_sizes_list, N)

        b_xq, b_scales = quantize_mx4_stacked(
            m_sizes, x, group_size=group_size, rounding_mode=RoundingMode.ceil
        )

        num_scale_cols = N // group_size
        n_col_blocks = math.ceil(num_scale_cols / 4)
        padded_cols = n_col_blocks * 4

        ref_xq_chunks = []
        for (
            i,
            m_i,
            seg_row_start,
            padded_start,
            padded_rows,
        ) in _per_segment_padded_offsets(m_sizes_list):
            if m_i == 0:
                continue
            seg_end = seg_row_start + m_i
            seg = x[seg_row_start:seg_end]
            seg_xq_ref, seg_scales_2d = torch_quantize_mx4_ref(
                seg, group_size=group_size, swizzle=False
            )
            ref_xq_chunks.append(seg_xq_ref)

            if is_rocm():
                # ROCm: plain [M, num_scale_cols] by logical row — compare the
                # segment's rows directly (no padded slice / swizzle).
                scales_u8 = b_scales.view(torch.uint8).reshape(-1, num_scale_cols)
                b_seg_plain = scales_u8[seg_row_start:seg_end]
                self.assertTrue(
                    torch.equal(
                        b_seg_plain.flatten(),
                        seg_scales_2d.view(torch.uint8).flatten(),
                    ),
                    f"Segment {i} (m={m_i}) plain-scale mismatch, "
                    f"m_sizes={m_sizes_list}",
                )
            else:
                b_seg_slice = _slice_padded_scales(
                    b_scales, padded_start, padded_rows, padded_cols
                )

                seg_scales_padded = torch.zeros(
                    (padded_rows, num_scale_cols),
                    dtype=torch.uint8,
                    device=seg.device,
                )
                seg_scales_padded[:m_i, :] = seg_scales_2d.view(torch.uint8)
                seg_swizzled = swizzle_scales_to_blocked(
                    seg_scales_padded,
                    torch.Size([padded_rows * padded_cols]),
                    convention="mslk",
                )

                self.assertTrue(
                    torch.equal(
                        b_seg_slice.view(torch.uint8).flatten(),
                        seg_swizzled.view(torch.uint8).flatten(),
                    ),
                    f"Segment {i} (m={m_i}) scale mismatch for m_sizes={m_sizes_list}",
                )

        ref_xq = (
            torch.cat(ref_xq_chunks, dim=0)
            if ref_xq_chunks
            else b_xq.new_zeros(b_xq.shape)
        )
        self.assertTrue(
            torch.equal(
                b_xq.view(torch.uint8).flatten(),
                ref_xq.view(torch.uint8).flatten(),
            ),
            f"FP4 data mismatch for m_sizes={m_sizes_list}, N={N}, gs={group_size}",
        )

    # NaN/Inf saturation: a special value in one segment must not corrupt the
    # per-segment output (each segment still matches independent quantize_mx4).

    def _check_per_segment_with_modified_x(self, m_sizes_list, N, modify_fn):
        """Apply modify_fn(x) to inject special values, then run per-segment check."""
        x, m_sizes = _make_stacked_input(m_sizes_list, N)
        modify_fn(x)

        xq_stacked, _ = quantize_mx4_stacked(m_sizes, x)

        seg_row_start = 0
        for i, m_i in enumerate(m_sizes_list):
            seg_end = seg_row_start + m_i
            x_i = x[seg_row_start:seg_end]
            xq_ref_i, _ = quantize_mx4(x_i)
            xq_stacked_i = xq_stacked[seg_row_start:seg_end]
            self.assertTrue(
                torch.equal(xq_stacked_i, xq_ref_i),
                f"Segment {i} (m={m_i}) xq differs after modify_fn",
            )
            seg_row_start += m_i

    @parameterized.expand(
        [
            ("inf", (5, 7), float("inf")),
            ("neg_inf", (10, 3), float("-inf")),
            ("nan", (20, 5), float("nan")),
        ]
    )
    def test_special_value_per_segment_unaffected(self, _name, pos, value):
        """A NaN/±Inf in one segment leaves every segment's xq matching quantize_mx4."""
        torch.manual_seed(10)

        def inject(x):
            x[pos] = value

        self._check_per_segment_with_modified_x([32, 64], 64, inject)

    def test_no_host_sync_under_cuda_graph(self):
        """Stacked MX4 quantization captures cleanly under CUDA graph.

        Any GPU→CPU sync inside the wrapper would fail capture; after replay,
        outputs must match an eager call.
        """
        torch.manual_seed(31)
        x, m_sizes = _make_stacked_input([32, 64, 128], 64)

        xq_ref, scales_ref = quantize_mx4_stacked(m_sizes, x)

        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                _ = quantize_mx4_stacked(m_sizes, x)
        torch.cuda.current_stream().wait_stream(s)

        with torch.cuda.graph(graph):
            xq_captured, scales_captured = quantize_mx4_stacked(m_sizes, x)

        graph.replay()
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(xq_captured, xq_ref))
        self.assertTrue(torch.equal(scales_captured, scales_ref))
