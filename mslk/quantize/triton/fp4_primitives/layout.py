# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Storage layout primitives for the Blackwell 128×4 FP4 scale arrangement.

Provides the blocked-layout offset helper and the stacked-MoE segment map used
by the MX4 quantize kernels.
"""

import triton  # @manual
from triton import language as tl  # @manual


@triton.jit
def blocked_scale_offset(logical_row, logical_col, n_col_blocks, num_cols):
    """Convert logical (row, col) scale indices to Blackwell blocked layout offsets.

    The row stride is padded to a multiple of 4 columns (``n_col_blocks * 4``), matching
    the canonical NVIDIA/cuBLAS block-scaled layout (and ``_to_blocked`` /
    ``_from_blocked``). When ``num_cols`` is already a multiple of 4 this is unchanged;
    otherwise it places each logical row at the padded stride so the dense un-blocking
    inverse (and the GEMM) read it back from the correct offset.

    Args:
        logical_row: Row indices into the logical scale matrix (int tensor).
        logical_col: Column indices into the logical scale matrix (int tensor).
        n_col_blocks: Number of 4-column blocks (ceil(total_scale_cols / 4)).
        num_cols: Unpadded scale columns per row; used by callers for the store mask
            (the padded tail columns are left zero), not for the offset itself.

    Returns:
        Flat offsets into the blocked layout buffer.
    """
    # Pad the row stride to a multiple of 4 columns so the layout matches the canonical
    # NVIDIA blocked-scale convention even when num_cols is not a multiple of 4.
    flat_idx = logical_row * (n_col_blocks * 4) + logical_col

    elems_per_row_block = 512 * n_col_blocks

    first_dim = flat_idx // elems_per_row_block
    second_dim = (flat_idx % elems_per_row_block) // (128 * n_col_blocks)
    third_dim = (flat_idx % (128 * n_col_blocks)) // (4 * n_col_blocks)
    fourth_dim = (flat_idx % (4 * n_col_blocks)) // 4
    fifth_dim = flat_idx % 4

    return (
        first_dim * elems_per_row_block
        + fourth_dim * 512
        + third_dim * 16
        + second_dim * 4
        + fifth_dim
    )


@triton.jit
def stacked_segment_map(
    m_sizes_ptr,  # [num_segments] int64 — rows per segment
    pid_m,  # program id along the padded-row (M) grid axis
    M_PER_BLOCK: tl.constexpr,  # rows per program (power of 2, capped at 128)
    NUM_SEGMENTS: tl.constexpr,  # number of expert segments
    PREFIX_NUM: tl.constexpr,  # next_power_of_2(NUM_SEGMENTS) for tl.cumsum
    BSEARCH_ITERS: tl.constexpr,  # ceil(log2(NUM_SEGMENTS)) binary search iterations
):
    """Map a tile of *padded* rows back to their MoE segments (stacked quant).

    Loads ``m_sizes``, computes inclusive/exclusive cumsums of the logical and
    128-row-padded segment row counts, then binary-searches each padded row in
    this program's tile to its segment, deriving the logical row index and a
    per-row validity mask.

    Returns (all shape ``[M_PER_BLOCK]`` unless noted):
      - ``padded_rows`` (int64): padded-row indices owned by this program.
      - ``logical_rows``: logical (unpadded) row index, clamped to 0 where invalid.
      - ``is_valid_row`` (bool): row is a real (non-padding) row within the buffer.
      - ``padded_total`` (scalar): total padded-row count across all segments.
      - ``seg_idx``: segment each padded row belongs to (used by NVFP4-stacked for
        the per-row global-scale gather; MX4-stacked ignores it).

    Only these five are returned. The intermediate cumsums
    (``cumsum_exc``/``padded_cumsum_exc``/``seg_cumsum``/``seg_padded_cumsum``/``seg_m``)
    are computed locally to derive ``logical_rows``/``is_valid_row`` and are not
    returned — no caller needs them.

    NOTE: the fully-slack-tile early-return
    (``if pid_m * M_PER_BLOCK >= padded_total: return``) cannot live here — a
    ``@triton.jit`` helper cannot return out of its caller — so each kernel must
    perform that early-return itself using the returned ``padded_total``.
    """
    seg_offs = tl.arange(0, PREFIX_NUM)
    seg_mask = seg_offs < NUM_SEGMENTS
    m_vals = tl.load(m_sizes_ptr + seg_offs, mask=seg_mask, other=0)
    cumsum_inc = tl.cumsum(m_vals, axis=0)
    padded_vals = ((m_vals + 127) // 128) * 128
    padded_cumsum_inc = tl.cumsum(padded_vals, axis=0)
    # Exclusive cumsums: [0, m_0, m_0+m_1, ...]
    cumsum_exc = cumsum_inc - m_vals
    padded_cumsum_exc = padded_cumsum_inc - padded_vals

    # Promote to int64: padded_r * num_scale_cols can overflow int32 for
    # large stacked inputs.
    padded_total = tl.sum(padded_vals, axis=0)  # actual padded-row count
    padded_rows = (pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)).to(
        tl.int64
    )  # [M_PER_BLOCK] — padded-row indices

    # Binary search padded_rows against padded_cumsum_exc → segment.
    lo = tl.zeros([M_PER_BLOCK], dtype=tl.int32)
    hi = tl.full([M_PER_BLOCK], NUM_SEGMENTS - 1, dtype=tl.int32)
    for _ in range(BSEARCH_ITERS):
        mid = (lo + hi + 1) // 2
        mid_val = tl.gather(padded_cumsum_exc, mid, 0)
        lo = tl.where(mid_val <= padded_rows, mid, lo)
        hi = tl.where(mid_val <= padded_rows, hi, mid - 1)
    seg_idx = lo  # [M_PER_BLOCK]

    seg_cumsum = tl.gather(cumsum_exc, seg_idx, 0)  # [M_PER_BLOCK]
    seg_padded_cumsum = tl.gather(padded_cumsum_exc, seg_idx, 0)
    seg_m = tl.gather(m_vals, seg_idx, 0)
    padded_r_in_seg = padded_rows - seg_padded_cumsum  # [M_PER_BLOCK]
    # Valid = within the segment's logical rows AND within actual padded total.
    is_valid_row = (padded_r_in_seg < seg_m) & (padded_rows < padded_total)
    # Logical row index (safe-clamped to 0 where invalid, for masked loads).
    logical_rows = tl.where(
        is_valid_row, seg_cumsum + padded_r_in_seg, 0
    )  # [M_PER_BLOCK]

    return (
        padded_rows,
        logical_rows,
        is_valid_row,
        padded_total,
        seg_idx,
    )
