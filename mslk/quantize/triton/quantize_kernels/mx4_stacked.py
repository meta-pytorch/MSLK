# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Self-contained MX4 (E8M0 scale) FP4 quantization kernel — stacked (MoE).

The shared per-segment row→segment mapping lives in ``stacked_segment_map``
(``fp4_primitives.layout``); the fully-slack-tile early-return stays inline
here (a ``@triton.jit`` helper cannot return out of its caller). Also hosts the
``quantize_mx4_stacked`` host wrapper that launches this kernel.
"""

from typing import Optional, Union

import torch
import triton  # @manual=//triton:triton
from mslk.quantize.triton.fp4_primitives import (
    blocked_scale_offset,
    convert_fp32_to_fp4_packed,
    mx4_scale_normalize_encode,
    RoundingMode,
    stacked_segment_map,
)
from mslk.quantize.triton.quantize_kernels import _resolve_seed
from triton import language as tl  # @manual=//triton:triton


@triton.jit
def quantize_mx4_stacked_kernel(
    x_ptr,  # [M, N] input tensor (bf16/fp16)
    q_ptr,  # [M, N//2] output — packed FP4 pairs (uint8, 2 values per byte)
    s_ptr,  # output — per-group E8M0 scales (int8) in padded blocked layout
    m_sizes_ptr,  # [num_segments] int64 — rows per segment
    stride_xm,  # row stride of x_ptr (elements to skip per row)
    stride_xn,  # col stride of x_ptr (usually 1 for contiguous)
    N,  # number of columns in the input matrix
    seed,  # int64 seed for tl.randint (Philox-3); only consulted when STOCHASTIC=True
    NUM_SEGMENTS: tl.constexpr,  # number of expert segments
    PREFIX_NUM: tl.constexpr,  # next_power_of_2(NUM_SEGMENTS) for tl.cumsum
    BSEARCH_ITERS: tl.constexpr,  # ceil(log2(NUM_SEGMENTS)) binary search iterations
    M_PER_BLOCK: tl.constexpr,  # rows per program (power of 2, capped at 128)
    USE_MASK: tl.constexpr,  # always True for stacked (segment boundaries unaligned)
    USE_INT64_INDEXING: tl.constexpr,  # True when M*N > 2^31-1
    GROUP_SIZE: tl.constexpr,  # elements per scale group (32 for MX4)
    ROUNDING_MODE: tl.constexpr,  # RoundingMode enum for shared exponent
    EBITS: tl.constexpr,  # exponent bits in target FP4 format (2 for E2M1)
    MBITS: tl.constexpr,  # mantissa bits in target FP4 format (1 for E2M1)
    N_COL_BLOCKS: tl.constexpr,  # ceil(num_scale_cols / 4) — blocked layout offset
    STOCHASTIC: tl.constexpr,  # True → enable software stochastic rounding
):
    """Stacked MX4 quantization kernel processing [M_PER_BLOCK, 64] tiles.

    Per-segment MoE quantization. The grid M-axis spans *padded* rows
    (``cdiv(padded_total_M, M_PER_BLOCK)``); each padded row is mapped to its
    segment + logical row via ``stacked_segment_map``, and per-group E8M0 scales
    are stored at the row's padded offset so each segment is 128-row aligned in
    the scale buffer. Padding rows within ``[0, padded_total)`` get
    ``is_valid_row=False`` → zero scales, written in-kernel; the over-allocated
    slack rows ``[padded_total, padded_total_M)`` are zeroed by the host
    ``torch.zeros`` allocation.

    Grid: (cdiv(N, 64), cdiv(padded_total_M, M_PER_BLOCK)).
    """
    TILE_N: tl.constexpr = 64  # type: ignore[Incompatible variable type]
    NUM_GROUPS: tl.constexpr = TILE_N // GROUP_SIZE

    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    # ========================================================================
    # STACKED setup — cumsum + binary search to map padded rows → segments.
    # The fully-slack-tile early-return stays inline (the helper cannot exit
    # its caller).
    # ========================================================================
    (
        padded_rows,
        logical_rows,
        is_valid_row,
        padded_total,
        _seg_idx,  # unused by MX4-stacked (no global scale); kept for tuple parity
    ) = stacked_segment_map(
        m_sizes_ptr, pid_m, M_PER_BLOCK, NUM_SEGMENTS, PREFIX_NUM, BSEARCH_ITERS
    )
    # Skip fully-slack tiles (beyond the last segment's padded rows).
    if pid_m * M_PER_BLOCK >= padded_total:
        return

    # Compute tile offsets and load [M_PER_BLOCK, 64] tile. Rows are indexed by
    # *logical* row (padded->logical mapped above); padding rows are masked out.
    offs_m = logical_rows[:, None]
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)[None, :]
    if USE_INT64_INDEXING:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    mask = is_valid_row[:, None] & (offs_n < N)
    other = 0.0

    load_offsets = offs_m * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptr + load_offsets, mask=mask, other=other)

    x_blocks = x.to(tl.float32).reshape(M_PER_BLOCK, NUM_GROUPS, GROUP_SIZE)

    # Per-group amax
    block_amax = tl.max(tl.abs(x_blocks), axis=2)  # [M_PER_BLOCK, NUM_GROUPS]

    # ========================================================================
    # MX4 scale computation (E8M0 biased exponent) + normalize + encode.
    # Shared with quantize_mx4_kernel via mx4_scale_normalize_encode.
    # ========================================================================
    x_blocks, encoded_scales = mx4_scale_normalize_encode(
        x_blocks,
        block_amax,
        pid_m,
        pid_n,
        seed,
        N,
        M_PER_BLOCK,
        NUM_GROUPS,
        GROUP_SIZE,
        ROUNDING_MODE,
        EBITS,
        MBITS,
        STOCHASTIC,
    )

    # ========================================================================
    # Mask out-of-bounds scales to 0 (padding rows + OOB columns)
    # ========================================================================
    if USE_MASK:
        scale_offs_n = pid_n * NUM_GROUPS + tl.arange(0, NUM_GROUPS)[None, :]
        num_scale_cols = N // GROUP_SIZE
        # Padding rows (is_valid_row=False) -> zero scales, written below.
        scale_mask = is_valid_row[:, None] & (scale_offs_n < num_scale_cols)
        encoded_scales = tl.where(scale_mask, encoded_scales, 0)

    # ========================================================================
    # Store scales — per-segment padded blocked layout.
    #
    # Each segment owns a contiguous run of 128-row-aligned blocks in the
    # blocked-layout buffer (per the dequantize_mx4_stacked slicing contract).
    # Split padded_r into a 128-row-block index and a within-block local row,
    # compute the local blocked offset using num_scale_cols, then add the
    # per-block base.
    # ========================================================================
    logical_col = pid_n * NUM_GROUPS + tl.arange(0, NUM_GROUPS)[None, :]
    num_scale_cols = N // GROUP_SIZE
    # The grid iterates padded rows directly, so padded_r == padded_rows.
    padded_r = padded_rows  # [M_PER_BLOCK]

    seg_block_idx = padded_r // 128  # [M_PER_BLOCK], int64
    padded_r_local = (padded_r % 128).to(tl.int32)  # [M_PER_BLOCK]
    local_offs = blocked_scale_offset(
        padded_r_local[:, None], logical_col, N_COL_BLOCKS, num_scale_cols
    )
    scale_offs = seg_block_idx[:, None].to(tl.int64) * (512 * N_COL_BLOCKS) + local_offs
    # Mask both row and column OOB:
    # - row: write every padded row in [0, padded_total) (padding rows carry
    #   zero encoded_scales from scale_mask above).
    # - column: logical_col >= num_scale_cols would collide with valid scales of
    #   subsequent rows.
    tl.store(
        s_ptr + scale_offs,
        encoded_scales,
        mask=(padded_rows[:, None] < padded_total) & (logical_col < num_scale_cols),
    )

    # ========================================================================
    # Pack to FP4 and store
    # ========================================================================
    x_fp4x2 = convert_fp32_to_fp4_packed(x_blocks.reshape(M_PER_BLOCK, 32, 2).split())

    q_offs_m = logical_rows[:, None]
    q_offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    q_mask = is_valid_row[:, None] & (q_offs_n < N // 2)

    if USE_INT64_INDEXING:
        q_offs_m = q_offs_m.to(tl.int64)
        q_offs_n = q_offs_n.to(tl.int64)

    store_offsets = q_offs_m * (N // 2) + q_offs_n
    tl.store(q_ptr + store_offsets, x_fp4x2, mask=q_mask)


def quantize_mx4_stacked(
    m_sizes: torch.Tensor,
    x: torch.Tensor,
    group_size: int = 32,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    *,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE activations to MX4 (E8M0 scales).

    MX4 analogue of ``quantize_nvfp4_stacked``: per-segment quantization with
    padded scale storage so each segment is 128-row aligned in the scale buffer.
    No global scale (MX4 has no per-tensor or per-segment global scale).

    Args:
        m_sizes: [num_segments] int64 — rows per expert segment.
        x: [M, N] concatenated activation tensor (bf16/fp16), M = sum(m_sizes).
        group_size: Elements per scale group (32 for MX4-32, 16 for MX4-16).
        rounding_mode: Rounding for shared exponent computation (default ceil).
            Supported: nearest (0), floor (1), even (2), stochastic (3), ceil (4).
        seed: Optional seed for stochastic rounding. Only used when
            ``rounding_mode == RoundingMode.stochastic``. If ``None`` (default),
            a seed is derived from
            ``torch.cuda.default_generators[device].initial_seed()`` so callers
            respect ``torch.manual_seed``. Pass a fixed integer for CUDA-graph
            reproducibility. Ignored for all deterministic rounding modes.

    Returns:
        xq: [M, N//2] packed FP4 data, dtype uint8 (matches ``quantize_mx4``).
        scales: E8M0 biased exponent bytes, dtype int8, **flattened 1D**.
            Logical shape is [padded_total_M_ub, padded_cols] in the Blackwell
            blocked layout, with each segment 128-row aligned. Padding rows
            between segments are zero.
    """
    stochastic = int(rounding_mode) == int(RoundingMode.stochastic)
    seed_int = _resolve_seed(seed, stochastic, x.device)

    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    num_segments = m_sizes.shape[0]

    assert N % group_size == 0, f"N must be divisible by {group_size}, got {N}"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Expected fp16/bf16, got {x.dtype}"
    assert m_sizes.dtype == torch.int64, f"Expected m_sizes int64, got {m_sizes.dtype}"
    assert m_sizes.is_cuda and m_sizes.device == x.device, (
        f"m_sizes must be on the same CUDA device as x "
        f"(got m_sizes.device={m_sizes.device}, x.device={x.device})"
    )

    # CUDA-graph-safe upper-bound padded rows (avoids a GPU→CPU sync on
    # m_sizes). Each segment can add at most 127 padding rows for 128-row
    # alignment.
    padded_total_M = M + num_segments * 127

    n_col_blocks = triton.cdiv(N // group_size, 4)
    padded_cols = n_col_blocks * 4

    xq = torch.empty(M, N // 2, dtype=torch.uint8, device=x.device)
    # Zero-init the scale buffer. The kernel writes scales only for rows in
    # ``[0, padded_total)`` where ``padded_total = sum(round_up(m_i, 128))`` is
    # the exact padded-row count it derives from ``m_sizes``; the CUDA-graph-safe
    # buffer is over-allocated to the upper bound ``padded_total_M = M +
    # num_segments * 127`` (to avoid a GPU→CPU sync), so the slack rows
    # ``[padded_total, padded_total_M)`` are never touched by the kernel and must
    # be zeroed here (``torch.empty`` leaves them uninitialized → NaN /
    # non-deterministic).
    scales = torch.zeros(padded_total_M, padded_cols, dtype=torch.int8, device=x.device)

    # M_PER_BLOCK: rows per program. Capped at 128 (scale layout alignment); at
    # M<256, cap at 64 to improve SM occupancy for small-M shapes.
    M_PER_BLOCK = 128 if M >= 256 else min(triton.next_power_of_2(M), 64)
    USE_MASK = True  # Always True for stacked (segment boundaries are unaligned)
    grid = (triton.cdiv(N, 64), triton.cdiv(padded_total_M, M_PER_BLOCK))
    use_int64 = M * N > 2**31 - 1
    prefix_num = triton.next_power_of_2(num_segments)
    bsearch_iters = max(1, (num_segments - 1).bit_length()) if num_segments > 1 else 1

    quantize_mx4_stacked_kernel[grid](  # pyre-ignore[28]
        x_2d,  # [M, N] input (2D-flattened)
        xq,  # [M, N//2] output packed FP4
        scales,  # [padded_total_M_ub, padded_cols] output scales (pre-zeroed)
        m_sizes,  # [num_segments] int64 rows per segment
        x_2d.stride(0),  # stride_xm
        x_2d.stride(1),  # stride_xn
        N,  # number of columns
        seed_int,  # int64 seed for tl.randint (unused when STOCHASTIC=False)
        # pyre-ignore[6]
        NUM_SEGMENTS=num_segments,
        # pyre-ignore[6]
        PREFIX_NUM=prefix_num,
        # pyre-ignore[6]
        BSEARCH_ITERS=bsearch_iters,
        # pyre-ignore[6]
        M_PER_BLOCK=M_PER_BLOCK,
        # pyre-ignore[6]
        USE_MASK=USE_MASK,
        # pyre-ignore[6]
        USE_INT64_INDEXING=use_int64,
        # pyre-ignore[6]
        GROUP_SIZE=group_size,
        # pyre-ignore[6]
        ROUNDING_MODE=int(rounding_mode),
        # pyre-ignore[6]
        EBITS=2,  # E2M1 exponent bits (MX4 is always E2M1)
        # pyre-ignore[6]
        MBITS=1,  # E2M1 mantissa bits
        # pyre-ignore[6]
        N_COL_BLOCKS=n_col_blocks,
        # pyre-ignore[6]
        STOCHASTIC=stochastic,
        # pyre-ignore[28]
        num_stages=3 if M >= 256 else 1,
        # pyre-ignore[28]
        num_warps=4,
    )

    return xq, scales.flatten()
