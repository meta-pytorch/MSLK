# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Self-contained MX4 (E8M0 scale) FP4 quantization kernel — non-stacked.

Also hosts the ``quantize_mx4`` host wrapper that launches this kernel.
"""

from typing import Optional, Union

import torch
import triton  # @manual=//triton:triton
from mslk.quantize.triton.fp4_primitives import (
    blocked_scale_offset,
    convert_fp32_to_fp4_packed,
    mx4_scale_normalize_encode,
    RoundingMode,
)
from mslk.quantize.triton.quantize_kernels import _resolve_seed
from mslk.utils.device import is_gfx950, is_rocm
from triton import language as tl  # @manual=//triton:triton


@triton.jit
def quantize_mx4_kernel(
    x_ptr,  # [M, N] input tensor (bf16/fp16)
    q_ptr,  # [M, N//2] output — packed FP4 pairs (uint8, 2 values per byte)
    s_ptr,  # output — per-group E8M0 scales (int8) in blocked layout
    stride_xm,  # row stride of x_ptr (elements to skip per row)
    stride_xn,  # col stride of x_ptr (usually 1 for contiguous)
    M,  # number of rows in the input matrix
    N,  # number of columns in the input matrix
    seed,  # int64 seed for tl.randint (Philox-3); only consulted when STOCHASTIC=True
    M_PER_BLOCK: tl.constexpr,  # rows per program (power of 2, capped at 128)
    USE_MASK: tl.constexpr,  # True when M or N don't align to tile boundaries
    USE_INT64_INDEXING: tl.constexpr,  # True when M*N > 2^31-1
    GROUP_SIZE: tl.constexpr,  # elements per scale group (32 for MX4)
    ROUNDING_MODE: tl.constexpr,  # RoundingMode enum for shared exponent
    EBITS: tl.constexpr,  # exponent bits in target FP4 format (2 for E2M1)
    MBITS: tl.constexpr,  # mantissa bits in target FP4 format (1 for E2M1)
    N_COL_BLOCKS: tl.constexpr,  # ceil(num_scale_cols / 4) — blocked layout offset
    STOCHASTIC: tl.constexpr,  # True → enable software stochastic rounding
    IS_ROCM: tl.constexpr,  # True → plain [M, K//32] scale layout (AMD GEMM contract)
    IS_GFX950: tl.constexpr,  # True → gfx950 native FP4 packing instruction
):
    """MX4 quantization kernel processing [M_PER_BLOCK, 64] tiles.

    Each program quantizes a tile of [M_PER_BLOCK, 64] elements into packed FP4
    bytes and per-group E8M0 (power-of-two biased int8) scales in the Blackwell
    blocked layout. MX4 has no global scale.

    Grid: (cdiv(N, 64), cdiv(M, M_PER_BLOCK) [+1 for tail zeroing when
    M_PER_BLOCK != 128]).
    """
    TILE_N: tl.constexpr = 64  # type: ignore[Incompatible variable type]
    NUM_GROUPS: tl.constexpr = TILE_N // GROUP_SIZE

    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    # ========================================================================
    # Tail-scale zeroing: when M_PER_BLOCK != 128, extra blocks zero out padded
    # scales in the 128-row-aligned tail for tensor core compatibility.
    # ========================================================================
    if (not IS_ROCM) and M_PER_BLOCK != 128 and pid_m * M_PER_BLOCK >= M:
        offs_m = tl.arange(0, 128)[:, None]
        logical_col = pid_n * NUM_GROUPS + tl.arange(0, NUM_GROUPS)[None, :]
        num_scale_cols = N // GROUP_SIZE
        scale_offs = blocked_scale_offset(
            offs_m, logical_col, N_COL_BLOCKS, num_scale_cols
        )
        oob_mask = (offs_m >= M) & tl.full((NUM_GROUPS,), True, dtype=tl.int1)[None, :]
        zero_scales = tl.full([128, NUM_GROUPS], 0, dtype=tl.int8)
        tl.store(s_ptr + scale_offs, zero_scales, mask=oob_mask)
        return

    # Compute tile offsets and load [M_PER_BLOCK, 64] tile.
    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)[None, :]
    if USE_INT64_INDEXING:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    if USE_MASK:
        mask = (offs_m < M) & (offs_n < N)
        other = 0.0
    else:
        mask = None
        other = None

    load_offsets = offs_m * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptr + load_offsets, mask=mask, other=other)

    x_blocks = x.to(tl.float32).reshape(M_PER_BLOCK, NUM_GROUPS, GROUP_SIZE)

    # Per-group amax
    block_amax = tl.max(tl.abs(x_blocks), axis=2)  # [M_PER_BLOCK, NUM_GROUPS]

    # ========================================================================
    # MX4 scale computation (E8M0 biased exponent) + normalize + encode.
    # Shared with quantize_mx4_stacked_kernel via mx4_scale_normalize_encode.
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
    # Mask out-of-bounds scales to 0
    # ========================================================================
    if USE_MASK:
        scale_offs_n = pid_n * NUM_GROUPS + tl.arange(0, NUM_GROUPS)[None, :]
        num_scale_cols = N // GROUP_SIZE
        scale_mask = (offs_m < M) & (scale_offs_n < num_scale_cols)
        encoded_scales = tl.where(scale_mask, encoded_scales, 0)

    # ========================================================================
    # Store scales — Blackwell blocked layout
    # ========================================================================
    logical_col = pid_n * NUM_GROUPS + tl.arange(0, NUM_GROUPS)[None, :]
    num_scale_cols = N // GROUP_SIZE
    logical_row = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    if USE_INT64_INDEXING:
        # flat_idx = logical_row * num_scale_cols overflows int32 once
        # M * (N // GROUP_SIZE) > 2**31; promote like the data-offset paths above.
        logical_row = logical_row.to(tl.int64)
        logical_col = logical_col.to(tl.int64)
    if IS_ROCM:
        # AMD GEMMs consume the plain [M, num_scale_cols] row-major layout (no
        # swizzle / 128-row pad). The buffer has exactly M rows, so tail rows
        # from a partial last tile (logical_row >= M) must be masked out.
        scale_offs = logical_row * num_scale_cols + logical_col
        scale_store_mask = (logical_col < num_scale_cols) & (logical_row < M)
    else:
        scale_offs = blocked_scale_offset(
            logical_row, logical_col, N_COL_BLOCKS, num_scale_cols
        )
        # Mask out scale columns beyond num_scale_cols (= N // GROUP_SIZE).
        # Without this mask, writes for col >= num_scale_cols collide with
        # valid scales of subsequent rows in the swizzled layout. Rows are
        # already on-tile by construction in the non-stacked path.
        scale_store_mask = logical_col < num_scale_cols
    tl.store(
        s_ptr + scale_offs,
        encoded_scales,
        mask=scale_store_mask,
    )

    # ========================================================================
    # Pack to FP4 and store
    # ========================================================================
    x_fp4x2 = convert_fp32_to_fp4_packed(
        x_blocks.reshape(M_PER_BLOCK, 32, 2).split(),
        IS_GFX950=IS_GFX950,
        IS_ROCM=IS_ROCM,
    )

    q_offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    q_offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    if USE_MASK:
        q_mask = (q_offs_m < M) & (q_offs_n < N // 2)
    else:
        q_mask = None

    if USE_INT64_INDEXING:
        q_offs_m = q_offs_m.to(tl.int64)
        q_offs_n = q_offs_n.to(tl.int64)

    store_offsets = q_offs_m * (N // 2) + q_offs_n
    tl.store(q_ptr + store_offsets, x_fp4x2, mask=q_mask)


def quantize_mx4(
    x: torch.Tensor,
    group_size: int = 32,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    *,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to MX4 format.

    Args:
        x: Input tensor, shape [..., N] where N % group_size == 0. bf16 or fp16.
           Supports arbitrary leading dimensions (e.g., [B, S, N] or [M, N]).
        group_size: Elements per scale group (32 for MX4-32, 16 for MX4-16).
        rounding_mode: Rounding for shared exponent computation (default: ceil).
            ceil (4) is the default and guarantees no overflow.
            The OCP MX v1.0 reference implementation uses even pre-rounding
            (RoundingMode.even = 2) which is slightly more precise (~0.5 bits).
            Supported: nearest (0), floor (1), even (2), stochastic (3), ceil (4).
        seed: Optional seed for stochastic rounding. Only used when
            ``rounding_mode == RoundingMode.stochastic``. If ``None`` (default),
            a seed is derived from
            ``torch.cuda.default_generators[device].initial_seed()`` so callers
            respect ``torch.manual_seed``. Pass a fixed integer for CUDA-graph
            reproducibility. Ignored for all deterministic rounding modes.

    Returns:
        xq: Packed FP4 data, dtype uint8, shape [..., -1, N//2]. Leading dims preserved.
        scales: E8M0 biased exponent bytes.
            NVIDIA: dtype int8, **flattened 1D**, size = padded_M × padded_K
                (both padded to 128/4 multiples). Layout: Blackwell blocked
                (swizzled).
            ROCm: dtype uint8, shape [..., -1, scale_K] where
                scale_K = N // group_size. Plain row-major, no padding.

    eg. [1, 5120] bf16 -> [1, 2560] uint8 xq
        NVIDIA: + [128×160] int8 scales (flattened blocked)
        ROCm:   + [1, 160] uint8 scales (row-major)
    """
    stochastic = int(rounding_mode) == int(RoundingMode.stochastic)
    seed_int = _resolve_seed(seed, stochastic, x.device)

    rocm = is_rocm()
    gfx950 = is_gfx950()
    if stochastic and rocm:
        raise NotImplementedError(
            "quantize_mx4: stochastic rounding is not supported on ROCm."
        )

    orig_leading_dims, orig_N = x.shape[:-2], x.shape[-1]
    x_2d = x.reshape(-1, orig_N)
    M, N = x_2d.shape

    assert N % group_size == 0, f"N must be divisible by {group_size}, got {N}"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"Expected fp16/bf16, got {x.dtype}"

    scale_K = N // group_size
    # Scale padding (128-row × 4-col alignment for tensor cores).
    n_col_blocks = triton.cdiv(scale_K, 4)

    xq = x.new_empty(M, N // 2, dtype=torch.uint8)
    if rocm:
        # AMD kernels consume the plain [M, K//32] E8M0 layout (no swizzle/pad).
        scales = torch.zeros(M, scale_K, dtype=torch.int8, device=x.device)
    else:
        padded_rows = triton.cdiv(M, 128) * 128
        padded_cols = n_col_blocks * 4
        # Zero-init so OOB positions (where the kernel does not write) are zero.
        scales = torch.zeros(
            padded_rows, padded_cols, dtype=torch.int8, device=x.device
        )

    # M_PER_BLOCK: rows per program. Capped at 128 (scale layout alignment).
    M_PER_BLOCK = min(triton.next_power_of_2(M), 128)
    USE_MASK = M % M_PER_BLOCK != 0 or N % 64 != 0
    grid = (triton.cdiv(N, 64), triton.cdiv(M, M_PER_BLOCK))
    # Extra row of blocks to zero out tail scales when M_PER_BLOCK < 128
    # (NVIDIA padded layout only; ROCm has no padded tail).
    if M_PER_BLOCK != 128 and not rocm:
        grid = (grid[0], grid[1] + 1)
    use_int64 = M * N > 2**31 - 1

    quantize_mx4_kernel[grid](  # pyre-ignore[28]
        x_2d,  # [M, N] input (2D-flattened)
        xq,  # [M, N//2] output packed FP4
        scales,  # [padded_M, padded_cols] output E8M0 scales
        x_2d.stride(0),  # stride_xm
        x_2d.stride(1),  # stride_xn
        M,  # number of rows
        N,  # number of columns
        seed_int,  # int64 seed for tl.randint (unused when STOCHASTIC=False)
        # pyre-ignore[6]
        M_PER_BLOCK=M_PER_BLOCK,
        # pyre-ignore[6]
        USE_MASK=USE_MASK,
        # pyre-ignore[6]
        USE_INT64_INDEXING=use_int64,
        # pyre-ignore[6]
        GROUP_SIZE=group_size,  # 32 for MX4-32, 16 for MX4-16
        # pyre-ignore[6]
        ROUNDING_MODE=int(rounding_mode),  # ceil, floor, nearest, even, or stochastic
        # pyre-ignore[6]
        EBITS=2,  # E2M1 exponent bits (MX4 is always E2M1)
        # pyre-ignore[6]
        MBITS=1,  # E2M1 mantissa bits
        # pyre-ignore[6]
        N_COL_BLOCKS=n_col_blocks,
        # pyre-ignore[6]
        STOCHASTIC=stochastic,
        # pyre-ignore[6]
        IS_ROCM=rocm,
        # pyre-ignore[6]
        IS_GFX950=gfx950,
        # pyre-ignore[28]
        num_stages=3 if M >= 256 else 1,
        # pyre-ignore[28]
        num_warps=4,
    )

    xq = xq.view(*orig_leading_dims, -1, N // 2)
    if rocm:
        # Plain [..., K//32] uint8 E8M0 layout (matches the legacy ROCm return).
        return xq, scales.view(torch.uint8).reshape(*orig_leading_dims, -1, scale_K)
    return xq, scales.flatten()
