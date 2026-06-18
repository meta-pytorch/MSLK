# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Fused MXFP8 quantization kernel: BF16 → FP8 E4M3 + blocked E8M0 scales.

Each program processes one row and GROUPS_PER_PROGRAM scale blocks (256 elements).
Grid = (M, K // (32*GROUPS_PER_PROGRAM)). Balances parallelism vs launch overhead.
"""

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton

E8M0_BIAS: int = 127
FP8_E4M3_MAX: float = 448.0


@triton.jit
def _fused_mxfp8_quant_kernel(
    X_ptr,
    OUT_ptr,
    SCALE_ptr,
    M,
    K,
    n_scale_cols,
    n_col_blocks,
    stride_xm,
    stride_xk,
    GROUPS_PER_PROG: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    tile_id = tl.program_id(1)

    if row >= M:
        return

    # Precompute blocked layout indices for this row
    row_block_idx = row // 128
    sub_row = (row % 128) // 32
    row_in_sub = (row % 128) % 32

    # This program handles GROUPS_PER_PROG consecutive scale groups
    start_group = tile_id * GROUPS_PER_PROG

    for g in tl.static_range(GROUPS_PER_PROG):
        scale_col = start_group + g
        if scale_col < n_scale_cols:
            base_k = scale_col * GROUP_SIZE
            offsets = tl.arange(0, GROUP_SIZE)

            x = tl.load(
                X_ptr + row * stride_xm + (base_k + offsets) * stride_xk,
            ).to(tl.float32)

            # Per-block amax
            abs_x = tl.abs(x)
            amax = tl.max(abs_x, axis=0)
            amax = tl.maximum(amax, 1e-12)

            # E8M0 scale
            log2_scale = tl.math.log2(amax / FP8_E4M3_MAX)
            biased_exp = tl.math.ceil(log2_scale)
            biased_exp = tl.minimum(tl.maximum(biased_exp, -127.0), 127.0)
            e8m0_scale = (biased_exp + E8M0_BIAS).to(tl.uint8)

            # Quantize
            descale = tl.math.exp2(-biased_exp)
            scaled = x * descale
            scaled = tl.minimum(tl.maximum(scaled, -FP8_E4M3_MAX), FP8_E4M3_MAX)

            # Store FP8
            tl.store(
                OUT_ptr + row * stride_xm + (base_k + offsets) * stride_xk,
                scaled.to(tl.float8e4nv),
            )

            # Blocked scale offset
            col_block_idx = scale_col // 4
            local_col = scale_col % 4

            blocked_offset = (
                row_block_idx * (n_col_blocks * 512)
                + col_block_idx * 512
                + row_in_sub * 16
                + sub_row * 4
                + local_col
            )

            tl.store(SCALE_ptr + blocked_offset, e8m0_scale)


def fused_mxfp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused BF16 → FP8 E4M3 + blocked E8M0 scales in a single kernel.

    Args:
        x: BF16 tensor of shape [M, K] where K % 32 == 0

    Returns:
        (fp8_data, blocked_scales): FP8 E4M3 tensor [M, K], uint8 scales [flat]
    """
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
    assert x.dtype == torch.bfloat16, f"Expected bfloat16, got {x.dtype}"
    M, K = x.shape
    assert K % 32 == 0, f"K must be multiple of 32, got {K}"

    GROUP_SIZE = 32
    GROUPS_PER_PROG = 16  # Each program handles 16 groups = 512 elements
    n_scale_cols = K // GROUP_SIZE
    n_row_blocks = (M + 127) // 128
    n_col_blocks = (n_scale_cols + 3) // 4

    out_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    total_scales = n_row_blocks * n_col_blocks * 512
    out_scales = torch.zeros(total_scales, dtype=torch.uint8, device=x.device)

    n_tiles = (n_scale_cols + GROUPS_PER_PROG - 1) // GROUPS_PER_PROG
    grid = (M, n_tiles)

    _fused_mxfp8_quant_kernel[grid](
        x,
        out_fp8,
        out_scales,
        M,
        K,
        n_scale_cols,
        n_col_blocks,
        x.stride(0),
        x.stride(1),
        GROUPS_PER_PROG=GROUPS_PER_PROG,
        GROUP_SIZE=GROUP_SIZE,
    )

    return out_fp8, out_scales
