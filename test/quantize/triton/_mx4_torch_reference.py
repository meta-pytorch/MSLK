# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Torch-based correctness helpers for the MSLK MX4 quantize tests."""

import torch
from torch import Tensor
from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked, pack_uint4


_F32_EXP_BIAS = 127
_F32_EXP_MASK = 0x7F800000
_F32_EXP_OFFSET = 23
_F32_CEIL_ROUND = (1 << 23) - 1
_FP4_EBITS = 2
_FP4_MBITS = 1


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _to_mxfp4(
    data_hp: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(
        *orig_shape[:-1], orig_shape[-1] // block_size, block_size
    ).to(torch.float32)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)
    min_normal = torch.finfo(torch.float32).smallest_normal
    safe_max_abs = torch.where(max_abs == 0, min_normal, max_abs)

    bits = safe_max_abs.view(torch.int32) + _F32_CEIL_ROUND
    exp_bits = (bits & _F32_EXP_MASK) >> _F32_EXP_OFFSET
    group_exp = (exp_bits - _F32_EXP_BIAS - _FP4_EBITS).clamp(-127, 125)

    scale = torch.exp2(group_exp.to(torch.float64)).to(torch.float32)
    unpacked = _f32_to_floatx_unpacked(
        (data_hp / scale).contiguous(), _FP4_EBITS, _FP4_MBITS
    )
    data_lp = pack_uint4(unpacked).view(torch.float4_e2m1fn_x2)
    final_shape = (*orig_shape[:-1], orig_shape[-1] // 2)
    data_lp = data_lp.reshape(final_shape)
    scale_e8m0 = (group_exp + _F32_EXP_BIAS).to(torch.uint8)
    return scale_e8m0.squeeze(-1).view(torch.float8_e8m0fnu), data_lp


def _blocked_scale_offsets(flat_idx: Tensor, n_col_blocks: int) -> Tensor:
    elems_per_row_block = 512 * n_col_blocks
    second_dim = (flat_idx % elems_per_row_block) // (128 * n_col_blocks)
    third_dim = (flat_idx % (128 * n_col_blocks)) // (4 * n_col_blocks)
    fourth_dim = (flat_idx % (4 * n_col_blocks)) // 4
    return (
        (flat_idx // elems_per_row_block) * elems_per_row_block
        + fourth_dim * 512
        + third_dim * 16
        + second_dim * 4
        + flat_idx % 4
    )


def swizzle_scales_to_blocked(
    scales_2d: Tensor, target_shape: torch.Size
) -> torch.Tensor:
    """Swizzle a 2-D scale matrix using the MSLK blocked layout."""
    if scales_2d.dim() != 2:
        raise AssertionError(
            f"Expected 2-D scales_2d, got shape {tuple(scales_2d.shape)}"
        )
    M, num_cols = scales_2d.shape
    n_row_blocks = _ceil_div(M, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    rows_idx = torch.arange(M, device=scales_2d.device).view(-1, 1).expand(-1, num_cols)
    cols_idx = torch.arange(num_cols, device=scales_2d.device).view(1, -1).expand(M, -1)
    flat_idx = rows_idx * (n_col_blocks * 4) + cols_idx

    out = torch.zeros(
        padded_rows * padded_cols,
        device=scales_2d.device,
        dtype=scales_2d.dtype,
    )
    out[_blocked_scale_offsets(flat_idx, n_col_blocks).flatten()] = scales_2d.flatten()
    return out.view(target_shape)


def _unblock_mx4_scales(
    scales_flat: torch.Tensor, M: int, num_scale_cols: int
) -> torch.Tensor:
    """Reverse the blocked layout used by MX4 kernels (columns padded to a multiple
    of 4).

    Gathers the valid ``[M, num_scale_cols]`` region back into a 2-D tensor. This
    is a byte-layout de-swizzle (scale bytes), NOT value dequantization.
    """
    n_col_blocks = _ceil_div(num_scale_cols, 4)
    rows = torch.arange(M, device=scales_flat.device).unsqueeze(1)
    cols = torch.arange(num_scale_cols, device=scales_flat.device).unsqueeze(0)
    flat_idx = rows * (n_col_blocks * 4) + cols
    offsets = _blocked_scale_offsets(flat_idx, n_col_blocks)
    return scales_flat[offsets.long()]


# ============================================================
# Public API
# ============================================================


def torch_quantize_mx4_ref(
    x: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch MX4 quantize reference. **RCEIL rounding only.**

    Returns ``(xq_uint8, scales_uint8)`` with unswizzled 2-D scales.
    """
    if x.dtype not in (torch.bfloat16, torch.float):
        # fp16 -> f32 is lossless and matches the kernel (tl.load(...).to(f32)).
        x_for_quant = x.to(torch.float32)
    else:
        x_for_quant = x.contiguous()

    scale_e8m0, data_lp = _to_mxfp4(x_for_quant, group_size)

    return data_lp.view(torch.uint8), scale_e8m0.view(torch.uint8)
