# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Torch-based bitwise correctness oracle for the MSLK MX4 quantize copy.

Pure-Torch references that ``test_quantize_mx4.py`` / ``test_stacked_quantize_mx4.py``
bitwise-compare ``quantize_mx4`` / ``quantize_mx4_stacked`` against:

    xq, scales = quantize_mx4(...)
    ref_xq, ref_scales_2d = torch_quantize_mx4_ref(..., swizzle=False)
    assert torch.equal(xq.flatten(), ref_xq.flatten())
    ref_swz = swizzle_scales_to_blocked(ref_scales_2d, scales.shape, convention="mslk")
    assert torch.equal(scales.flatten(), ref_swz.flatten())

Self-contained (imports only ``torch``/``math``); MX4-only (no NVFP4 / SiLU / RMS /
dequant helpers).

Provenance (original upstreams; also vendored into PyTorch
``caffe2/torch/testing/_internal/common_quantized.py``):
- ``_to_mxfp`` <- TorchAO ``torchao/prototype/mx_formats/mx_tensor.py`` (RCEIL).
  (https://github.com/pytorch/ao/blob/v0.12.0/torchao/prototype/mx_formats/mx_tensor.py#L142;
  common_quantized.py:578-656).
- ``_f32_to_floatx_unpacked`` <- TorchAO ``torchao/prototype/custom_fp_utils.py``
  (https://github.com/pytorch/ao/blob/bc4f51da86956275da7db0da6e420c506df97820/torchao/prototype/custom_fp_utils.py#L27-L142;
  common_quantized.py:247-360).
- ``_pack_uint4`` <- common_quantized.py:555-561 (TorchAO custom_fp_utils lineage).
- ``_to_blocked`` <- transformer_nuggets
  (https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/mx/to_blocked.py;
  common_quantized.py:515-546).
- ``_round_log2_via_bits_ceil`` mirrors fp4_primitives/scale.py:91-115.
- ``_blocked_scale_offset_swizzle`` mirrors fp4_primitives/layout.py:22-58.
- ``_unblock_mx4_scales`` (+ ``_five_dim_offset_pt``): inverse scale-byte de-swizzle
  (matches ``blocked_scale_offset``), used by the scale-inspection tests.

Rounding-mode pinning: MX4 bitwise tests pin ``RoundingMode.ceil`` -- the only mode
``_to_mxfp`` (RCEIL) implements.
"""

import math

import torch
from torch import Tensor


# ============================================================
# Constants
# ============================================================

# FP4 (E2M1) format parameters.
FP4_EBITS: int = 2
FP4_MBITS: int = 1

# FP32 IEEE-754 parameters.
_EBITS_F32: int = 8
_MBITS_F32: int = 23


def _n_ones(n: int) -> int:
    return (1 << n) - 1


_F32_EXP_BIAS: int = _n_ones(_EBITS_F32 - 1)

# Bit-extract ceil(log2) constants (mirror the _round_log2_via_bits, CEIL_ROUND).
_FP32_EXP_MASK: int = 0x7F800000
_FP32_EXP_OFFSET: int = 23
_F32_CEIL_ROUND: int = (1 << 23) - 1


# ============================================================
# GPU gate
# ============================================================


def is_blackwell_or_newer() -> bool:
    """Return True if the current CUDA device is Blackwell (SM100) or newer.

    The MX4 quantize kernels exercised here dispatch to inline-PTX paths (e.g.
    ``cvt.satfinite.e2m1x2.f32``) that are only available on SM100+ devices.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


# ============================================================
# Ported helpers (TorchAO / transformer_nuggets upstreams; see module docstring).
# ============================================================


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _f32_to_floatx_unpacked(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Ported from common_quantized.py:247-360.

    Convert FP32 to sub-byte float with ``ebits``/``mbits``; returns uint8 codes.
    """
    if x.dtype != torch.float:
        raise AssertionError(f"Expected x.dtype to be torch.float, got {x.dtype}")
    if 1 + ebits + mbits > 8:
        raise AssertionError(
            f"Expected 1 + ebits + mbits <= 8, got {1 + ebits + mbits}"
        )

    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)
    magic_adder = _n_ones(_MBITS_F32 - mbits - 1)

    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (_F32_EXP_BIAS - exp_bias) + (_MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << _MBITS_F32

    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    x = x.view(torch.int32)
    sign = x & 0x80000000

    x = x ^ sign
    x = x.view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (_MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - _F32_EXP_BIAS) << _MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (_MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    sign_lp = sign >> (_MBITS_F32 + _EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _down_size(size):
    if size[-1] % 2 != 0:
        raise AssertionError(f"{size} last dim not divisible by two")
    return (*size[:-1], size[-1] // 2)


def _pack_uint4(uint8_data) -> torch.Tensor:
    """Ported from common_quantized.py:555-561."""
    shape = uint8_data.shape
    if shape[-1] % 2 != 0:
        raise AssertionError(
            f"Expected shape[-1] to be divisible by 2, got {shape[-1]}"
        )
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(_down_size(shape))


def _round_log2_via_bits_ceil(x: torch.Tensor) -> torch.Tensor:
    """Exact ``ceil(log2(x))`` for positive normal f32 via FP32 bit manipulation.

    Ports the ``_round_log2_via_bits(x, CEIL_ROUND)`` from
    ``fp4_primitives/scale.py:91-115`` (avoids the libm ``ceil(log2)`` ULP
    imprecision on exact powers of two).
    """
    if x.dtype != torch.float:
        raise AssertionError(f"Expected x.dtype to be torch.float, got {x.dtype}")
    bits = x.view(torch.int32) + _F32_CEIL_ROUND
    exp_bits = (bits & _FP32_EXP_MASK) >> _FP32_EXP_OFFSET
    return (exp_bits - _F32_EXP_BIAS).to(torch.float32)


def _to_blocked(input_matrix: Tensor) -> torch.Tensor:
    """Ported from common_quantized.py:515-546.

    Rearrange a 2-D matrix into the 32x16 blocked layout (PyTorch pad-first
    convention). Used only by the NVFP4/``pytorch`` swizzle branch; kept so
    ``swizzle_scales_to_blocked`` stays drop-in identical.
    """
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _blocked_scale_offset_swizzle(scales_2d: Tensor) -> torch.Tensor:
    """FBGEMM blocked-layout swizzle for a 2-D scale matrix (columns padded to a
    multiple of 4, matching the canonical NVIDIA layout / ``_to_blocked``). Mirrors
    ``blocked_scale_offset`` in ``fp4_primitives/layout.py``."""
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

    elems_per_row_block = 512 * n_col_blocks

    first_dim = flat_idx // elems_per_row_block
    second_dim = (flat_idx % elems_per_row_block) // (128 * n_col_blocks)
    third_dim = (flat_idx % (128 * n_col_blocks)) // (4 * n_col_blocks)
    fourth_dim = (flat_idx % (4 * n_col_blocks)) // 4
    fifth_dim = flat_idx % 4

    out_offsets = (
        first_dim * elems_per_row_block
        + fourth_dim * 512
        + third_dim * 16
        + second_dim * 4
        + fifth_dim
    )

    out = torch.zeros(
        padded_rows * padded_cols,
        device=scales_2d.device,
        dtype=scales_2d.dtype,
    )
    out[out_offsets.flatten()] = scales_2d.flatten()
    return out


def _to_mxfp(
    data_hp: torch.Tensor,
    block_size: int = 32,
    format: str = "mxfp8",
):
    """Ported from common_quantized.py:578-656.

    RCEIL-only MX quantization. Returns ``(scale_e8m0_biased, data_lp)``.

    For ``format="mxfp4"`` the encoding follows FBGEMM:
    ``stored_byte = (ceil(log2(max_abs)) - EBITS + 127).to(int8)`` with
    ``EBITS = FP4_EBITS = 2``.
    """
    if data_hp.dtype not in (torch.bfloat16, torch.float):
        raise AssertionError(f"{data_hp.dtype} is not supported yet")
    if data_hp.shape[-1] % block_size != 0:
        raise AssertionError(
            f"the last dimension of shape {data_hp.shape} must be divisible "
            f"by block_size {block_size}"
        )
    if not data_hp.is_contiguous():
        raise AssertionError("unsupported: data_hp must be contiguous")

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(
        *orig_shape[:-1], orig_shape[-1] // block_size, block_size
    )

    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)
    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    if format == "mxfp8":
        max_pos = torch.finfo(torch.float8_e4m3fn).max
        descale = max_abs / max_pos
        exponent = torch.where(
            torch.isnan(descale),
            0xFF,
            (
                torch.clamp(
                    torch.ceil(torch.log2(descale)),
                    min=-_F32_EXP_BIAS,
                    max=_F32_EXP_BIAS,
                )
                + _F32_EXP_BIAS
            ).to(torch.uint8),
        )
        descale_fp = torch.where(
            exponent == 0,
            1.0,
            torch.exp2(_F32_EXP_BIAS - exponent.to(torch.float32)),
        )
        data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
        scale_e8m0_biased = exponent
        data_lp = data_lp.to(torch.float8_e4m3fn)
        data_lp = data_lp.reshape(orig_shape)
    elif format == "mxfp4":
        BF16_MIN_NORMAL = 1.1754943508222875e-38
        safe_max_abs = torch.where(
            max_abs == 0,
            torch.tensor(BF16_MIN_NORMAL, dtype=torch.float32, device=max_abs.device),
            max_abs,
        )
        group_exp_unbiased = _round_log2_via_bits_ceil(safe_max_abs)
        group_exp_unbiased = group_exp_unbiased - float(FP4_EBITS)
        group_exp_unbiased = torch.clamp(group_exp_unbiased, min=-127.0, max=125.0)
        scale_float = torch.exp2(group_exp_unbiased.to(torch.float64)).to(torch.float32)
        data_lp = data_hp / scale_float
        scale_e8m0_biased = (group_exp_unbiased + _F32_EXP_BIAS).to(torch.uint8)
        data_lp_unpacked = _f32_to_floatx_unpacked(
            data_lp.contiguous(), FP4_EBITS, FP4_MBITS
        )
        data_lp = _pack_uint4(data_lp_unpacked).view(torch.float4_e2m1fn_x2)
        final_shape = list(orig_shape)
        final_shape[-1] //= 2
        data_lp = data_lp.reshape(final_shape)
    else:
        raise AssertionError(f"unsupported format: {format}")

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)
    return scale_e8m0_biased, data_lp


# ============================================================
# Scale-byte de-swizzle (inverse layout transform — NOT dequantization).
# Inverse scale-byte de-swizzle (matches blocked_scale_offset).
# ============================================================


def _five_dim_offset_pt(flat_idx, n_col_blocks):
    """PyTorch vectorized 5-dim blocked layout offset (matches Triton
    ``blocked_scale_offset``). Reverses the sequential-indexed blocked layout."""
    epb = 512 * n_col_blocks
    first = flat_idx // epb
    second = (flat_idx % epb) // (128 * n_col_blocks)
    third = (flat_idx % (128 * n_col_blocks)) // (4 * n_col_blocks)
    fourth = (flat_idx % (4 * n_col_blocks)) // 4
    fifth = flat_idx % 4
    return first * epb + fourth * 512 + third * 16 + second * 4 + fifth


def _unblock_mx4_scales(
    scales_flat: torch.Tensor, M: int, num_scale_cols: int
) -> torch.Tensor:
    """Reverse the blocked layout used by MX4 kernels (columns padded to a multiple
    of 4).

    Gathers the valid ``[M, num_scale_cols]`` region back into a 2-D tensor. This
    is a byte-layout de-swizzle (scale bytes), NOT value dequantization.
    """
    n_col_blocks = math.ceil(num_scale_cols / 4)
    rows = torch.arange(M, device=scales_flat.device).unsqueeze(1)
    cols = torch.arange(num_scale_cols, device=scales_flat.device).unsqueeze(0)
    flat_idx = rows * (n_col_blocks * 4) + cols
    offsets = _five_dim_offset_pt(flat_idx, n_col_blocks)
    return scales_flat[offsets.long()]


# ============================================================
# Public API
# ============================================================


def torch_quantize_mx4_ref(
    x: torch.Tensor,
    *,
    group_size: int = 32,
    swizzle: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch MX4 quantize reference. **RCEIL rounding only.**

    Returns ``(xq_uint8, scales_uint8)``: ``xq_uint8`` shape
    ``[*x.shape[:-1], x.shape[-1] // 2]``; ``scales_uint8`` 1-D padded if
    ``swizzle=True`` else 2-D ``[*x.shape[:-1], x.shape[-1] // group_size]``.
    """
    if x.dtype not in (torch.bfloat16, torch.float):
        # fp16 -> f32 is lossless and matches the kernel (tl.load(...).to(f32)).
        x_for_quant = x.to(torch.float32)
    else:
        x_for_quant = x.contiguous()

    scale_e8m0, data_lp = _to_mxfp(x_for_quant, block_size=group_size, format="mxfp4")

    xq_uint8 = data_lp.view(torch.uint8)
    scales_uint8 = scale_e8m0.view(torch.uint8)

    if swizzle:
        if scales_uint8.dim() > 2:
            scales_uint8 = scales_uint8.reshape(-1, scales_uint8.shape[-1])
        scales_uint8 = _blocked_scale_offset_swizzle(scales_uint8)

    return xq_uint8, scales_uint8


def swizzle_scales_to_blocked(
    scales_unswizzled: torch.Tensor,
    target_shape: torch.Size,
    *,
    convention: str = "auto",
) -> torch.Tensor:
    """Apply the appropriate blocked swizzle to match the swizzled scale layout.

    MX4 callers (E8M0 scales) must pass ``convention="mslk"`` when operating on
    uint8 views (the float8_e8m0fnu dtype hint is lost after a uint8 view-cast).
    """
    if convention == "auto":
        if scales_unswizzled.dtype == torch.float8_e8m0fnu:
            convention = "mslk"
        else:
            convention = "pytorch"
    if convention not in ("mslk", "pytorch"):
        raise ValueError(
            f"convention must be 'auto', 'mslk', or 'pytorch'; got {convention!r}"
        )
    if scales_unswizzled.dtype != torch.uint8:
        scales_unswizzled = scales_unswizzled.view(torch.uint8)
    if scales_unswizzled.dim() > 2:
        scales_unswizzled = scales_unswizzled.reshape(-1, scales_unswizzled.shape[-1])
    if convention == "mslk":
        return _blocked_scale_offset_swizzle(scales_unswizzled).view(target_shape)
    return _to_blocked(scales_unswizzled).view(target_shape)
