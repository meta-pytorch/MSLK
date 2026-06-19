# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from mslk.quantize.triton.legacy import primitives as _primitives
from mslk.quantize.triton.legacy.fake_quantize import (  # noqa: F401
    triton_fake_quantize_nvfp4_per_tensor,
)
from mslk.quantize.triton.legacy.fused_rms import (  # noqa: F401
    triton_nvfp4_quant_stacked_rms,
    triton_rms_quantize_mx4_unpack,
    triton_scale_nvfp4_quant_rms,
)
from mslk.quantize.triton.legacy.fused_silu import (  # noqa: F401
    triton_nvfp4_quant_stacked_silu,
    triton_scale_nvfp4_quant_silu,
    triton_silu_quantize_mx4_unpack,
)
from mslk.quantize.triton.legacy.global_scale import calculate_group_max  # noqa: F401
from mslk.quantize.triton.legacy.naive import (  # noqa: F401
    get_nvfp4_global_scales_naive,
    quantize_nvfp4_naive,
)
from mslk.quantize.triton.legacy.primitives import (  # noqa: F401
    cal_global_scale_mx4_as_nvfp4,
    FP4_E2M1_MAX,
    FP4_EBITS,
    FP4_MBITS,
    FP8_E4M3_MAX,
    get_mx4_exp_bias,
    RoundingMode,
)
from mslk.quantize.triton.legacy.quantize import (  # noqa: F401
    _kernel_quantize_mx4_unpack,
    triton_quantize_mx4_unpack,
    triton_quantize_nvfp4,
)
from mslk.quantize.triton.legacy.quantize_stacked import (  # noqa: F401
    _nvfp4_quantize_stacked_kernel,
    mega_fp4_quantize_kernel,
    nvfp4_quantize_stacked,
    nvfp4_quantize_stacked_with_token_scale,
)
from mslk.quantize.triton.legacy.unpack import (  # noqa: F401
    mega_fp4_pack,
    mega_fp4_unpack,
)

# Private symbol aliases (same-object re-exports for internal consumers)
_compute_exp = _primitives._compute_exp  # noqa: F401
_fp32_to_e8m0 = _primitives._fp32_to_e8m0  # noqa: F401
_from_blocked = _primitives._from_blocked  # noqa: F401
_to_blocked = _primitives._to_blocked  # noqa: F401
_e2m1_round_to_even = _primitives._e2m1_round_to_even  # noqa: F401
_floor_log2 = _primitives._floor_log2  # noqa: F401
unsigned_fp32_to_e8m0 = _primitives.unsigned_fp32_to_e8m0  # noqa: F401
nvfp4_scale_swizzle = _primitives.nvfp4_scale_swizzle  # noqa: F401
convert_fp32_to_fp4_packed = _primitives.convert_fp32_to_fp4_packed  # noqa: F401

__all__ = [
    "triton_quantize_nvfp4",
    "triton_quantize_mx4_unpack",
    "triton_fake_quantize_nvfp4_per_tensor",
    "calculate_group_max",
    "nvfp4_quantize_stacked",
    "nvfp4_quantize_stacked_with_token_scale",
    "mega_fp4_quantize_kernel",
    "mega_fp4_pack",
    "mega_fp4_unpack",
    "triton_scale_nvfp4_quant_silu",
    "triton_scale_nvfp4_quant_rms",
    "triton_silu_quantize_mx4_unpack",
    "triton_rms_quantize_mx4_unpack",
    "triton_nvfp4_quant_stacked_silu",
    "triton_nvfp4_quant_stacked_rms",
    "get_nvfp4_global_scales_naive",
    "quantize_nvfp4_naive",
    "cal_global_scale_mx4_as_nvfp4",
    "RoundingMode",
    "get_mx4_exp_bias",
    "FP4_E2M1_MAX",
    "FP8_E4M3_MAX",
    "FP4_EBITS",
    "FP4_MBITS",
]
