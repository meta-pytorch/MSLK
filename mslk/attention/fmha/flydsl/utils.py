# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared low-level FlyDSL helpers for attention kernels.

Thin re-export of the arch-generic CDNA primitives in
mslk.flydsl.kernels.common.kernel_intrinsics; kept as a stable import site for the
pa_decode_* kernels.
"""

from mslk.flydsl.kernels.common.kernel_intrinsics import (  # noqa: F401
    dpp_xor_f32,
    exp2_f32,
    exp_f32,
    extract_global_ptr,
    global_load_f16x2,
    global_load_f32,
    global_load_i64x2,
    maxnumf,
    mfma_f32_16x16x16_bf16,
    mfma_f32_16x16x16_f16,
    mfma_f32_16x16x4_f32,
    rcp_f32,
    select_f32,
    smem_bytes,
    SMEM_BYTES_GFX942,
    SMEM_BYTES_GFX950,
    WARP_SIZE,
    wave_reduce_max_f32,
    wave_reduce_sum_f32,
)
from mslk.flydsl.kernels.common.kernels_common import get_warp_size  # noqa: F401
