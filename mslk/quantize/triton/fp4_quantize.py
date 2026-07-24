# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import torch
from mslk.quantize.triton.legacy import primitives as _primitives
from mslk.quantize.triton.legacy.fake_quantize import (  # noqa: F401
    triton_fake_quantize_nvfp4_per_tensor,
)
from mslk.quantize.triton.legacy.global_scale import calculate_group_max  # noqa: F401
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
    triton_quantize_nvfp4,
)
from mslk.quantize.triton.legacy.quantize_stacked import (  # noqa: F401
    _nvfp4_quantize_stacked_kernel,
    nvfp4_quantize_stacked,
    nvfp4_quantize_stacked_with_token_scale,
)
from mslk.quantize.triton.quantize_kernels.mx4 import quantize_mx4


def triton_quantize_mx4_unpack(
    input: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: RoundingMode | int = RoundingMode.ceil,
    stochastic_casting: bool = False,
    *,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 (E2M1 / E8M0, group size 32) quantize.

    Thin adapter over ``quantize_kernels.mx4.quantize_mx4``. Returns packed
    ``uint8 [..., K//2]`` FP4 data + E8M0 scales, preserving the input's leading
    dims. Only E2M1 (``ebits=2, mbits=1``) is supported. ``quantize_mx4`` handles
    the device split internally: CUDA/Blackwell emits the swizzled scale layout;
    ROCm emits the plain ``[M, K//32]`` layout the ROCm MXFP4 GEMMs consume (and
    rejects stochastic rounding). Stochastic rounding on CUDA is requested via
    ``rounding_mode=RoundingMode.stochastic`` or by ``stochastic_casting=True``.
    The flag is one-way: ``True`` forces stochastic rounding on regardless of
    ``rounding_mode``, but ``False`` does not turn it off (an explicit
    ``rounding_mode=RoundingMode.stochastic`` still applies). ``seed`` is
    forwarded for reproducible output. ``group_size`` is retained for the MX4-16
    (``group_size=16``) recipe; for the standard group-32 path prefer
    ``triton_quantize_mx4``.
    """
    if ebits != 2 or mbits != 1:
        raise NotImplementedError(
            "triton_quantize_mx4_unpack: only E2M1 (ebits=2, mbits=1) is supported, "
            f"got ebits={ebits}, mbits={mbits}."
        )
    effective_rounding_mode: RoundingMode | int = (
        RoundingMode.stochastic if stochastic_casting else rounding_mode
    )
    xq, scale = quantize_mx4(
        input,
        group_size=group_size,
        rounding_mode=effective_rounding_mode,
        seed=seed,
    )
    # Restore the output-shape contract on the packed data: collapse to 1-D for
    # 1-D input; a no-op for higher-rank input (quantize_mx4 already returns a
    # shape-equivalent view). ``scale`` is returned exactly as produced.
    xq = xq.reshape(list(input.shape[:-1]) + [-1])
    return xq, scale


def triton_quantize_mx4(
    input: torch.Tensor,
    *,
    rounding_mode: RoundingMode | int = RoundingMode.ceil,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to MXFP4 (OCP MX v1.0: E2M1 elements, E8M0 scales, group size 32).

    Preferred public MXFP4 quantize entrypoint. Input is ``[..., N]`` (1-D or
    higher) bf16/fp16 with ``N % 32 == 0``. Returns packed ``uint8 [..., N//2]``
    FP4 data + E8M0 scales -- CUDA/Blackwell: flattened swizzled ``int8``;
    ROCm/gfx950: plain ``[..., N//32]`` ``uint8`` -- delegating to
    ``quantize_kernels.mx4.quantize_mx4`` (the device split is handled there).

    Stochastic rounding (CUDA only) is requested via
    ``rounding_mode=RoundingMode.stochastic``; ``seed`` is optional and, when
    ``None``, is derived from the CUDA default generator so callers respect
    ``torch.manual_seed``.

    For the legacy signature (``ebits``/``mbits``/``group_size`` incl. MX4-16, or
    the ``stochastic_casting`` flag), use ``triton_quantize_mx4_unpack``.
    """
    return quantize_mx4(input, group_size=32, rounding_mode=rounding_mode, seed=seed)


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
    "triton_quantize_mx4",
    "triton_fake_quantize_nvfp4_per_tensor",
    "calculate_group_max",
    "nvfp4_quantize_stacked",
    "nvfp4_quantize_stacked_with_token_scale",
    "cal_global_scale_mx4_as_nvfp4",
    "RoundingMode",
    "get_mx4_exp_bias",
    "FP4_E2M1_MAX",
    "FP8_E4M3_MAX",
    "FP4_EBITS",
    "FP4_MBITS",
]
