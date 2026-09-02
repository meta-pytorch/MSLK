# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 groupwise-scaled grouped GEMM via FlyDSL.

Registers two ops, both backed by the same kernel:

* ``mslk::f8f8bf16_groupwise_grouped`` -- the ROCm implementation of the plain
  op, taking row-major ``[G, N, K]`` weights.
* ``mslk::f8f8bf16_groupwise_grouped_preshuffle`` -- a sibling that consumes
  weights already in the MFMA B-preshuffle layout (see
  ``mslk.quantize.shuffle.preshuffle_b_mfma``). Callers shuffle once at load
  time; the op does no shuffling.

Tensor contract:
  XQ      : [TotalM, K]             FP8  -- all groups concatenated along M
  WQ      : [G, N, K]               FP8  -- per-group weights, MFMA-preshuffled
                                            for the preshuffle op
  x_scale :                         FP32 -- per-token per-128K scales, in the
                                            per-group block layout produced by
                                            quantize_fp8_group(m_sizes=...)
  w_scale : [G, K//128, N//128]     FP32 -- per-group per-block scales
  M_sizes : [G]                     int64 -- rows per group (sum to TotalM)
  Output  : [TotalM, N]             BF16
"""

import torch
from mslk.flydsl.common import require_flydsl
from mslk.gemm.flydsl import grouped_dispatch
from mslk.utils.device import supports_float8_fnuz

_OP_NAME = "mslk::f8f8bf16_groupwise_grouped_preshuffle"


def _f8f8bf16_groupwise_grouped_preshuffle_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    TotalM = XQ.shape[0]
    N = WQ.shape[1]
    return XQ.new_empty((TotalM, N), dtype=torch.bfloat16)


def _dispatch_grouped_gemm(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
    *,
    b_preshuffled: bool,
) -> torch.Tensor:
    """Shared dispatch for both grouped ops. WQ is already in the layout the
    variant expects (MFMA-preshuffled if b_preshuffled else plain [G,N,K]).

    Tile selection lives in grouped_dispatch. Registration does not probe for
    FlyDSL, so this is the first point at which it is required.

    """
    require_flydsl()
    assert XQ.ndim == 2, f"XQ must be [TotalM, K], got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be [G, N, K], got {WQ.shape}"
    assert M_sizes.ndim == 1, f"M_sizes must be [G], got {M_sizes.shape}"
    TotalM, K = XQ.shape
    G, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    assert M_sizes.shape[0] == G, f"M_sizes length {M_sizes.shape[0]} must equal G={G}"
    # The MFMA instructions read the operands in the arch's native FP8 format, and
    # the kernel passes them through as raw bytes, so an fnuz/OCP mismatch would
    # be applied with the wrong exponent bias rather than rejected.
    expected_fp8 = (
        torch.float8_e4m3fnuz if supports_float8_fnuz() else torch.float8_e4m3fn
    )
    assert XQ.dtype == expected_fp8, f"XQ must be {expected_fp8}, got {XQ.dtype}"
    assert WQ.dtype == expected_fp8, f"WQ must be {expected_fp8}, got {WQ.dtype}"
    assert M_sizes.dtype == torch.int64, f"M_sizes must be int64, got {M_sizes.dtype}"

    return grouped_dispatch.dispatch(
        XQ,
        WQ,
        x_scale,
        w_scale,
        M_sizes,
        b_preshuffled=b_preshuffled,
        blockscale=True,
    )


def matmul_f8f8bf16_groupwise_grouped_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """Preshuffled-B grouped groupwise FP8 GEMM (WQ already MFMA-preshuffled)."""
    return _dispatch_grouped_gemm(XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=True)


def matmul_f8f8bf16_groupwise_grouped(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """Plain (non-preshuffled) grouped groupwise FP8 GEMM via FlyDSL.

    Same contract as the preshuffle sibling, but WQ is plain row-major
    ``[G, N, K]``. Uses the shared kernel with ``b_preshuffled=False``, which
    stages B through LDS instead of loading it straight to registers.
    """
    return _dispatch_grouped_gemm(
        XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=False
    )


# This module registers nothing. Both mslk::f8f8bf16_groupwise_grouped and its
# _preshuffle sibling are registered in mslk/gemm/__init__.py, whose impls import
# this module on the first call. Keeping registration out of here is what lets
# //mslk:gemm_ops avoid depending on //mslk/mslk/gemm:flydsl_ops, and so keeps the
# FlyDSL wheel out of every binary that merely imports mslk.gemm. Shape inference
# for the preshuffle op lives in mslk/gemm/_meta.py for the same reason.
