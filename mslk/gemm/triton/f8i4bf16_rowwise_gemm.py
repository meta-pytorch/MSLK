# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
FP8 x INT4 weight-only GEMM for ROCm/AMD GPUs, producing BF16 output.

Thin wrapper around the unified _bf16i4_rowwise_kernel (int4_gemm.py).
The kernel is shared with the BF16xINT4 path; the only differences are:
  1. Activations are FP8 — byte-split and reinterpreted as FP8 TensorWrappers
     here, then upcast to bfloat16 inside the kernel via .to(tl.bfloat16).
  2. HAS_X_SCALE=True — the kernel multiplies the accumulator by a per-row
     activation dequant scale before storing to workspace.

Op signature (mslk::f8i4bf16_rowwise):
  XQ          : [M, K]           FP8 activations (float8_e4m3fnuz on AMD)
  WQ          : [N, K//2]        int8 packed INT4 (lo nibble = even K, hi = odd K)
  x_scale     : [M]              float32 per-row activation dequant scale
  w_scale     : [num_groups, N]  float32/bf16/fp16 per-group weight scale
  w_zp        : [num_groups, N]  same dtype as w_scale, per-group zero point
  output      : [M, N]           bfloat16
"""

import torch
import triton  # @manual
from mslk.gemm.triton.int4_gemm import (
    _bf16i4_rowwise_kernel,
    _bf16i4_splitk_reduce,
    _MAX_SPLIT_K,
    _TL_CAT_HAS_DIM,
)
from mslk.utils.triton.fp8_utils import get_fp8_constants, reinterpret_fp8_type


def matmul_f8i4bf16_rowwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    w_zp: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 activation x INT4 weight GEMM with per-row x_scale and per-group w_scale/w_zp.

    Args:
        XQ      : [M, K]           FP8 activations (float8_e4m3fnuz on AMD)
        WQ      : [N, K//2]        int8 packed INT4 (lo nibble=even K, hi=odd K)
        x_scale : [M]              float32 per-row activation dequant scale
        w_scale : [num_groups, N]  float32/bf16/fp16 per-group weight scale
        w_zp    : [num_groups, N]  same dtype, per-group weight zero point

    Returns:
        Y : [M, N]  bfloat16
    """
    assert XQ.ndim == 2, f"XQ must be 2D [M, K], got shape {XQ.shape}"
    assert WQ.ndim == 2, f"WQ must be 2D [N, K//2], got shape {WQ.shape}"
    assert x_scale.ndim == 1, f"x_scale must be 1D [M], got shape {x_scale.shape}"
    assert w_scale.ndim == 2, (
        f"w_scale must be 2D [num_groups, N], got shape {w_scale.shape}"
    )
    assert w_zp.ndim == 2, f"w_zp must be 2D [num_groups, N], got shape {w_zp.shape}"
    M, K = XQ.shape
    N = WQ.shape[0]
    K2 = K // 2
    num_groups = w_scale.shape[0]
    assert K % 2 == 0, f"K={K} must be even for packed INT4 weights"

    assert WQ.shape == (N, K2), f"WQ must be [N, K//2], got {WQ.shape}"
    assert x_scale.shape == (M,), f"x_scale must be [M]={M}, got {x_scale.shape}"
    assert w_scale.shape == (num_groups, N), (
        f"w_scale must be [num_groups, N]={num_groups, N}, got {w_scale.shape}"
    )
    assert w_zp.shape == (num_groups, N), (
        f"w_zp must be [num_groups, N]={num_groups, N}, got {w_zp.shape}"
    )
    assert WQ.dtype == torch.int8, f"WQ must be int8, got {WQ.dtype}"
    assert XQ.is_contiguous(), "XQ must be contiguous"
    assert WQ.is_contiguous(), "WQ must be contiguous"
    assert all(tensor.device == XQ.device for tensor in (WQ, x_scale, w_scale, w_zp)), (
        "all inputs must be on the same device"
    )

    pt_fp8_dtype, tl_fp8_dtype, _, _ = get_fp8_constants()
    assert XQ.dtype == pt_fp8_dtype, f"XQ must be {pt_fp8_dtype}, got {XQ.dtype}"

    if M == 0 or N == 0 or K == 0:
        return torch.zeros((M, N), dtype=torch.bfloat16, device=XQ.device)

    assert num_groups > 0, "w_scale must contain at least one quantization group"
    assert K % num_groups == 0, f"K={K} must be divisible by num_groups={num_groups}"
    group_size = K // num_groups
    assert group_size % 64 == 0, (
        f"group_size={group_size} must be divisible by 64 (2 * BLOCK_K_min=32)"
    )

    XQ_int8 = XQ.view(torch.int8)
    x_even_t = XQ_int8[:, 0::2].contiguous()
    x_odd_t = XQ_int8[:, 1::2].contiguous()
    x_even = reinterpret_fp8_type(x_even_t, tl_fp8_dtype)
    x_odd = reinterpret_fp8_type(x_odd_t, tl_fp8_dtype)

    w_scale = w_scale.to(torch.float32).contiguous()
    w_zp = w_zp.to(torch.float32).contiguous()
    x_scale = x_scale.to(torch.float32).contiguous()

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
            meta["SPLIT_K"],
        )

    workspace_splits = 1 if M >= 512 else _MAX_SPLIT_K
    workspace = torch.empty(
        (workspace_splits, M, N), dtype=torch.float32, device=XQ.device
    )

    _bf16i4_rowwise_kernel[grid](
        x_even,
        x_odd,
        WQ,
        workspace,
        w_scale,
        w_zp,
        x_scale,
        M,
        N,
        K2,
        group_size,
        x_even.stride(0),
        x_even.stride(1),
        WQ.stride(0),
        WQ.stride(1),
        M * N,
        N,
        1,
        w_scale.stride(0),
        w_scale.stride(1),
        FUSE_DOT=_TL_CAT_HAS_DIM,
        HAS_X_SCALE=True,
    )

    split_k = _bf16i4_rowwise_kernel.best_config.kwargs["SPLIT_K"]

    if split_k == 1:
        return workspace[0].to(torch.bfloat16)

    Y_bf16 = torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)
    reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    _bf16i4_splitk_reduce[reduce_grid](
        workspace,
        Y_bf16,
        M,
        N,
        SPLIT_K=split_k,
        BLOCK_M=32,  # pyre-ignore[6]
        BLOCK_N=32,  # pyre-ignore[6]
    )
    return Y_bf16


# ---------------------------------------------------------------------------
# Register as ROCm implementation of mslk::f8i4bf16_rowwise
# ---------------------------------------------------------------------------

if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    if hasattr(torch.ops.mslk, "f8i4bf16_rowwise"):

        @torch.library.impl("mslk::f8i4bf16_rowwise", "CUDA")
        def _f8i4bf16_rowwise_rocm(
            XQ: torch.Tensor,
            WQ: torch.Tensor,
            x_scale: torch.Tensor,
            w_scale: torch.Tensor,
            w_zp: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_f8i4bf16_rowwise(XQ, WQ, x_scale, w_scale, w_zp)
