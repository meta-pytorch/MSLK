# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
FP8 (E4M3) x FP8 (E4M3) groupwise-scaled GEMM for ROCm/AMD.

This is the ROCm Triton counterpart of the NVIDIA CUTLASS implementation in
csrc/gemm/cutlass/f8f8bf16_groupwise.cu.

Scaling scheme ("groupwise"):
  x_scale : [K // 128, M]         float32 -- one scale per (K-group-of-128, row)
  w_scale : [K // 128, N // 128]  float32 -- one scale per (K-group, N-group-of-128)

Both scales use k_major=False layout (K dimension is outermost), matching the
output of quantize_fp8_group(x, k_major=False) and
quantize_fp8_block(w, block_m=128, block_k=128, k_major=False).

FP8 dtype (hardware-native formats):
  gfx942 (MI300X, CDNA3) -- tl.float8e4b8  (torch.float8_e4m3fnuz, bias=8, AMD fnuz)
  gfx950 (MI350X, CDNA4) -- tl.float8e4nv  (torch.float8_e4m3fn,   bias=7, OCP)
  CUDA                   -- tl.float8e4nv  (torch.float8_e4m3fn,   bias=7, OCP)

The Triton kernel infers the FP8 pointer type from the input tensor dtype, so it
automatically uses the correct hardware format as long as the caller supplies
tensors produced by quantize_fp8_group / quantize_fp8_block, which call
get_fp8_constants() to select the right dtype per device.
"""

from typing import List, Optional

import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton import Config  # @manual

# Scale group size in K: must match the quantize_fp8_group/block group_size.
_SCALE_GROUP_K: int = 128


# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------


def _get_configs() -> List[Config]:
    """
    Tile configs for the groupwise FP8 GEMM kernel.

    BLOCK_K is fixed to 128 to match the scale group size: each K tile maps
    to exactly one scale group, avoiding partial-scale complexity.
    BLOCK_M and BLOCK_N are autotuned.
    """
    configs = []
    for bm in [16, 32, 64, 128, 256]:
        for bn in [16, 32, 64, 128, 256]:
            for nw in [4, 8]:
                for ns in [2, 3, 4]:
                    configs.append(
                        Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


def _prune_configs(configs, named_args, **kwargs):
    """Drop configs whose tile exceeds the problem in M or N."""
    M = named_args["M"]
    N = named_args["N"]
    pruned = []
    for cfg in configs:
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        if bm > M or bn > N:
            continue
        if M <= 16 and bm > 16:
            continue
        if M <= 32 and bm > 32:
            continue
        pruned.append(cfg)
    return pruned if pruned else configs


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _f8f8bf16_groupwise_kernel(
    XQ_ptr,  # [M, K]           FP8 activations
    WQ_ptr,  # [N, K]           FP8 weights (W already transposed)
    XS_ptr,  # [K//128, M]      float32 x_scale
    WS_ptr,  # [K//128, N//128] float32 w_scale
    Out_ptr,  # [M, N]           bfloat16 output
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_xs_k,
    stride_xs_m,  # x_scale strides: [K//128, M]
    stride_ws_k,
    stride_ws_n,  # w_scale strides: [K//128, N//128]
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # fixed to 128 = scale group size
) -> None:
    """
    Computes Out[M, N] = dequant(XQ) @ dequant(WQ).T in bfloat16.

    Scale application: for each K-group g of 128 elements:
      combined_scale[m, n] = x_scale[g, m] * w_scale[g, n // 128]
      contribution = dot(XQ[:, g*128:(g+1)*128], WQ[:, g*128:(g+1)*128].T)
                   * combined_scale
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Number of scale groups in K and N.
    # N-group index for each column in this tile (for w_scale indexing).
    # w_scale has shape [K//128, N//128]; all columns in [pid_n*BLOCK_N, (pid_n+1)*BLOCK_N)
    # map to N-group indices offs_n // 128.  We compute the w_scale pointer offsets
    # per column using the N-group stride.
    n_group_offs = offs_n // 128  # [BLOCK_N] int, each in [0, N//128)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_groups = tl.cdiv(K, BLOCK_K)  # = K // 128 when K % 128 == 0

    for k_g in tl.range(0, num_k_groups):
        k_start = k_g * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load XQ tile [BLOCK_M, BLOCK_K] in FP8.
        # Triton infers the element type from the pointer dtype (FP8), so no
        # explicit cast is needed.
        xq = tl.load(
            XQ_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Load WQ tile [BLOCK_N, BLOCK_K] in FP8; transposed inside dot.
        wq = tl.load(
            WQ_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Dot product for this K group: [BLOCK_M, BLOCK_N] float32.
        tmp = tl.dot(xq, tl.trans(wq), out_dtype=tl.float32)

        # Load x_scale[k_g, offs_m]: one float32 per row in this tile.
        x_sc = tl.load(
            XS_ptr + k_g * stride_xs_k + offs_m * stride_xs_m,
            mask=offs_m < M,
            other=1.0,
        )  # [BLOCK_M]

        # Load w_scale[k_g, offs_n // 128]: one float32 per N-group-of-128.
        w_sc = tl.load(
            WS_ptr + k_g * stride_ws_k + n_group_offs * stride_ws_n,
            mask=offs_n < N,
            other=1.0,
        )  # [BLOCK_N]

        # Combined scale: [BLOCK_M, BLOCK_N].
        combined = x_sc[:, None] * w_sc[None, :]
        acc += tmp * combined

    # Store output as bfloat16.
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def matmul_f8f8bf16_groupwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FP8 (E4M3) groupwise-scaled GEMM -> BFloat16.

    Matches the CUDA CUTLASS API contract for f8f8bf16_groupwise.

    Args:
        XQ      : [M, K]         torch.float8_e4m3fnuz (ROCm) or float8_e4m3fn (CUDA)
        WQ      : [N, K]         same FP8 dtype, W already transposed by caller
        x_scale : [K//128, M]   float32 -- per-row, per-K-group activation scales
        w_scale : [K//128, N//128] float32 -- per-K-group, per-N-group weight scales
        output  : [M, N]         optional pre-allocated bfloat16 output

    Returns:
        [M, N] bfloat16 tensor.
    """
    assert XQ.ndim == 2, f"XQ must be 2D, got shape {XQ.shape}"
    assert WQ.ndim == 2, f"WQ must be 2D [N, K], got shape {WQ.shape}"

    M, K = XQ.shape
    N = WQ.shape[0]
    assert WQ.shape[1] == K, f"WQ K dim mismatch: WQ.shape={WQ.shape}, XQ K={K}"
    assert K % _SCALE_GROUP_K == 0, (
        f"K={K} must be a multiple of scale group size {_SCALE_GROUP_K}"
    )

    if output is None:
        output = torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)

    if M == 0 or N == 0 or K == 0:
        return output

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _f8f8bf16_groupwise_kernel[grid](
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        M,
        N,
        K,
        XQ.stride(0),
        XQ.stride(1),
        WQ.stride(0),
        WQ.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        output.stride(0),
        output.stride(1),
    )
    return output


# ---------------------------------------------------------------------------
# Register as the ROCm implementation of mslk::f8f8bf16_groupwise
# ---------------------------------------------------------------------------
# The C++ schema is declared in gemm_ops.cpp (shared between CUDA and ROCm).
# On ROCm the CUDA C++ implementation is absent; this module registers the
# Triton kernel above as the dispatch target via torch.library.impl.

if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    if hasattr(torch.ops.mslk, "f8f8bf16_groupwise"):
        try:

            @torch.library.impl("mslk::f8f8bf16_groupwise", "CUDA")
            def _f8f8bf16_groupwise_rocm(
                XQ: torch.Tensor,
                WQ: torch.Tensor,
                x_scale: torch.Tensor,
                w_scale: torch.Tensor,
            ) -> torch.Tensor:
                return matmul_f8f8bf16_groupwise(XQ, WQ, x_scale, w_scale)

        except RuntimeError:
            pass  # already registered (e.g. module imported more than once)
