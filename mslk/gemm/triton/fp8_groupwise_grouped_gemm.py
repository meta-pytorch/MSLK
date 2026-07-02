# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
FP8 (E4M3) x FP8 (E4M3) groupwise-scaled grouped GEMM for ROCm/AMD.

This is the ROCm Triton counterpart of the NVIDIA CUTLASS implementation of
f8f8bf16_groupwise_grouped.

Tensor layout:
  XQ      : [TotalM, K]         FP8 -- all groups concatenated along M
  WQ      : [G, N, K]           FP8 -- one weight matrix per group
  x_scale : [TotalM, K//128]    float32 -- row-major activation scales (matches CUTLASS layout)
  w_scale : [G, K//128, N//128] float32 -- per-group, per-K-group, per-N-group weight scales
  M_sizes : [G]                 int64   -- number of rows per group
  Output  : [TotalM, N]         bfloat16

FP8 dtype (hardware-native formats):
  gfx942 (MI300X, CDNA3) -- tl.float8e4b8  (torch.float8_e4m3fnuz, bias=8, AMD fnuz)
  gfx950 (MI350X, CDNA4) -- tl.float8e4nv  (torch.float8_e4m3fn,   bias=7, OCP)

The kernel infers the FP8 pointer type from the input tensor dtype, so it
automatically uses the correct hardware format as long as the caller supplies
tensors produced by quantize_fp8_group / quantize_fp8_block.
"""

from typing import Dict, List, Optional, Tuple

import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton import Config  # @manual

# Scale group size in K: must match the quantize_fp8_group/block group_size.
_SCALE_GROUP_K: int = 128

# Module-level cache for CPU-side precomputed values (M_starts, max_M).
# Keyed by (M_sizes.data_ptr(), G, N, K) so the cache is invalidated if
# M_sizes changes.  Populated on the first (non-capturing) call; reused
# inside CUDA graph capture without device-to-host transfers.
_grouped_precomp_cache: Dict[Tuple, Tuple] = {}


# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------


def _get_configs() -> List[Config]:
    """
    Tile configs for the grouped groupwise FP8 GEMM kernel.

    BLOCK_K is fixed to 128 to match the scale group size.
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
    """Drop configs whose tile exceeds the problem in N."""
    N = named_args["N"]
    pruned = [cfg for cfg in configs if cfg.kwargs["BLOCK_N"] <= N]
    return pruned if pruned else configs


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_configs(),
    key=["G", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _f8f8bf16_groupwise_grouped_kernel(
    XQ_ptr,  # [TotalM, K]           FP8 activations (all groups)
    WQ_ptr,  # [G, N, K]             FP8 weights (one matrix per group)
    XS_ptr,  # [TotalM, K//128]      float32 x_scale, row-major (CUTLASS layout)
    WS_ptr,  # [G, K//128, N//128]   float32 w_scale
    Out_ptr,  # [TotalM, N]           bfloat16 output
    M_sizes_ptr,  # [G]  int64 -- rows per group
    M_starts_ptr,  # [G]  int64 -- cumulative M offset per group
    G,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_xs_m,
    stride_xs_k,
    stride_ws_g,
    stride_ws_k,
    stride_ws_n,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # fixed to 128 = scale group size
) -> None:
    """
    Computes Out[M_start:M_start+M_g, N] = dequant(XQ[group]) @ dequant(WQ[g]).T

    Grid: (G, ceil(max_M / BLOCK_M), ceil(N / BLOCK_N))
    Programs whose pid_m tile falls outside the group's M are discarded early.
    """
    pid_g = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Load this group's row count and starting global row index.
    M_g = tl.load(M_sizes_ptr + pid_g)
    M_start = tl.load(M_starts_ptr + pid_g)

    # Discard tiles that exceed this group's M dimension.
    if pid_m * BLOCK_M >= M_g:
        return

    # Global row indices into XQ, x_scale, and Out.
    offs_m = M_start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # N-group index for w_scale [G, K//128, N//128].
    n_group_offs = offs_n // 128  # [BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_groups = tl.cdiv(K, BLOCK_K)  # = K // 128 when K % 128 == 0
    row_limit = M_start + M_g  # exclusive upper bound for this group

    for k_g in tl.range(0, num_k_groups):
        k_start = k_g * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load XQ tile [BLOCK_M, BLOCK_K] from the concatenated activation tensor.
        xq = tl.load(
            XQ_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < row_limit) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Load WQ tile [BLOCK_N, BLOCK_K] from group g's weight matrix.
        wq = tl.load(
            WQ_ptr
            + pid_g * stride_wg
            + offs_n[:, None] * stride_wn
            + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Dot product for this K group: [BLOCK_M, BLOCK_N] float32.
        tmp = tl.dot(xq, tl.trans(wq), out_dtype=tl.float32)

        local_m = offs_m - M_start  # [BLOCK_M] row index within this group
        x_sc = tl.load(
            XS_ptr + M_start * stride_xs_m + local_m * stride_xs_k + k_g * M_g,
            mask=offs_m < row_limit,
            other=1.0,
        )  # [BLOCK_M]

        # Load w_scale[g, k_g, offs_n // 128]: one float32 per N-group.
        w_sc = tl.load(
            WS_ptr
            + pid_g * stride_ws_g
            + k_g * stride_ws_k
            + n_group_offs * stride_ws_n,
            mask=offs_n < N,
            other=1.0,
        )  # [BLOCK_N]

        # Combined scale: [BLOCK_M, BLOCK_N].
        acc += tmp * x_sc[:, None] * w_sc[None, :]

    # Store output as bfloat16.
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < row_limit) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def matmul_f8f8bf16_groupwise_grouped(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FP8 (E4M3) grouped groupwise-scaled GEMM -> BFloat16.

    Args:
        XQ      : [TotalM, K]         FP8 -- all groups concatenated
        WQ      : [G, N, K]           FP8 -- one weight matrix per group
        x_scale : [TotalM, K//128]    float32 -- activation scales, row-major (matches CUTLASS)
        w_scale : [G, K//128, N//128] float32 -- weight scales per group
        M_sizes : [G]                 int64 -- rows per group (must sum to TotalM)
        output  : [TotalM, N]         optional pre-allocated bfloat16 output

    Returns:
        [TotalM, N] bfloat16 tensor.
    """
    assert XQ.ndim == 2, f"XQ must be 2D [TotalM, K], got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be 3D [G, N, K], got {WQ.shape}"
    assert M_sizes.ndim == 1, f"M_sizes must be 1D [G], got {M_sizes.shape}"

    TotalM, K = XQ.shape
    G, N, Kw = WQ.shape
    assert Kw == K, f"WQ K mismatch: WQ.shape={WQ.shape}, XQ K={K}"
    assert K % _SCALE_GROUP_K == 0, (
        f"K={K} must be a multiple of scale group size {_SCALE_GROUP_K}"
    )
    assert M_sizes.shape[0] == G, f"M_sizes length {M_sizes.shape[0]} must equal G={G}"

    if output is None:
        output = torch.empty((TotalM, N), dtype=torch.bfloat16, device=XQ.device)

    if TotalM == 0 or N == 0 or K == 0 or G == 0:
        return output

    # Precompute M_starts (cumulative row offsets) and max_M on CPU.
    # Results are cached so CUDA graph capture (which forbids device-to-host
    # transfers) can reuse the values computed on the first non-capturing call.
    cache_key = (M_sizes.data_ptr(), G, N, K)
    if cache_key not in _grouped_precomp_cache:
        m_sizes_cpu = M_sizes.cpu()
        m_starts_cpu = torch.zeros(G, dtype=torch.int64)
        if G > 1:
            m_starts_cpu[1:] = m_sizes_cpu[:-1].cumsum(0)
        max_M = int(m_sizes_cpu.max().item())
        _grouped_precomp_cache[cache_key] = (
            m_starts_cpu.to(XQ.device),
            max_M,
        )

    M_starts, max_M = _grouped_precomp_cache[cache_key]

    grid = lambda meta: (  # noqa: E731
        G,
        triton.cdiv(max_M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _f8f8bf16_groupwise_grouped_kernel[grid](
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        M_sizes,
        M_starts,
        G,
        N,
        K,
        XQ.stride(0),
        XQ.stride(1),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        x_scale.stride(0),  # stride_xs_m: [TotalM, K//128] row stride
        x_scale.stride(1),  # stride_xs_k: column stride (=1 for contiguous)
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        output.stride(0),
        output.stride(1),
    )
    return output


# ---------------------------------------------------------------------------
# Register as the ROCm implementation of mslk::f8f8bf16_groupwise_grouped
# ---------------------------------------------------------------------------
# The C++ schema is declared in gemm_ops.cpp (shared between CUDA and ROCm).
# On ROCm the CUDA C++ implementation is absent; this module registers the
# Triton kernel above as the dispatch target via torch.library.impl.

if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    if hasattr(torch.ops.mslk, "f8f8bf16_groupwise_grouped"):
        try:

            @torch.library.impl("mslk::f8f8bf16_groupwise_grouped", "CUDA")
            def _f8f8bf16_groupwise_grouped_rocm(
                XQ: torch.Tensor,
                WQ: torch.Tensor,
                x_scale: torch.Tensor,
                w_scale: torch.Tensor,
                M_sizes: torch.Tensor,
            ) -> torch.Tensor:
                return matmul_f8f8bf16_groupwise_grouped(
                    XQ, WQ, x_scale, w_scale, M_sizes
                )

        except RuntimeError:
            pass  # already registered (e.g. module imported more than once)
