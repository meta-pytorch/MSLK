# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
MXFP8 (E4M3) activation x MXFP8 (E4M3) weight GEMM for ROCm/AMD gfx950+.

This is the ROCm Triton counterpart of csrc/gemm/cutlass/mx8mx8bf16_grouped.cu.
It uses Triton's tl.dot_scaled, which lowers to the CDNA4 scaled-MFMA
instructions (V_MFMA_SCALE_F32_16X16X128_F8F6F4) available on gfx950
(MI350/MI355X).  It is NOT compatible with gfx942 (MI300X) or older CDNA
architectures, which lack hardware block-scaling support.

This mirrors mx8mx4_gemm.py exactly, except the weight operand is FP8 E4M3
(not packed MXFP4), so both operands take the "e4m3" path of tl.dot_scaled and
there is no nibble unpacking or K//2 byte addressing on the weight side.

Tensor shapes (matching the CUDA CUTLASS convention):
  Grouped 2D-3D (groups along M):
    XQ      : [total_M, K]    WQ : [G, K, N] (col-major)   out : [total_M, N]
  Grouped 2D-2D (groups along K):
    XQ      : [M, total_K]    WQ : [total_K, N]            out : [G, M, N]

Scale factor format:
  OCP MX uses one 8-bit (E8M0) exponent per block of 32 values along K, stored
  in the _to_blocked flat layout produced by fp4_quantize._to_blocked.
"""

from typing import List, Optional

import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton import Config  # @manual


# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------


def _get_configs() -> List[Config]:
    """
    Tile configs tuned for the gfx950 CU layout.

    BLOCK_K is fixed to 128: V_MFMA_SCALE_F32_16X16X128_F8F6F4 requires K=128
    FP8 elements per tile.
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
    """Drop tile configs larger than the problem in M or N."""
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


def _prune_configs_grouped(configs, named_args, **kwargs):
    """Adapter for grouped kernels: maps total_M -> M for _prune_configs."""
    named_args = dict(named_args)
    named_args["M"] = named_args.get("total_M", named_args.get("M", 1))
    return _prune_configs(configs, named_args, **kwargs)


# ---------------------------------------------------------------------------
# Grouped 2D-3D kernel (groups along M): XQ [total_M, K], WQ [G, K, N] col-major
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_configs(),
    key=["total_M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs_grouped},
)
@triton.jit
def _mx8mx8bf16_grouped_2d3d_kernel(
    XQ_ptr,  # [total_M, K]   float8_e4m3 bytes
    WQ_ptr,  # [G, K, N]      float8_e4m3 bytes (col-major W)
    XS_ptr,  # flat uint8  _to_blocked, groups concatenated along M
    WS_ptr,  # flat uint8  _to_blocked, groups stacked (all same N)
    Out_ptr,  # [total_M, N]  bfloat16
    m_sizes_ptr,  # [G]  int32
    m_starts_ptr,  # [G]  int32
    total_M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wg,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    NUM_SMS: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Persistent grouped MXFP8 x MXFP8 GEMM (2D-3D), CUDA-graph safe."""
    tidx = tl.program_id(0)

    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
    N_COL_BLOCKS: tl.constexpr = (K // 32 + 3) // 4

    n_scale_cols = tl.cdiv(K, 32)
    ws_n_row_blocks = tl.cdiv(N, 128)
    ws_group_stride = ws_n_row_blocks * N_COL_BLOCKS * 512

    iterated_tiles = 0
    xs_offset_acc = 0

    for g in tl.range(G):
        m_size = tl.load(m_sizes_ptr + g)
        m_start = tl.load(m_starts_ptr + g)

        xs_group_offset = xs_offset_acc
        ws_group_offset = g * ws_group_stride

        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles

        while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
            gidx = tidx - iterated_tiles
            tile_m = gidx % num_m_tiles
            tile_n = gidx // num_m_tiles

            local_m_start = tile_m * BLOCK_M
            abs_m_start = m_start + local_m_start
            group_end = m_start + m_size

            offs_m = abs_m_start + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            local_m = local_m_start + tl.arange(0, BLOCK_M)

            row_valid = offs_m[:, None] < group_end

            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for k_idx in tl.range(0, tl.cdiv(K, BLOCK_K)):
                k_start = k_idx * BLOCK_K
                offs_k = k_start + tl.arange(0, BLOCK_K)
                offs_ks = k_start // 32 + tl.arange(0, SCALE_BLOCK_K)

                xq = tl.load(
                    XQ_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=row_valid & (offs_k[None, :] < K),
                    other=0,
                )
                # WQ [G, K, N] col-major -> B operand tile [BLOCK_K, BLOCK_N].
                wq = tl.load(
                    WQ_ptr
                    + g * stride_wg
                    + offs_k[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0,
                )

                ks2d = offs_ks[None, :]
                lm2d = local_m[:, None]
                xs_flat = (
                    xs_group_offset
                    + (lm2d // 128 * N_COL_BLOCKS + ks2d // 4) * 512
                    + (lm2d % 32) * 16
                    + (lm2d % 128 // 32) * 4
                    + ks2d % 4
                )
                xs = tl.load(
                    XS_ptr + xs_flat,
                    mask=row_valid & (ks2d < n_scale_cols),
                    other=0,
                )

                n2d = offs_n[:, None]
                ws_flat = (
                    ws_group_offset
                    + (n2d // 128 * N_COL_BLOCKS + ks2d // 4) * 512
                    + (n2d % 32) * 16
                    + (n2d % 128 // 32) * 4
                    + ks2d % 4
                )
                ws = tl.load(
                    WS_ptr + ws_flat,
                    mask=(n2d < N) & (ks2d < n_scale_cols),
                    other=0,
                )

                acc = tl.dot_scaled(xq, xs, "e4m3", wq, ws, "e4m3", acc)

            tl.store(
                Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                acc.to(tl.bfloat16),
                mask=row_valid & (offs_n[None, :] < N),
            )

            tidx += NUM_SMS

        iterated_tiles += num_tiles
        xs_offset_acc += tl.cdiv(m_size, 128) * N_COL_BLOCKS * 512


# ---------------------------------------------------------------------------
# Grouped 2D-2D kernel (groups along K): XQ [M, total_K], WQ [total_K, N]
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_configs(),
    key=["M", "N"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _mx8mx8bf16_grouped_2d2d_kernel(
    XQ_ptr,  # [M, total_K]   float8_e4m3 bytes
    WQ_ptr,  # [total_K, N]   float8_e4m3 bytes (row-major: B operand directly)
    XS_ptr,  # flat uint8  per-group _to_blocked concatenated along K
    WS_ptr,  # flat uint8  per-group _to_blocked concatenated along K
    Out_ptr,  # [G, M, N]    bfloat16
    k_sizes_ptr,  # [G]  int32  K per group
    k_starts_ptr,  # [G]  int32  cumulative K start per group
    xs_bases_ptr,  # [G]  int64  element base offset into XS for group g
    ws_bases_ptr,  # [G]  int64  element base offset into WS for group g
    M,
    N,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_og,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Grouped MXFP8 x MXFP8 GEMM (2D-2D). Grid: (G, cdiv(M,BM), cdiv(N,BN))."""
    pid_g = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32

    Kg = tl.load(k_sizes_ptr + pid_g)
    k_start = tl.load(k_starts_ptr + pid_g)
    xs_base = tl.load(xs_bases_ptr + pid_g)
    ws_base = tl.load(ws_bases_ptr + pid_g)

    # Per-group scale geometry (runtime; Kg varies across groups).
    n_scale_cols = tl.cdiv(Kg, 32)
    n_col_blocks = tl.cdiv(n_scale_cols, 4)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_idx in tl.range(0, tl.cdiv(Kg, BLOCK_K)):
        k_local = k_idx * BLOCK_K
        offs_k = k_start + k_local + tl.arange(0, BLOCK_K)  # global K index
        offs_ks = k_local // 32 + tl.arange(0, SCALE_BLOCK_K)  # local scale col
        k_in_group = k_local + tl.arange(0, BLOCK_K)

        xq = tl.load(
            XQ_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_in_group[None, :] < Kg),
            other=0,
        )
        # WQ [total_K, N] row-major -> B operand tile [BLOCK_K, BLOCK_N].
        wq = tl.load(
            WQ_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(k_in_group[:, None] < Kg) & (offs_n[None, :] < N),
            other=0,
        )

        ks2d = offs_ks[None, :]
        m2d = offs_m[:, None]
        xs_flat = (
            xs_base
            + (m2d // 128 * n_col_blocks + ks2d // 4) * 512
            + (m2d % 32) * 16
            + (m2d % 128 // 32) * 4
            + ks2d % 4
        )
        xs = tl.load(
            XS_ptr + xs_flat,
            mask=(m2d < M) & (ks2d < n_scale_cols),
            other=0,
        )

        n2d = offs_n[:, None]
        ws_flat = (
            ws_base
            + (n2d // 128 * n_col_blocks + ks2d // 4) * 512
            + (n2d % 32) * 16
            + (n2d % 128 // 32) * 4
            + ks2d % 4
        )
        ws = tl.load(
            WS_ptr + ws_flat,
            mask=(n2d < N) & (ks2d < n_scale_cols),
            other=0,
        )

        acc = tl.dot_scaled(xq, xs, "e4m3", wq, ws, "e4m3", acc)

    tl.store(
        Out_ptr
        + pid_g * stride_og
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul_mx8mx8bf16_grouped(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    offsets: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    actual_num_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Grouped MXFP8 (E4M3) x MXFP8 (E4M3) GEMM -> BFloat16.

    Dispatches on WQ.dim(): 3D -> 2D-3D (groups along M), 2D -> 2D-2D (groups
    along K).  Matches the CUDA mx8mx8bf16_grouped_mm API contract.

    Args:
        XQ       : [total_M, K] (2D-3D) or [M, total_K] (2D-2D)  float8_e4m3fn
        WQ       : [G, K, N] (2D-3D, col-major) or [total_K, N] (2D-2D)
        x_scale  : flat uint8  _to_blocked scales (per-group concatenated)
        w_scale  : flat uint8  _to_blocked scales
        offsets  : [G]  int32  cumulative end indices (M for 2D-3D, K for 2D-2D)
        output   : optional pre-allocated bfloat16 output
        actual_num_tokens : ignored on ROCm (CUDA-only heuristic hint)
    """
    assert XQ.ndim == 2, f"XQ must be 2D, got {XQ.shape}"
    G = offsets.shape[0]
    offsets_i32 = offsets.to(dtype=torch.int32, device=XQ.device)

    xq_raw = XQ.view(torch.uint8)
    xs_raw = x_scale.view(torch.uint8).contiguous()
    ws_raw = w_scale.view(torch.uint8).contiguous()

    if WQ.ndim == 3:
        # 2D-2D would have WQ.ndim == 2; 3D is the 2D-3D (groups along M) case.
        total_M, K = XQ.shape
        _, Kw, N = WQ.shape
        assert Kw == K, f"WQ K mismatch: WQ={WQ.shape}, XQ K={K}"
        assert K % 32 == 0, f"K={K} must be a multiple of 32"
        wq_raw = WQ.view(torch.uint8)

        if output is None:
            output = torch.empty((total_M, N), dtype=torch.bfloat16, device=XQ.device)
        if total_M == 0 or N == 0 or K == 0:
            return output

        m_starts = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=XQ.device), offsets_i32[:-1]]
        )
        m_sizes = offsets_i32 - m_starts

        NUM_SMS = torch.cuda.get_device_properties(XQ.device).multi_processor_count

        _mx8mx8bf16_grouped_2d3d_kernel[(NUM_SMS,)](
            xq_raw,
            wq_raw,
            xs_raw,
            ws_raw,
            output,
            m_sizes,
            m_starts,
            total_M,
            N,
            K,
            xq_raw.stride(0),
            xq_raw.stride(1),
            wq_raw.stride(0),
            wq_raw.stride(1),
            wq_raw.stride(2),
            output.stride(0),
            output.stride(1),
            NUM_SMS=NUM_SMS,
            G=G,
        )
        return output

    # 2D-2D case: groups along K.
    M, total_K = XQ.shape
    Kw, N = WQ.shape
    assert Kw == total_K, f"WQ K mismatch: WQ={WQ.shape}, XQ total_K={total_K}"
    wq_raw = WQ.view(torch.uint8)

    if output is None:
        output = torch.empty((G, M, N), dtype=torch.bfloat16, device=XQ.device)
    if M == 0 or N == 0 or total_K == 0:
        return output

    k_starts = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=XQ.device), offsets_i32[:-1]]
    )
    k_sizes = offsets_i32 - k_starts

    # Per-group element base offsets into the concatenated blocked scale buffers.
    # Each group's blocked X scale occupies cdiv(M,128) * cdiv(Kg//32,4) * 512
    # bytes; W scale occupies cdiv(N,128) * cdiv(Kg//32,4) * 512 bytes.
    n_scale_cols = torch.div(k_sizes + 31, 32, rounding_mode="floor")
    n_col_blocks = torch.div(n_scale_cols + 3, 4, rounding_mode="floor")
    m_row_blocks = (M + 127) // 128
    n_row_blocks = (N + 127) // 128
    xs_group_sizes = (m_row_blocks * n_col_blocks * 512).to(torch.int64)
    ws_group_sizes = (n_row_blocks * n_col_blocks * 512).to(torch.int64)
    xs_bases = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=XQ.device),
            xs_group_sizes.cumsum(0)[:-1],
        ]
    )
    ws_bases = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=XQ.device),
            ws_group_sizes.cumsum(0)[:-1],
        ]
    )

    grid = lambda meta: (  # noqa: E731
        G,
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _mx8mx8bf16_grouped_2d2d_kernel[grid](
        xq_raw,
        wq_raw,
        xs_raw,
        ws_raw,
        output,
        k_sizes,
        k_starts,
        xs_bases,
        ws_bases,
        M,
        N,
        xq_raw.stride(0),
        xq_raw.stride(1),
        wq_raw.stride(0),
        wq_raw.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
    )
    return output


# ---------------------------------------------------------------------------
# Register as the ROCm implementation of mslk::mx8mx8bf16_grouped_mm
# ---------------------------------------------------------------------------

if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    if hasattr(torch.ops.mslk, "mx8mx8bf16_grouped_mm"):

        @torch.library.impl("mslk::mx8mx8bf16_grouped_mm", "CUDA")
        def _mx8mx8bf16_grouped_mm_rocm(
            XQ: torch.Tensor,
            WQ: torch.Tensor,
            x_scale: torch.Tensor,
            w_scale: torch.Tensor,
            offsets: torch.Tensor,
            output: Optional[torch.Tensor] = None,
            actual_num_tokens: Optional[int] = None,
        ) -> torch.Tensor:
            return matmul_mx8mx8bf16_grouped(
                XQ, WQ, x_scale, w_scale, offsets, output, actual_num_tokens
            )
