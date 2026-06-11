# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
MXFP8 (E4M3) activation x MXFP4 (E2M1) weight GEMM for ROCm/AMD gfx950+.

This is the ROCm counterpart of csrc/gemm/cutlass/mx8mx4bf16.cu.
It uses Triton's tl.dot_scaled, which lowers to the CDNA4 scaled-MFMA
instructions (V_MFMA_SCALE_F32_16X16X128_F8F6F4) available on gfx950
(MI350/MI355X).  The kernel is NOT compatible with gfx942 (MI300X) or
older CDNA architectures which lack hardware block-scaling support.

Tensor shapes (matching the CUDA CUTLASS convention):
  XQ      : [M, K]     — MXFP8 E4M3, stored as torch.float8_e4m3fnuz
  WQ      : [N, K//2]  — MXFP4 E2M1, two nibbles packed per uint8 byte
  x_scale : [M, K//32] — E8M0 per-block scale factors for XQ (block_size=32)
  w_scale : [N, K//32] — E8M0 per-block scale factors for WQ (block_size=32)
  output  : [M, N]     — bfloat16

Scale factor format:
  OCP MX uses one 8-bit (E8M0) exponent per block of 32 values along K.
  On AMD, tl.dot_scaled expects the scale tensor with shape [rows, K//32]
  where the inner dimension indexes blocks of 32 consecutive K elements.

Dequantisation (implicit in hardware / tl.dot_scaled):
  val_float = val_mxfp4 * 2^(scale_e8m0 - 127)   (per 32-element block)
  val_float = val_mxfp8 * 2^(scale_e8m0 - 127)   (per 32-element block)

References:
  - Triton tutorial 10: Block Scaled Matrix Multiplication
    https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
  - tl.dot_scaled API docs
    https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html
  - AMD CDNA4 ISA: V_MFMA_SCALE_F32_16X16X128_F8F6F4
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
    Tile configs tuned for the gfx950 CU layout (256 CUs, 64 threads/wavefront).

    BLOCK_M / BLOCK_N must be multiples of the MFMA tile (16).
    BLOCK_K must be 128: V_MFMA_SCALE_F32_16X16X128_F8F6F4 requires K=128
    FP8 elements per tile; smaller values fall back to a software-emulation
    path that emits an invalid 'f' constraint on the AMDGPU LLVM backend.
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
    """
    Drop tile configs that are clearly suboptimal for the given (M, N, K).

    Rules (gfx950, 256 CUs):
    - Skip configs whose tile is larger than the problem in M or N — they
      waste registers and reduce occupancy with no compute benefit.
    - For very small M (≤ 16) prefer small BLOCK_M to maximise the number
      of CTAs and keep more CUs busy.
    - Require at least 64 CTAs to fill one wave on 256 CUs when M*N is
      large enough to afford smaller tiles.
    """
    M = named_args["M"]
    N = named_args["N"]
    pruned = []
    for cfg in configs:
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        # Never use a tile larger than the problem dimension.
        if bm > M or bn > N:
            continue
        # For small M avoid large BLOCK_M that leaves most of the tile masked.
        if M <= 16 and bm > 16:
            continue
        if M <= 32 and bm > 32:
            continue
        pruned.append(cfg)
    return pruned if pruned else configs


def _prune_configs_grouped(configs, named_args, **kwargs):
    """Adapter for the grouped kernel: maps total_M -> M for _prune_configs."""
    named_args = dict(named_args)
    named_args["M"] = named_args.get("total_M", named_args.get("M", 1))
    return _prune_configs(configs, named_args, **kwargs)


# ---------------------------------------------------------------------------
# Core Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _mx8mx4bf16_kernel(
    XQ_ptr,  # [M, K]     float8_e4m3  (raw bytes; reinterpreted in kernel)
    WQ_ptr,  # [N, K//2]  uint8        (two MXFP4 nibbles per byte)
    XS_ptr,  # flat uint8  _to_blocked layout (E8M0 exponents for XQ)
    WS_ptr,  # flat uint8  _to_blocked layout (E8M0 exponents for WQ)
    Out_ptr,  # [M, N]     bfloat16
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,  # strides over WQ in *bytes* (K//2 bytes wide)
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # K tile in FP8 elements; MXFP4 uses BLOCK_K//2 bytes
) -> None:
    """
    Computes Out[M, N] = dequant(XQ)[M, K] @ dequant(WQ).T[K, N]
    using hardware-accelerated block-scaled tensor operations via tl.dot_scaled.

    XS/WS must be in the _to_blocked flat layout produced by fp4_quantize._to_blocked.
    Scale semantics:
      tl.dot_scaled expects scales shaped [BLOCK_M, BLOCK_K // 32] for A
      and [BLOCK_N, BLOCK_K // 32] for B (one scale per 32-element MX block).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ---- tile offsets ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # K//32 blocks per BLOCK_K FP8 elements
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_idx in tl.range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_idx * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)  # FP8 K indices
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)  # packed FP4 byte indices
        offs_ks = k_start // 32 + tl.arange(0, SCALE_BLOCK_K)  # scale block indices

        # ---- load XQ tile [BLOCK_M, BLOCK_K] in FP8 E4M3 ----
        xq_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        xq = tl.load(
            XQ_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=xq_mask,
            other=0,
        )  # dtype: float8_e4m3 (passed through as raw bytes to dot_scaled)

        # ---- load WQ tile [BLOCK_N, BLOCK_K//2] in packed uint8 ----
        # tl.dot_scaled with "e2m1" format unpacks the FP4 nibbles internally.
        wq_mask = (offs_n[:, None] < N) & (offs_k2[None, :] < K // 2)
        wq = tl.load(
            WQ_ptr + offs_n[:, None] * stride_wn + offs_k2[None, :] * stride_wk,
            mask=wq_mask,
            other=0,
        )  # dtype: uint8 (two FP4 values per byte)

        # ---- load scale tiles ----
        # XS and WS are in the _to_blocked flat layout produced by fp4_quantize._to_blocked:
        #   padded.view(R//128, 4, 32, C//4, 4).permute(0,3,2,1,4).reshape(-1,32,16).flatten()
        # Inverse mapping for logical (row, col):
        #   flat = (row//128 * n_col_blocks + col//4) * 512
        #         + (row%32) * 16 + (row%128//32) * 4 + col%4
        n_scale_cols = tl.cdiv(K, 32)  # K//32, number of scale columns
        n_col_blocks = tl.cdiv(
            n_scale_cols, 4
        )  # ceil(K//32 / 4) for _to_blocked layout

        m2d = offs_m[:, None]
        ks2d = offs_ks[None, :]
        xs_flat = (
            (m2d // 128 * n_col_blocks + ks2d // 4) * 512
            + (m2d % 32) * 16
            + (m2d % 128 // 32) * 4
            + ks2d % 4
        )
        xs = tl.load(XS_ptr + xs_flat, mask=(m2d < M) & (ks2d < n_scale_cols), other=0)

        n2d = offs_n[:, None]
        ws_flat = (
            (n2d // 128 * n_col_blocks + ks2d // 4) * 512
            + (n2d % 32) * 16
            + (n2d % 128 // 32) * 4
            + ks2d % 4
        )
        ws = tl.load(WS_ptr + ws_flat, mask=(n2d < N) & (ks2d < n_scale_cols), other=0)

        # ---- block-scaled dot product ----
        # tl.dot_scaled dispatches to mfma_scaled_* on gfx950.
        # A format "e4m3" = MXFP8 E4M3; B format "e2m1" = MXFP4 E2M1.
        # WQ is transposed: shape [BLOCK_N, BLOCK_K//2] -> [BLOCK_K//2, BLOCK_N].
        # rhs_scale must NOT be transposed per tl.dot_scaled API contract:
        # it always expects shape [N, K//group_size] regardless of rhs layout.
        acc = tl.dot_scaled(
            xq,
            xs,
            "e4m3",  # A [BLOCK_M, BLOCK_K], lhs_scale [BLOCK_M, SCALE_BLOCK_K]
            tl.trans(wq),
            ws,
            "e2m1",  # B [BLOCK_K//2, BLOCK_N], rhs_scale [BLOCK_N, SCALE_BLOCK_K]
            acc,
        )

    # ---- store output ----
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def matmul_mx8mx4bf16(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MXFP8 (E4M3) activation × MXFP4 (E2M1) weight GEMM → BFloat16.

    Matches the CUDA CUTLASS API contract: x_scale and w_scale must be in
    the _to_blocked flat layout (produced by fp4_quantize._to_blocked and
    triton_quantize_mx4_unpack respectively).  The kernel reads blocked
    scales directly via fused index arithmetic — no _from_blocked copy.

    Args:
        XQ      : [M, K]         torch.float8_e4m3fnuz  — MX-quantised activations
        WQ      : [N, K//2]      torch.uint8            — packed MXFP4 (2 nibbles/byte)
        x_scale : flat uint8     — _to_blocked output for XQ scales
        w_scale : flat uint8     — _to_blocked output for WQ scales
        output  : [M, N]         torch.bfloat16         — optional pre-allocated output

    Returns:
        [M, N] bfloat16 tensor.
    """
    assert XQ.ndim == 2, f"XQ must be 2D, got {XQ.shape}"
    assert WQ.ndim == 2, f"WQ must be 2D, got {WQ.shape}"
    M, K = XQ.shape
    N = WQ.shape[0]

    assert K % 32 == 0, f"K={K} must be a multiple of the MX block size (32)"

    wq_raw = WQ.view(torch.uint8).reshape(N, K // 2)

    # Pass blocked scales directly — the kernel reads them via fused
    # _to_blocked index arithmetic (BLOCKED_SCALES=True path).
    xs_raw = x_scale.view(torch.uint8)
    ws_raw = w_scale.view(torch.uint8)

    if output is None:
        output = torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)

    if M == 0 or N == 0 or K == 0:
        return output

    xq_raw = XQ.view(torch.uint8)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _mx8mx4bf16_kernel[grid](
        xq_raw,
        wq_raw,
        xs_raw,
        ws_raw,
        output,
        M,
        N,
        K,
        xq_raw.stride(0),
        xq_raw.stride(1),
        wq_raw.stride(0),
        wq_raw.stride(1),
        output.stride(0),
        output.stride(1),
    )
    return output


# ---------------------------------------------------------------------------
# Grouped kernel (2D-3D): XQ [total_M, K], WQ [G, K//2, N] (col-major W)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_configs(),
    key=["total_M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs_grouped},
)
@triton.jit
def _mx8mx4bf16_grouped_kernel(
    XQ_ptr,  # [total_M, K]   float8_e4m3 bytes
    WQ_ptr,  # [G, K//2, N]   uint8  (two MXFP4 nibbles per byte, col-major W)
    XS_ptr,  # flat uint8  _to_blocked layout, groups concatenated
    WS_ptr,  # flat uint8  _to_blocked layout, groups stacked (all same N)
    Out_ptr,  # [total_M, N]   bfloat16
    offsets_ptr,  # [G]            int32  cumulative row-end per group
    xs_offsets_ptr,  # [G]            int32  byte offset into XS_ptr for each group
    m_tiles_ptr,  # [5*G]          int32  cumulative pid_m tile-ends for each
    #                       BLOCK_M in {16,32,64,128,256} × group
    total_M,  # used only for autotuner key/prune; not referenced in kernel body
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wg,
    stride_wk2,
    stride_wn,  # WQ strides ([G, K//2, N])
    stride_om,
    stride_on,
    G,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """
    Grouped MXFP8 x MXFP4 GEMM.

    Each CTA is assigned to a tile within ONE group's M range, guaranteed by the grid.
    The grid uses sum_g(ceil(M_g / BLOCK_M)) tiles along M so tiles never cross group
    boundaries. This avoids using wrong-group scales for cross-boundary rows.

    m_tiles_ptr holds cumulative tile counts for all BLOCK_M candidates (5 rows × G cols).
    The kernel selects the right row via BLOCK_M constexpr.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32

    # ---- select m_tiles row for this BLOCK_M value ----
    if BLOCK_M == 16:  # noqa: SIM108
        bm_row: tl.constexpr = 0
    elif BLOCK_M == 32:
        bm_row: tl.constexpr = 1
    elif BLOCK_M == 64:
        bm_row: tl.constexpr = 2
    elif BLOCK_M == 128:
        bm_row: tl.constexpr = 3
    else:  # BLOCK_M == 256
        bm_row: tl.constexpr = 4
    m_tiles_base = m_tiles_ptr + bm_row * G

    # ---- map pid_m to (group_id, local_tile_within_group) ----
    # m_tiles_base[g] = cumulative tile count for groups 0..g for this BLOCK_M.
    group_id = 0
    group_tile_start = 0
    group_start = 0
    for g in tl.range(0, G):
        tile_end = tl.load(m_tiles_base + g)
        if pid_m >= tile_end:
            group_id = g + 1
            group_tile_start = tile_end
            group_start = tl.load(offsets_ptr + g)

    # local tile index within this group, and its starting local row
    local_tile = pid_m - group_tile_start
    local_m_start = local_tile * BLOCK_M
    abs_m_start = group_start + local_m_start

    offs_m = abs_m_start + tl.arange(0, BLOCK_M)  # absolute rows in output
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_m = local_m_start + tl.arange(0, BLOCK_M)  # row within this group

    group_end = tl.load(offsets_ptr + group_id)  # absolute end row for this group

    n_scale_cols = tl.cdiv(K, 32)
    n_col_blocks = tl.cdiv(n_scale_cols, 4)

    xs_group_offset = tl.load(xs_offsets_ptr + group_id)

    ws_n_row_blocks = tl.cdiv(N, 128)
    ws_group_offset = group_id * (ws_n_row_blocks * n_col_blocks * 512)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_idx in tl.range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_idx * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        offs_ks = k_start // 32 + tl.arange(0, SCALE_BLOCK_K)

        row_valid = offs_m[:, None] < group_end
        xq = tl.load(
            XQ_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=row_valid & (offs_k[None, :] < K),
            other=0,
        )

        wq = tl.load(
            WQ_ptr
            + group_id * stride_wg
            + offs_k2[:, None] * stride_wk2
            + offs_n[None, :] * stride_wn,
            mask=(offs_k2[:, None] < K // 2) & (offs_n[None, :] < N),
            other=0,
        )

        ks2d = offs_ks[None, :]
        lm2d = local_m[:, None]
        xs_flat = (
            xs_group_offset
            + (lm2d // 128 * n_col_blocks + ks2d // 4) * 512
            + (lm2d % 32) * 16
            + (lm2d % 128 // 32) * 4
            + ks2d % 4
        )
        xs = tl.load(XS_ptr + xs_flat, mask=row_valid & (ks2d < n_scale_cols), other=0)

        n2d = offs_n[:, None]
        ws_flat = (
            ws_group_offset
            + (n2d // 128 * n_col_blocks + ks2d // 4) * 512
            + (n2d % 32) * 16
            + (n2d % 128 // 32) * 4
            + ks2d % 4
        )
        ws = tl.load(WS_ptr + ws_flat, mask=(n2d < N) & (ks2d < n_scale_cols), other=0)

        acc = tl.dot_scaled(
            xq,
            xs,
            "e4m3",
            wq,
            ws,
            "e2m1",
            acc,
        )

    tl.store(
        Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < group_end) & (offs_n[None, :] < N),
    )


def matmul_mx8mx4bf16_grouped(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    offsets: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Grouped MXFP8 (E4M3) × MXFP4 (E2M1) GEMM → BFloat16.

    Matches the CUDA mx8mx4bf16_grouped_mm API contract (2D-3D inputs):

    Args:
        XQ       : [total_M, K]    torch.float8_e4m3fnuz  — concatenated activations
        WQ       : [G, K//2, N]    torch.float4_e2m1fn_x2 — stacked weights, col-major
                   (caller must pass WQ.transpose(-2, -1) as in CUDA: [G, N, K] -> [G, K//2, N])
        x_scale  : flat uint8 — per-group _to_blocked buffers concatenated (empty groups skipped)
        w_scale  : flat uint8 — per-group _to_blocked buffers stacked (one per group, same N)
        offsets  : [G]  int32       — cumulative M end indices per group
        output   : [total_M, N]    optional pre-allocated bfloat16 output

    Returns:
        [total_M, N] bfloat16 tensor.
    """
    assert XQ.ndim == 2, f"XQ must be 2D, got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be 3D [G, K//2, N], got {WQ.shape}"

    total_M, K = XQ.shape
    G_wq, K2, N = WQ.shape
    assert K2 == K // 2, f"WQ K dim mismatch: got {K2}, expected {K // 2}"
    assert K % 32 == 0, f"K={K} must be a multiple of 32"

    G = offsets.shape[0]
    assert G_wq == G, f"WQ group dim {G_wq} != offsets length {G}"

    if output is None:
        output = torch.empty((total_M, N), dtype=torch.bfloat16, device=XQ.device)

    if total_M == 0 or N == 0 or K == 0:
        return output

    xq_raw = XQ.view(torch.uint8)
    wq_raw = WQ.view(torch.uint8)  # [G, K//2, N]
    xs_raw = x_scale.view(torch.uint8).contiguous()
    ws_raw = w_scale.view(torch.uint8).contiguous()
    offsets_i32 = offsets.to(torch.int32)

    # Compute per-group byte offsets into xs_raw.
    # Each group g contributes ceil(M_g, 128) * n_col_blocks * 512 bytes.
    # Empty groups (M_g == 0) contribute 0 bytes and are skipped in xs_raw.
    n_col_blocks = (K // 32 + 3) // 4
    m_starts = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=offsets.device), offsets_i32[:-1]]
    )
    m_sizes = offsets_i32 - m_starts
    xs_group_sizes = ((m_sizes + 127) // 128) * n_col_blocks * 512
    xs_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=offsets.device),
            xs_group_sizes.cumsum(0, dtype=torch.int32)[:-1],
        ]
    )

    # Precompute cumulative tile counts for all 5 BLOCK_M candidates {16,32,64,128,256}.
    # m_tiles[bm_row, g] = cumulative M-tile count for groups 0..g at BLOCK_M=bm.
    # Stored flat as [5*G] int32 so the kernel can index [bm_row * G + group_id].
    # This lets the kernel select the right row via BLOCK_M constexpr without
    # any dependency on which config the autotuner chooses.
    _bm_values = [16, 32, 64, 128, 256]
    m_tiles = torch.stack(
        [((m_sizes + bm - 1) // bm).cumsum(0, dtype=torch.int32) for bm in _bm_values]
    ).contiguous()  # [5, G] int32, on same device as offsets

    def grid(meta):  # noqa: E731
        bm = meta["BLOCK_M"]
        total_m_tiles = int(((m_sizes + bm - 1) // bm).sum().item())
        return (total_m_tiles, triton.cdiv(N, meta["BLOCK_N"]))

    _mx8mx4bf16_grouped_kernel[grid](
        xq_raw,
        wq_raw,
        xs_raw,
        ws_raw,
        output,
        offsets_i32,
        xs_offsets,
        m_tiles,
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
        G,
    )
    return output


# ---------------------------------------------------------------------------
# Register as the ROCm implementation of mslk::mx8mx4bf16
# ---------------------------------------------------------------------------
# The C++ schema for mx8mx4bf16 is declared in gemm_ops.cpp under USE_ROCM
# (see the block added there).  Importing this module at Python startup
# registers the Triton kernel above as the CUDA-dispatch implementation,
# so torch.ops.mslk.mx8mx4bf16(...) calls into Triton on AMD GPUs.

if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    if hasattr(torch.ops.mslk, "mx8mx4bf16"):

        @torch.library.impl("mslk::mx8mx4bf16", "CUDA")
        def _mx8mx4bf16_rocm(
            XQ: torch.Tensor,
            WQ: torch.Tensor,
            x_scale: torch.Tensor,
            w_scale: torch.Tensor,
            output: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            return matmul_mx8mx4bf16(XQ, WQ, x_scale, w_scale, output)

    if hasattr(torch.ops.mslk, "mx8mx4bf16_grouped_mm"):

        @torch.library.impl("mslk::mx8mx4bf16_grouped_mm", "CUDA")
        def _mx8mx4bf16_grouped_mm_rocm(
            XQ: torch.Tensor,
            WQ: torch.Tensor,
            x_scale: torch.Tensor,
            w_scale: torch.Tensor,
            offsets: torch.Tensor,
            output: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            return matmul_mx8mx4bf16_grouped(XQ, WQ, x_scale, w_scale, offsets, output)
