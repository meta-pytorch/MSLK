# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
Triton f4f4bf16 GEMM for AMD ROCm (gfx950 / MI350x).

Native MXFP4 only: ``mxfp4_gemm`` uses ``tl.dot_scaled`` with
``lhs_format="e2m1"``, ``rhs_format="e2m1"``, ``scale_format="e8m0"``,
block_size=32 on gfx950 (MI350x) ``v_mfma_scale_f32_*_f8f6f4``.

NVFP4 and MXFP4 emulation (dequant→FP8 fallback on gfx942 / MI300X) are
intentionally not implemented on ROCm — use CUDA for NVFP4/MXFP4_16, and gfx950
for native MXFP4 block_size=32.
"""

from typing import Optional

import torch
import triton  # @manual
import triton.language as tl  # @manual


# AMD HIPOptions (NOT triton.Config fields, NOT keys autotune dedupes on —
# they're launch-time kwargs). Same defaults across all kernels in this file.
#   waves_per_eu=1     hint to grow register tile / shrink occupancy
#   matrix_instr_nonkdim=32  pick v_mfma_*_32x32x* over the 16x16x* variant
# `kpack` is deprecated on gfx950 — don't pass it.
_HIP_OPTS = {"waves_per_eu": 1, "matrix_instr_nonkdim": 32}


# -----------------------------------------------------------------------------
# MXFP4 kernel — native dot_scaled path (gfx950 / SM100)
# -----------------------------------------------------------------------------


def _mxfp4_autotune_configs():
    """Sweep (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages, waves_per_eu).

    Constraints: BLOCK_K must be a multiple of 32 (E8M0 scale block) and of 2
    (FP4 packing). The native MFMA tile is M=N=32, K=64 — keep BLOCK_K ≥ 64.

    The interesting axes after the initial autotune (which picked
    BM=256 BN=128 BK=256 ns=3) were `waves_per_eu` (occupancy hint to hide LDS
    latency) and `GROUP_M` (L2-friendly super-grouping). We sweep both here
    around the previously-winning shape.
    """
    configs = []
    for bm, bn, bk in (
        (256, 256, 256),  # WINNER on large square shapes
        (256, 128, 256),
        (128, 256, 256),
        (256, 256, 128),
        (128, 128, 256),
        (128, 128, 128),  # small-M fallback
        (64, 128, 128),
        (64, 64, 128),
    ):
        for gm in (4, 8):
            for nw in (4, 8):
                for ns in (2, 3, 4):
                    if bm * bn > 256 * 256:
                        continue
                    if bm * bk + bk * bn > 256 * 256 * 2:
                        continue
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": bm,
                                "BLOCK_N": bn,
                                "BLOCK_K": bk,
                                "GROUP_M": gm,
                            },
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs


@triton.autotune(configs=_mxfp4_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _mxfp4_gemm_kernel(
    # Pointers
    a_ptr,  # [M, K//2] uint8
    b_ptr,  # [N, K//2] uint8
    a_scale_ptr,  # [M, K//32] uint8 (E8M0)
    b_scale_ptr,  # [N, K//32] uint8 (E8M0)
    c_ptr,  # [M, N] bf16
    # Sizes
    M,
    N,
    K,
    # Strides (in elements/bytes for the respective tensor)
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    # Meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # K elements, must be multiple of 32 and of 2
    GROUP_M: tl.constexpr,  # L2-friendly super-group width along M
):
    """MXFP4 × MXFP4 → BF16 via tl.dot_scaled.

    Layout assumptions:
      - A packed [M, K//2] uint8: low nibble is even-K element, high nibble odd.
      - B packed [N, K//2] uint8 (same packing).
      - a_scale [M, K//32] uint8 (E8M0, one scale per 32 K-elements per row).
      - b_scale [N, K//32] uint8 (E8M0).
      - C [M, N] bf16.

    BLOCK_K is in *unpacked* K elements; we load BLOCK_K//2 packed bytes per
    iteration. SCALE_PER_BLOCK_K = BLOCK_K // 32 scale elements per K block.

    GROUP_M reorders program IDs so adjacent workgroups in M share B tiles in
    L2 (standard Triton GEMM super-grouping). Cuts HBM B-tile traffic ~2-4x
    versus row-major program-id ordering.
    """
    # L2 super-group reorder of (pid_m, pid_n).
    pid = tl.program_id(0) + tl.program_id(1) * tl.num_programs(0)
    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    PACKED_BLOCK_K: tl.constexpr = BLOCK_K // 2
    SCALE_PER_BLOCK_K: tl.constexpr = BLOCK_K // 32

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_packed = tl.arange(0, PACKED_BLOCK_K)
    offs_k_scale = tl.arange(0, SCALE_PER_BLOCK_K)

    # Clamp row/col indices used for *addressing* so out-of-range lanes (when
    # BLOCK_M/BLOCK_N don't divide M/N) read a valid in-bounds element instead
    # of running off the end of the allocation. `tl.dot_scaled` needs full
    # dense tiles (no masked load), so we over-read clamped-but-valid rows and
    # discard them via the store mask below. Without this, large-K shapes whose
    # M isn't a multiple of BLOCK_M fault (the row stride pushes the over-read
    # past mapped pages).
    safe_m = tl.minimum(offs_m, M - 1)
    safe_n = tl.minimum(offs_n, N - 1)

    # A: row-major (M, K) — load tile as [BLOCK_M, PACKED_BLOCK_K] uint8.
    a_base = a_ptr + safe_m[:, None] * stride_am + offs_k_packed[None, :] * stride_ak
    # B: stored as (N, K) row-major; tl.dot_scaled rhs wants [K_packed, BLOCK_N].
    b_base = b_ptr + safe_n[None, :] * stride_bn + offs_k_packed[:, None] * stride_bk
    # Scales: tl.dot_scaled expects both as [outer, K_blocks].
    as_base = (
        a_scale_ptr + safe_m[:, None] * stride_asm + offs_k_scale[None, :] * stride_ask
    )
    bs_base = (
        b_scale_ptr + safe_n[:, None] * stride_bsn + offs_k_scale[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_packed = K // 2
    for k_off_packed in range(0, K_packed, PACKED_BLOCK_K):
        a_pack = tl.load(a_base + k_off_packed * stride_ak)
        b_pack = tl.load(b_base + k_off_packed * stride_bk)
        scale_off = (k_off_packed * 2) // 32
        a_scale = tl.load(as_base + scale_off * stride_ask)
        b_scale = tl.load(bs_base + scale_off * stride_bsk)
        # Chain accumulation through dot_scaled (avoids a separate add).
        acc = tl.dot_scaled(
            a_pack,
            a_scale,
            "e2m1",
            b_pack,
            b_scale,
            "e2m1",
            acc=acc,
            out_dtype=tl.float32,
        )

    c = acc.to(tl.bfloat16)
    c_base = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_base, c, mask=mask)


def mxfp4_gemm(
    XQ: torch.Tensor,  # [M, K//2] uint8
    WQ: torch.Tensor,  # [N, K//2] uint8
    x_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0)
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MXFP4 × MXFP4 → BF16 via native ``tl.dot_scaled`` MFMA (gfx950 only on ROCm)."""
    _require_native_mxfp4()
    # Accept either raw uint8 or torch's packed FP4 dtypes (float4_e2m1fn_x2 for
    # the packed nibble pair, float8_e8m0fnu for the E8M0 scale). The kernel
    # only knows uint8; view the typed tensors as uint8 for indexing arithmetic.
    if XQ.dtype != torch.uint8:
        XQ = XQ.view(torch.uint8)
    if WQ.dtype != torch.uint8:
        WQ = WQ.view(torch.uint8)
    if x_scale.dtype != torch.uint8:
        x_scale = x_scale.view(torch.uint8)
    if w_scale.dtype != torch.uint8:
        w_scale = w_scale.view(torch.uint8)
    M = XQ.shape[0]
    N = WQ.shape[0]
    K = XQ.shape[1] * 2
    assert WQ.shape[1] * 2 == K
    assert K % 32 == 0

    if output is None:
        output = torch.empty((M, N), device=XQ.device, dtype=torch.bfloat16)

    # autotune picks BLOCK_M/N/K/GROUP_M; grid is computed per-config.
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    _mxfp4_gemm_kernel[grid](
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
        **_HIP_OPTS,
    )
    return output


def _has_native_mxfp4_mfma() -> bool:
    """True iff `tl.dot_scaled(e2m1, e2m1, e8m0)` has a native MFMA lowering.

    gfx950 (MI350x) on ROCm; CUDA SM100+ otherwise."""
    if not torch.cuda.is_available():
        return False
    if torch.version.hip is None:
        return True  # CUDA SM100 has native dot_scaled
    arch = (
        getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    ).lower()
    return "gfx950" in arch


def _require_native_mxfp4() -> None:
    if not _has_native_mxfp4_mfma():
        raise RuntimeError(
            "Native MXFP4 GEMM requires gfx950 (MI350) on ROCm. "
            "FP4 emulation paths (NVFP4 or dequant→FP8 on gfx942) are not supported."
        )


# -----------------------------------------------------------------------------
# Grouped MXFP4 kernel — mslk::f4f4bf16_grouped_mm
#
# Maps to ``csrc/gemm/cutlass/f4f4bf16_grouped.cu::f4f4bf16_grouped_mm``.
# Inputs:
#   XQ: [total_M, K//2] uint8 contiguous — A's expert chunks concatenated along M
#   WQ: [G, K//2, N]    uint8 — incoming as WQ.transpose(-2,-1).is_contiguous()
#                               (i.e. caller transposed an original [G, N, K//2])
#   x_scale: [total_M, K//32] uint8 E8M0
#   w_scale: [G, N,       K//32] uint8 E8M0
#   offsets: [G] int32, cumulative M endpoints (offsets[-1] == total_M)
# Output:
#   [total_M, N] bf16
#
# Per-expert dispatch: pid_g IS the expert index — each program reads two
# adjacent `offsets` entries with O(1) loads, no linear scan. Scales to large
# G (e.g. G=256 DeepSeek MoE) without per-program overhead. Within an expert,
# programs tile (M, N) the same way as the dense MXFP4 kernel.
# -----------------------------------------------------------------------------


def _mxfp4_grouped_autotune_configs():
    """Grouped MXFP4 GEMM configs spanning small-M (MoE decode) → large shapes.

    G ∈ {4,8}, M ∈ {128, 512, 1024}, N=4096, K ∈ {2048,4096}. Findings:
      - Small-M (M=128): BM ≤ 128 + BN ∈ {64,128,256}, num_warps=2-4 (not 8 —
        too little K work per warp). BM=32 BN=64 nw=2 hits ~450 TFLOPS at M=128.
      - Large-M (M≥1024): BM=128 BN=256 BK=256 nw=8 ns=2 wins (1502 TFLOPS).
      - num_stages=2 beats ns=3 on almost every shape; ns=3 only ties at mid-M.
      - num_warps=2 surprisingly competitive on small-M (compute-bound becomes
        latency-bound when the tile is tiny).
    """
    configs = []
    # (BM, BN, BK, GM, num_warps, num_stages) tuples directly — avoid the
    # combinatorial blowup of nested loops and keep the sweep focused.
    spec = [
        # Small-M decode-ish winners (M ≤ 256 per expert)
        (32, 64, 256, 1, 2, 3),
        (32, 256, 256, 1, 2, 3),
        (64, 128, 256, 1, 8, 2),
        (64, 128, 256, 4, 4, 3),
        (64, 256, 256, 1, 8, 2),
        (128, 64, 128, 1, 4, 2),
        (128, 64, 128, 4, 4, 2),
        (128, 64, 256, 1, 2, 2),
        # Mid-M (M ~ 512)
        (128, 256, 256, 1, 8, 2),
        (128, 256, 256, 1, 4, 2),
        (128, 256, 128, 1, 8, 3),
        (128, 128, 256, 4, 8, 2),
        # Large-M (M ≥ 1024) — winner of the sweep
        (128, 256, 256, 4, 8, 2),
        (128, 256, 256, 4, 2, 2),
        (128, 128, 256, 1, 8, 2),
        (128, 128, 256, 1, 2, 3),
        # Fallback for tiny shapes (BM ≤ M)
        (16, 64, 128, 1, 4, 2),
        (16, 128, 128, 1, 4, 2),
        (16, 256, 256, 1, 4, 2),
    ]
    for bm, bn, bk, gm, nw, ns in spec:
        configs.append(
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


@triton.autotune(
    configs=_mxfp4_grouped_autotune_configs(), key=["MAX_M_PER_GROUP", "N", "K", "G"]
)
@triton.jit
def _mxfp4_grouped_mm_kernel(
    # Pointers
    a_ptr,  # [total_M, K//2] uint8
    b_ptr,  # [G, K//2, N] uint8 (caller passed transpose-of-[G,N,K//2])
    a_scale_ptr,  # [total_M, K//32] uint8
    b_scale_ptr,  # [G, N, K//32] uint8
    c_ptr,  # [total_M, N] bf16
    offsets_ptr,  # [G] int32, cumulative M ends
    # Sizes
    total_M,
    N,
    K,
    G,
    MAX_M_PER_GROUP,  # autotune key only (not used inside)
    # Strides
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_asm,
    stride_ask,
    stride_bsg,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    # Meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """One program per (expert × M-tile × N-tile) — expert located via linear
    scan over `offsets`.

    We grid over the *maximum* per-group tile count and skip empty programs
    inside the kernel; this avoids an additional host-side prefix-sum pass and
    keeps the launcher trivial. For G≤16 this overhead is negligible.
    """
    # tile id within this expert is on grid-X: its upper bound is
    # cdiv(total_M, BLOCK_M) * cdiv(N, BLOCK_N), which can exceed HIP's 65535
    # grid-Y/Z limit, so it must live on X (limit 2**31-1). Expert id is on Y
    # (G is small).
    pid = tl.program_id(0)  # tile id within this expert
    pid_g = tl.program_id(1)  # expert id 0..G-1

    # Resolve [m_start, m_end) for this expert. Use a mask on the load itself
    # (Triton evaluates both branches of tl.where, so a guarded subtraction would
    # still OOB-read offsets[-1] when pid_g == 0).
    safe_prev = tl.maximum(pid_g - 1, 0)
    prev = tl.load(offsets_ptr + safe_prev, mask=pid_g > 0, other=0).to(tl.int32)
    m_start = prev
    m_end = tl.load(offsets_ptr + pid_g).to(tl.int32)
    M_local = m_end - m_start

    num_pid_m = tl.cdiv(M_local, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n
    if pid >= total_tiles:
        return

    # L2 super-group reorder of (pid_m, pid_n).
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    PACKED_BLOCK_K: tl.constexpr = BLOCK_K // 2
    SCALE_PER_BLOCK_K: tl.constexpr = BLOCK_K // 32

    offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Clamp to valid M rows so pointer arithmetic doesn't go OOB (Triton evaluates
    # `ptr + i*stride` even where mask is False — the mask only suppresses the load).
    offs_m_global = tl.minimum(m_start + offs_m_local, total_M - 1)
    offs_n_unclamped = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.minimum(offs_n_unclamped, N - 1)
    offs_k_packed = tl.arange(0, PACKED_BLOCK_K)
    offs_k_scale = tl.arange(0, SCALE_PER_BLOCK_K)

    m_mask = offs_m_local < M_local
    n_mask = offs_n_unclamped < N

    # A: row-major [total_M, K//2], pick the expert's slab via m_start offset.
    a_base = (
        a_ptr + offs_m_global[:, None] * stride_am + offs_k_packed[None, :] * stride_ak
    )
    # B: [G, K//2, N], stored with caller-supplied .transpose(-2,-1) so the
    # K-major view here is what tl.dot_scaled rhs expects.
    b_base = (
        b_ptr
        + pid_g * stride_bg
        + offs_k_packed[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    as_base = (
        a_scale_ptr
        + offs_m_global[:, None] * stride_asm
        + offs_k_scale[None, :] * stride_ask
    )
    bs_base = (
        b_scale_ptr
        + pid_g * stride_bsg
        + offs_n[:, None] * stride_bsn
        + offs_k_scale[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    K_packed = K // 2
    for k_off_packed in range(0, K_packed, PACKED_BLOCK_K):
        a_pack = tl.load(
            a_base + k_off_packed * stride_ak, mask=m_mask[:, None], other=0
        )
        b_pack = tl.load(b_base + k_off_packed * stride_bk)
        scale_off = (k_off_packed * 2) // 32
        a_scale = tl.load(
            as_base + scale_off * stride_ask, mask=m_mask[:, None], other=0
        )
        b_scale = tl.load(bs_base + scale_off * stride_bsk)
        acc = tl.dot_scaled(
            a_pack,
            a_scale,
            "e2m1",
            b_pack,
            b_scale,
            "e2m1",
            acc=acc,
            out_dtype=tl.float32,
        )

    c = acc.to(tl.bfloat16)
    c_base = c_ptr + offs_m_global[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_base, c, mask=mask)


def mxfp4_grouped_mm(
    XQ: torch.Tensor,  # [total_M, K//2] uint8
    WQ: torch.Tensor,  # [G, K//2, N] uint8 (K-major, caller transposed)
    x_scale: torch.Tensor,  # [total_M, K//32] uint8
    w_scale: torch.Tensor,  # [G, N, K//32] uint8
    offsets: torch.Tensor,  # [G] int32, cumulative M ends
    output: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,  # unused in MXFP4 path
) -> torch.Tensor:
    """ROCm Triton implementation of mslk::f4f4bf16_grouped_mm (MXFP4 path)."""
    _require_native_mxfp4()
    if XQ.dtype != torch.uint8:
        XQ = XQ.view(torch.uint8)
    if WQ.dtype != torch.uint8:
        WQ = WQ.view(torch.uint8)
    if x_scale.dtype != torch.uint8:
        x_scale = x_scale.view(torch.uint8)
    if w_scale.dtype != torch.uint8:
        w_scale = w_scale.view(torch.uint8)

    assert offsets.dtype == torch.int32 and offsets.dim() == 1

    if WQ.dim() != 3:
        raise RuntimeError(
            "ROCm f4f4bf16_grouped_mm only supports 2D-3D grouped GEMM "
            "(XQ [total_M, K//2], WQ [G, K//2, N] after caller transpose). "
            "2D-2D K-grouped layout is CUDA-only."
        )

    total_M = XQ.shape[0]
    G = WQ.shape[0]
    # WQ here is [G, K//2, N] (already transposed by caller per the CUTLASS
    # convention WQ.transpose(-2,-1).is_contiguous()).
    K = WQ.shape[1] * 2
    N = WQ.shape[2]
    assert offsets.shape[0] == G
    assert K % 32 == 0

    # CUTLASS accepts both 2D [total_M, K//32] and 3D [G, M, K//32] x_scale
    # (the latter is what the existing test produces via torch.stack). Flatten
    # to 2D — stack produces contiguous memory so the byte layout is identical
    # to a per-expert concatenation.
    if x_scale.dim() == 3:
        assert x_scale.is_contiguous(), "3D x_scale must be contiguous"
        x_scale = x_scale.reshape(-1, K // 32)
    assert x_scale.shape == (total_M, K // 32), (
        f"x_scale shape mismatch: got {tuple(x_scale.shape)}, expected ({total_M}, {K // 32})"
    )
    assert w_scale.shape == (G, N, K // 32), (
        f"w_scale shape mismatch: got {tuple(w_scale.shape)}, expected ({G},{N},{K // 32})"
    )

    if output is None:
        output = torch.empty((total_M, N), device=XQ.device, dtype=torch.bfloat16)

    # Autotune key only: the *average* per-group M is a stable bucketing hint
    # (we don't read the device `offsets` tensor at launch time).
    max_m_per_group = (total_M + G - 1) // G

    # Grid must cover the LARGEST possible single-expert tile count, not the
    # average — token distribution across experts is uneven in real MoE, and an
    # expert with more than total_M/G rows would otherwise have its tail M-tiles
    # never launched (silent wrong output). total_M is the safe upper bound on
    # any one expert's row count; the kernel's `pid >= total_tiles` guard skips
    # the over-provisioned tiles per expert.
    def grid(META):
        return (
            triton.cdiv(total_M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            G,
        )

    _mxfp4_grouped_mm_kernel[grid](
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        offsets,
        total_M,
        N,
        K,
        G,
        max_m_per_group,
        XQ.stride(0),
        XQ.stride(1),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        output.stride(0),
        output.stride(1),
        **_HIP_OPTS,
    )
    return output


# -----------------------------------------------------------------------------
# Grouped-stacked MXFP4 — mslk::f4f4bf16_grouped_stacked
#
# CUTLASS signature (csrc/gemm/cutlass/f4f4bf16_grouped.cu):
#   XQ:        [total_M, K//2] uint8 contiguous
#   WQ:        [G, N, K//2]    uint8 — NOT pre-transposed (cf. _grouped_mm which gets [G, K//2, N])
#   x_scale:   [total_M, K//32] uint8 (2D); also accepts [G, M, K//32] 3D when stacked uniformly
#   w_scale:   [G, N, K//32]   uint8
#   M_sizes:   [G] int64 — per-expert token counts (NOT cumulative)
#   global_scale, starting_row_after_padding, use_mx — optional / ignored on AMD MXFP4 path
#
# Output: [total_M, N] bf16
#
# Implementation: reuse _mxfp4_grouped_mm_kernel by (1) converting M_sizes →
# cumulative offsets, (2) remapping WQ's [G, N, K//2] strides into the kernel's
# (stride_bg, stride_bk, stride_bn) slots so the inner-loop addressing math stays
# the same. No new kernel needed.
# -----------------------------------------------------------------------------


def mxfp4_grouped_stacked_gemm(
    XQ: torch.Tensor,  # [total_M, K//2] uint8 contiguous
    WQ: torch.Tensor,  # [G, N, K//2] uint8
    x_scale: torch.Tensor,  # [total_M, K//32] or [G, M, K//32] uint8
    w_scale: torch.Tensor,  # [G, N, K//32] uint8
    M_sizes: torch.Tensor,  # [G] int64
    output: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,  # unused (NVFP4 only)
    starting_row_after_padding: Optional[
        torch.Tensor
    ] = None,  # CUTLASS-specific; ignored on AMD
    use_mx: bool = True,
    offsets_override: Optional[
        torch.Tensor
    ] = None,  # AMD-only: pre-computed int32 cumulative M ends; skips cumsum
) -> torch.Tensor:
    """ROCm Triton implementation of mslk::f4f4bf16_grouped_stacked (MXFP4 path)."""
    assert use_mx, (
        "AMD path only supports MXFP4 (use_mx=True). NVFP4 path not implemented."
    )
    _require_native_mxfp4()

    if XQ.dtype != torch.uint8:
        XQ = XQ.view(torch.uint8)
    if WQ.dtype != torch.uint8:
        WQ = WQ.view(torch.uint8)
    if x_scale.dtype != torch.uint8:
        x_scale = x_scale.view(torch.uint8)
    if w_scale.dtype != torch.uint8:
        w_scale = w_scale.view(torch.uint8)

    assert M_sizes.dim() == 1
    G = M_sizes.shape[0]
    assert WQ.shape[0] == G, f"WQ.shape[0]={WQ.shape[0]} != G={G}"
    N = WQ.shape[1]
    K = WQ.shape[2] * 2
    total_M = XQ.shape[0]
    assert K % 32 == 0

    # CUTLASS accepts 2D or 3D x_scale; flatten 3D to 2D (contig stack === concat byte layout).
    if x_scale.dim() == 3:
        assert x_scale.is_contiguous(), "3D x_scale must be contiguous"
        x_scale = x_scale.reshape(-1, K // 32)
    assert x_scale.shape == (total_M, K // 32)
    assert w_scale.shape == (G, N, K // 32), (
        f"w_scale shape mismatch: got {tuple(w_scale.shape)}, expected ({G},{N},{K // 32})"
    )

    # Convert M_sizes (per-expert counts) → cumulative offsets (kernel expects this).
    if offsets_override is not None:
        assert offsets_override.dtype == torch.int32 and offsets_override.shape == (G,)
        offsets = offsets_override
    else:
        offsets = torch.cumsum(M_sizes, dim=0).to(torch.int32)

    if output is None:
        output = torch.empty((total_M, N), device=XQ.device, dtype=torch.bfloat16)

    # Autotune key only (average per-group M); grid covers the largest possible
    # single-expert tile count via total_M — see mxfp4_grouped_mm for rationale
    # (uneven expert sizes would otherwise drop tail M-tiles).
    max_m_per_group = (total_M + G - 1) // G

    def grid(META):
        return (
            triton.cdiv(total_M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            G,
        )

    # Stride remap: kernel reads B as [G, K//2, N] with (stride_bg, stride_bk, stride_bn).
    # We have [G, N, K//2] with (stride_g, stride_n, stride_k). Re-bind:
    #   stride_bg = WQ.stride(0)
    #   stride_bk = WQ.stride(2)
    #   stride_bn = WQ.stride(1)
    _mxfp4_grouped_mm_kernel[grid](
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        offsets,
        total_M,
        N,
        K,
        G,
        max_m_per_group,
        XQ.stride(0),
        XQ.stride(1),
        WQ.stride(0),
        WQ.stride(2),
        WQ.stride(1),  # <-- swapped
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        output.stride(0),
        output.stride(1),
        **_HIP_OPTS,
    )
    return output


# -----------------------------------------------------------------------------
# Op signature matching csrc/gemm/cutlass/f4f4bf16.cu::f4f4bf16
# -----------------------------------------------------------------------------


def f4f4bf16(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    mxfp4_block_size: int = 32,
) -> torch.Tensor:
    """Triton ROCm implementation of mslk::f4f4bf16 (gfx950 native MXFP4 only)."""
    if global_scale is not None:
        raise RuntimeError(
            "NVFP4 is not supported on ROCm. Use native MXFP4 "
            "(mxfp4_block_size=32, global_scale=None) on gfx950."
        )
    if mxfp4_block_size != 32:
        raise RuntimeError(
            "ROCm f4f4bf16 only supports standard MXFP4 (mxfp4_block_size=32). "
            f"MXFP4_16 and NVFP4 are CUDA-only; got mxfp4_block_size={mxfp4_block_size}."
        )
    return mxfp4_gemm(XQ, WQ, x_scale, w_scale, output=output)
