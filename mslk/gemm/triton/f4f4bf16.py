# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
Triton f4f4bf16 GEMM for AMD ROCm (gfx950 / MI350x).

Two paths in one file, both consuming the same `f4f4bf16` op signature used by
the CUTLASS SM100 kernel in ``csrc/gemm/cutlass/f4f4bf16.cu``:

- ``mxfp4_gemm``: native MXFP4 × MXFP4 → BF16 using ``tl.dot_scaled`` with
  ``lhs_format="e2m1"``, ``rhs_format="e2m1"``, ``scale_format="e8m0"``,
  block_size=32. Targets gfx950 (MI350x) ``v_mfma_scale_f32_*_f8f6f4``.

- ``nvfp4_dequant_gemm``: NVFP4 (block_size=16, E4M3 scales + per-tensor FP32
  global_scale) → dequant to arch-native fp8 (block scale folded in) once into
  HBM → tuned fp8 MFMA GEMM. NVFP4's 16-element E4M3 block scales are not
  compatible with the native MXFP4 scaled MFMA (32-element E8M0), so NVFP4 uses
  this path on both gfx942 (MI300X) and gfx950 (MI350X).

A third entry point, ``mxfp4_dequant_gemm``, runs the MXFP4 path on hardware
without native FP4 MFMA (gfx942) through the same pre-dequant→fp8 path. Both
share ``_fp4_dequant_to_fp8_kernel`` (switched by an ``IS_NVFP4`` flag) feeding
an arch-native fp8 GEMM — converting each FP4 element exactly once rather than
re-dequantizing per output tile.
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
                    configs.append(triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm},
                        num_warps=nw, num_stages=ns,
                    ))
    return configs


@triton.autotune(configs=_mxfp4_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _mxfp4_gemm_kernel(
    # Pointers
    a_ptr,             # [M, K//2] uint8
    b_ptr,             # [N, K//2] uint8
    a_scale_ptr,       # [M, K//32] uint8 (E8M0)
    b_scale_ptr,       # [N, K//32] uint8 (E8M0)
    c_ptr,             # [M, N] bf16
    # Sizes
    M, N, K,
    # Strides (in elements/bytes for the respective tensor)
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    stride_cm, stride_cn,
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
    as_base = a_scale_ptr + safe_m[:, None] * stride_asm + offs_k_scale[None, :] * stride_ask
    bs_base = b_scale_ptr + safe_n[:, None] * stride_bsn + offs_k_scale[None, :] * stride_bsk

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
            a_pack, a_scale, "e2m1",
            b_pack, b_scale, "e2m1",
            acc=acc,
            out_dtype=tl.float32,
        )

    c = acc.to(tl.bfloat16)
    c_base = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_base, c, mask=mask)


def mxfp4_gemm(
    XQ: torch.Tensor,         # [M, K//2] uint8
    WQ: torch.Tensor,         # [N, K//2] uint8
    x_scale: torch.Tensor,    # [M, K//32] uint8 (E8M0)
    w_scale: torch.Tensor,    # [N, K//32] uint8 (E8M0)
    output: Optional[torch.Tensor] = None,
    dequant_dtype: str = "fp8",
) -> torch.Tensor:
    """MXFP4 × MXFP4 → BF16.

    On gfx950 / SM100 uses native ``tl.dot_scaled`` MFMA — ``dequant_dtype``
    is **ignored** on this path. On gfx942 (no native FP4 hardware) falls
    back to in-kernel dequant → FP8/BF16 MFMA (controlled by
    ``dequant_dtype``: ``"fp8"`` default for ~2× throughput, ``"bf16"`` for
    accuracy debugging). The kwarg name and default are kept in sync with
    :func:`nvfp4_dequant_gemm` so callers can swap paths without changing
    arguments.

    """
    if not _has_native_mxfp4_mfma():
        return mxfp4_dequant_gemm(
            XQ, WQ, x_scale, w_scale, output=output, dequant_dtype=dequant_dtype,
        )
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

    # autotune picks BLOCK_M/N/K/GROUP_M; grid is computed per-config via a lambda.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _mxfp4_gemm_kernel[grid](
        XQ, WQ, x_scale, w_scale, output,
        M, N, K,
        XQ.stride(0), XQ.stride(1),
        WQ.stride(0), WQ.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        output.stride(0), output.stride(1),
        **_HIP_OPTS,
    )
    return output


# -----------------------------------------------------------------------------
# NVFP4 dequant-to-BF16 kernel
#
# Required because:
#   - gfx942 (MI300X) has NO native FP4 hardware — this is the only FP4 path.
#   - gfx950 (MI350X) has native MXFP4 but NOT NVFP4 (incompatible block size
#     16 vs 32 and scale format E4M3 vs E8M0); use this when consuming an
#     NVFP4-quantized checkpoint without re-quantizing to MXFP4 first.
#
# Algorithm per inner-loop iteration:
#   1. Load packed [BLOCK_M, BLOCK_K//2] uint8 for A, [BLOCK_K//2, BLOCK_N] for B.
#   2. Unpack each byte → two E2M1 codes → fp32 via _e2m1_decode (LUT-based).
#   3. Load [BLOCK_M, BLOCK_K//16] FP8 (E4M3) scales; reinterpret as float8,
#      cast to fp32, divide by global_scale (NVFP4 convention: stored scale
#      already includes global_scale, so dequant is value * scale / global).
#   4. Broadcast scales across 16 K-elements, multiply unpacked codes.
#   5. Cast tiles to bf16, accumulate via tl.dot (BF16 MFMA on AMD).
#
# -----------------------------------------------------------------------------


@triton.jit
def _e2m1_decode(x_uint4):
    """Decode a 4-bit E2M1 code (uint8 in [0, 15]) to fp32.

    E2M1 layout: bit3 = sign, bits[2:1] = exp, bit0 = mantissa.
    Representable values: {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.

    Bit-twiddle decode: build the fp16 bit pattern directly and bitcast, rather
    than a fp32 `tl.where` ladder. E2M1 is sign(1)|exp(2)|mant(1); fp16 is
    sign(1)|exp(5)|mant(10).
      - Normal (exp >= 1): |v| = 2^(exp-1)*(1+mant/2). In fp16 that is biased
        exp = (exp-1)+15 = exp+14 with the single mant bit in fp16's mantissa
        MSB (bit 9):  mag_bits = ((exp+14) << 10) | (mant << 9).
      - Subnormal (exp == 0): only {0 -> +0.0, 1 -> +0.5}; 0.5 is fp16 0x3800,
        so mag_bits = mant * 0x3800. One select handles this case.
      - Sign folds into bit 15 (no select needed).
    Fewer VALU ops than the old fp32 `tl.where` ladder (drops the cndmask / cmp
    / fp-mul chain that dominated). Shared by both dequant paths, so NVFP4 and
    MXFP4-on-gfx942 both benefit. Verified exact (incl. signed zero) on all 16
    E2M1 codes.

    Uses only generic int ops + one fp16 bitcast (no arch intrinsics), so it is
    portable to gfx942. The earlier `tl.exp2((exp-1).to(fp32))` form crashed the
    ROCm 7.2.4 LLVM SDAG backend; this rewrite avoids exp2 entirely.

    Verified against the reference table:
      code  sign exp mant  |val|  signed
      0000  +    00  0     0.0    +0.0
      0001  +    00  1     0.5    +0.5
      0010  +    01  0     1.0    +1.0
      0011  +    01  1     1.5    +1.5
      0100  +    10  0     2.0    +2.0
      0101  +    10  1     3.0    +3.0
      0110  +    11  0     4.0    +4.0
      0111  +    11  1     6.0    +6.0
      1xxx  same magnitudes, negated.
    """
    x = x_uint4.to(tl.int32)
    sign = (x >> 3) & 0x1
    exp = (x >> 1) & 0x3
    mant = x & 0x1
    # Normal: fp16 magnitude bits = ((exp+14)<<10) | (mant<<9).
    normal_mag = ((exp + 14) << 10) | (mant << 9)
    # Subnormal (exp==0): {0 -> 0x0000 (+0.0), 1 -> 0x3800 (+0.5)}.
    sub_mag = mant * 0x3800
    mag = tl.where(exp == 0, sub_mag, normal_mag)
    bits = (sign << 15) | mag
    return bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


def nvfp4_dequant_gemm(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,         # [M, K//16] uint8 viewed as FP8 E4M3
    w_scale: torch.Tensor,         # [N, K//16] uint8
    global_scale: torch.Tensor,    # either () fp32 (shared) OR (2,) fp32 [a_global, b_global]
    output: Optional[torch.Tensor] = None,
    dequant_dtype: str = "fp8",
) -> torch.Tensor:
    """NVFP4 × NVFP4 → BF16 via pre-dequant to fp8 + tuned fp8 MFMA GEMM.

    Works on any GPU with FP8 MFMA (gfx942 / gfx950 / SM90+). Uses no native FP4
    instructions — this is the path that lets NVFP4 checkpoints run on AMD.
    (NVFP4's 16-element E4M3 block scales aren't compatible with gfx950's native
    MXFP4 MFMA, which wants 32-element E8M0, so even on MI350 NVFP4 goes through
    dequant rather than native.)

    FP4 is dequantized to arch-native fp8 (block scale folded in) once into HBM,
    then a tuned fp8 GEMM runs — each element converted exactly once, at fp8-MFMA
    rate. fp8 ⊃ fp4 so the conversion is lossless for the element; total error
    matches the original NVFP4 quantization.

    Scale layout MUST be plain ``[M, K//16]`` uint8 (NOT the TCGen5-swizzled
    layout produced by triton_quantize_nvfp4 — that helper is Blackwell-only).
    Use ``triton_quantize_nvfp4_amd`` to produce inputs.

    ``global_scale`` may be a scalar (both A and B share it) or a 2-element
    tensor ``[a_global, b_global]`` for the typical case where activations and
    weights have separate per-tensor scales.

    ``dequant_dtype`` is accepted for signature compatibility but ignored — the
    path always dequants to arch-native fp8.
    """
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
    assert K % 32 == 0, f"K={K} must be a multiple of 32 (FP4 packing + 16-elem scale block)"
    assert x_scale.shape == (M, K // 16), f"x_scale {tuple(x_scale.shape)} != ({M}, {K//16})"
    assert w_scale.shape == (N, K // 16), f"w_scale {tuple(w_scale.shape)} != ({N}, {K//16})"

    # NVFP4 convention: stored per-block scale already absorbs global_scale, so
    # dequant uses (scale / global_scale). Pre-compute reciprocals as scalars
    # so the kernel hot path stays multiply-only.
    gs = global_scale.to(torch.float32).reshape(-1)
    if gs.numel() == 1:
        inv_a = inv_b = (1.0 / gs).item()
    elif gs.numel() == 2:
        inv_a = (1.0 / gs[0]).item()
        inv_b = (1.0 / gs[1]).item()
    else:
        raise ValueError(f"global_scale must be scalar or 2-element, got numel={gs.numel()}")

    # Pre-dequant to arch-native fp8 in HBM + tuned fp8 GEMM. Converts each FP4
    # element once (vs grid_m/grid_n times in a fused loop) and runs at fp8-MFMA
    # rate — several× faster on both gfx942 and gfx950 (NVFP4 has no native MFMA
    # on either part). See _predequant_fp8_gemm.
    return _predequant_fp8_gemm(
        XQ, WQ, x_scale, w_scale, inv_a, inv_b, M, N, K, True, output,
    )


# -----------------------------------------------------------------------------
# MXFP4 dequant wrapper — gfx942 (MI300X) MXFP4 path
#
# MI300X has BF16/FP8 MFMA but NO native FP4 hardware (no `tl.dot_scaled`
# lowering). For MXFP4-quantized inputs on gfx942 we reuse the shared
# pre-dequant path (`_fp4_dequant_to_fp8_kernel` with `IS_NVFP4=0`): decode FP4
# → fp8 with the E8M0 scale folded in, then a tuned fp8 GEMM.
# -----------------------------------------------------------------------------


def mxfp4_dequant_gemm(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    dequant_dtype: str = "fp8",
) -> torch.Tensor:
    """MXFP4 × MXFP4 → BF16 via pre-dequant to fp8 + tuned fp8 MFMA GEMM.

    Use this on hardware without native FP4 MFMA (gfx942 / MI300X). On gfx950
    prefer :func:`mxfp4_gemm` (native scaled MFMA, ~3-4x faster). MXFP4's E8M0
    scale is a power of two, so the fp8 conversion is exact. ``dequant_dtype`` is
    accepted for signature compatibility but ignored (always arch-native fp8).
    """
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
    assert x_scale.shape == (M, K // 32)
    assert w_scale.shape == (N, K // 32)

    # Pre-dequant to arch-native fp8 + tuned fp8 GEMM. MXFP4 has no per-tensor
    # global scale, so inv_*_global = 1.0.
    return _predequant_fp8_gemm(
        XQ, WQ, x_scale, w_scale, 1.0, 1.0, M, N, K, False, output,
    )


def _has_native_mxfp4_mfma() -> bool:
    """True iff `tl.dot_scaled(e2m1, e2m1, e8m0)` has a native MFMA lowering
    on the current device. That's gfx950 (MI350x) today; gfx942 (MI300x) must
    use the dequant→FP8 fallback path."""
    if not torch.cuda.is_available():
        return False
    if torch.version.hip is None:
        return True  # CUDA SM100 has native dot_scaled
    arch = (getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or "").lower()
    return "gfx950" in arch


def _use_fnuz_fp8() -> bool:
    """True iff the device's native FP8 is E4M3FNUZ (gfx942 / MI300X), False for
    OCP E4M3FN (gfx950 / MI350X, CUDA). Picks the FP8 type the matrix core runs
    at full rate — using the wrong one upcasts to fp16 and erases the speedup."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    arch = (getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or "").lower()
    return "gfx942" in arch


# -----------------------------------------------------------------------------
# Pre-dequant FP4 path: convert FP4 → FP8 (block scale folded in) once into HBM,
# then run a tuned FP8 MFMA GEMM. Each FP4 element is converted exactly once
# instead of grid_m/grid_n times as in a fused dequant loop — far faster on any
# arch without native FP4 MFMA (gfx942) and even on gfx950 for NVFP4 (no native
# NVFP4 MFMA). The FP8 conversion is lossless for the element (FP8 ⊃ FP4) and
# adds no measurable error vs the fused path. FP8 type is arch-native
# (E4M3FNUZ on gfx942, OCP E4M3FN on gfx950).
# -----------------------------------------------------------------------------


@triton.jit
def _fp4_dequant_to_fp8_kernel(
    a_ptr,                 # [R, K//2] uint8 packed E2M1
    a_scale_ptr,           # [R, K//SCALE_GROUP] uint8 (E4M3 if NVFP4, E8M0 if MXFP4)
    inv_global,            # fp32 = 1/global_scale (NVFP4); 1.0 for MXFP4
    out_ptr,               # [R, K] fp8 (e4m3fnuz or e4m3fn per FNUZ flag)
    R, K,
    stride_ar, stride_ak,
    stride_sr, stride_sk,
    stride_or, stride_ok,
    IS_NVFP4: tl.constexpr,
    FNUZ: tl.constexpr,      # 1 = e4m3fnuz (gfx942), 0 = OCP e4m3fn (gfx950)
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,   # unpacked K elements per tile (multiple of SCALE_GROUP)
):
    """Dequant one [BLOCK_R, BLOCK_K] FP4 tile to fp8, scale folded in.

    out[r,k] = e2m1_decode(nibble) * block_scale  (scale already /global for
    NVFP4). The downstream FP8 GEMM runs with unit row-scales, so the arithmetic
    matches the old fused kernel. Each FP4 element is touched exactly once.
    """
    SCALE_GROUP: tl.constexpr = 16 if IS_NVFP4 else 32
    pid_r = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    kcols = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)   # packed byte cols
    r_mask = rows < R
    k_mask = kcols < (K // 2)
    a = tl.load(
        a_ptr + rows[:, None] * stride_ar + kcols[None, :] * stride_ak,
        mask=r_mask[:, None] & k_mask[None, :], other=0,
    )
    lo = _e2m1_decode(a & 0xF)
    hi = _e2m1_decode((a >> 4) & 0xF)
    # Byte col c covers elements 2c (lo) and 2c+1 (hi); both share scale group
    # floor(2c / SCALE_GROUP) == floor(c / (SCALE_GROUP//2)).
    scol = (kcols * 2) // SCALE_GROUP
    s_u8 = tl.load(
        a_scale_ptr + rows[:, None] * stride_sr + scol[None, :] * stride_sk,
        mask=r_mask[:, None], other=0,
    )
    if IS_NVFP4:
        s = s_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32) * inv_global
    else:
        is_nan = s_u8 == 0xFF
        s = tl.where(is_nan, 0.0, tl.exp2((s_u8.to(tl.int32) - 127).to(tl.float32)))
    lo = lo * s
    hi = hi * s
    if FNUZ:
        lo = lo.to(tl.float8e4b8)     # E4M3FNUZ — native on gfx942
        hi = hi.to(tl.float8e4b8)
    else:
        lo = lo.to(tl.float8e4nv)     # OCP E4M3FN — native on gfx950
        hi = hi.to(tl.float8e4nv)
    out = tl.interleave(lo, hi)       # back to [BLOCK_R, BLOCK_K] element order
    ocols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    o_mask = ocols < K
    tl.store(
        out_ptr + rows[:, None] * stride_or + ocols[None, :] * stride_ok,
        out, mask=r_mask[:, None] & o_mask[None, :],
    )


def _fp4_dequant_to_fp8(XQ, scale, inv_global, R, K, is_nvfp4, fnuz):
    """FP4 [R, K//2] + per-block scale → fp8 [R, K] (scale folded in)."""
    out_dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    out = torch.empty((R, K), dtype=out_dtype, device=XQ.device)
    BLOCK_R, BLOCK_K = 64, 256
    grid = (triton.cdiv(R, BLOCK_R), triton.cdiv(K, BLOCK_K))
    _fp4_dequant_to_fp8_kernel[grid](
        XQ, scale, inv_global, out, R, K,
        XQ.stride(0), XQ.stride(1),
        scale.stride(0), scale.stride(1),
        out.stride(0), out.stride(1),
        IS_NVFP4=1 if is_nvfp4 else 0, FNUZ=1 if fnuz else 0,
        BLOCK_R=BLOCK_R, BLOCK_K=BLOCK_K,
    )
    return out


def _ocp_fp8_gemm_configs():
    configs = []
    for bm, bn, bk, nw, ns in [
        (128, 256, 128, 8, 2), (256, 128, 128, 8, 2), (128, 128, 128, 8, 2),
        (256, 256, 128, 8, 2), (128, 256, 64, 8, 2), (64, 128, 128, 4, 2),
        (128, 64, 128, 4, 2), (256, 256, 64, 8, 2),
    ]:
        configs.append(triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
            num_warps=nw, num_stages=ns,
        ))
    return configs


@triton.autotune(configs=_ocp_fp8_gemm_configs(), key=["M", "N", "K"])
@triton.jit
def _ocp_fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C[M,N] = A[M,K] @ B[N,K]^T for OCP e4m3fn fp8 inputs, bf16 out.

    gfx950 only — `matmul_fp8_row` forces E4M3FNUZ (upcasts to fp16 on gfx950),
    so we run a plain OCP-fp8 MFMA GEMM here for the gfx950 pre-dequant path.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * num_pid_n
    group_id = pid // width
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % width) % group_size_m)
    pid_n = (pid % width) // group_size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    n_mask = rn < N
    # Clamp row/col indices used for addressing so over-hang lanes (when
    # BLOCK_M/N don't divide M/N) read an in-bounds element instead of running
    # off the allocation; the load masks below zero them out. Without this,
    # small-M × large-K shapes fault (the row stride pushes the over-read past
    # mapped pages).
    rm_safe = tl.where(m_mask, rm, 0)
    rn_safe = tl.where(n_mask, rn, 0)
    a_ptrs = a_ptr + rm_safe[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rn_safe[None, :] * stride_bn + rk[:, None] * stride_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = rk < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.bfloat16)
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        c_ptr + cm[:, None] * stride_cm + cn[None, :] * stride_cn, c,
        mask=(cm[:, None] < M) & (cn[None, :] < N),
    )


def _ocp_fp8_gemm(a8, b8, output):
    """A[M,K] @ B[N,K]^T for OCP e4m3fn fp8 → bf16 [M,N] (gfx950)."""
    M, K = a8.shape
    N = b8.shape[0]
    out = output if output is not None else torch.empty(
        (M, N), dtype=torch.bfloat16, device=a8.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    # NOTE: do not pass _HIP_OPTS here — matrix_instr_nonkdim=32 aborts MFMA
    # instruction selection for some of this kernel's tile configs (BK=64 /
    # 64-row tiles). The plain fp8 GEMM autotunes fine without it.
    _ocp_fp8_gemm_kernel[grid](
        a8, b8, out, M, N, K,
        a8.stride(0), a8.stride(1), b8.stride(0), b8.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def _predequant_fp8_gemm(XQ, WQ, x_scale, w_scale, inv_a, inv_b, M, N, K, is_nvfp4, output):
    """Dequant both operands to arch-native fp8 in HBM (block scale folded in),
    then a tuned fp8 MFMA GEMM with unit scales (math matches the fused path).

    gfx942: E4M3FNUZ + MSLK's `matmul_fp8_row`. gfx950: OCP E4M3FN + the local
    `_ocp_fp8_gemm` (matmul_fp8_row is FNUZ-only and would upcast to fp16)."""
    fnuz = _use_fnuz_fp8()
    a8 = _fp4_dequant_to_fp8(XQ, x_scale, inv_a, M, K, is_nvfp4, fnuz)
    b8 = _fp4_dequant_to_fp8(WQ, w_scale, inv_b, N, K, is_nvfp4, fnuz)
    if fnuz:
        from mslk.gemm.triton.fp8_gemm import matmul_fp8_row

        ones_a = torch.ones(M, dtype=torch.float32, device=XQ.device)
        ones_b = torch.ones(N, dtype=torch.float32, device=XQ.device)
        out = matmul_fp8_row(a8, b8, ones_a, ones_b)
        if output is not None:
            output.copy_(out)
            return output
        return out.to(torch.bfloat16)
    return _ocp_fp8_gemm(a8, b8, output)


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
        configs.append(triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm},
            num_warps=nw, num_stages=ns,
        ))
    return configs


@triton.autotune(configs=_mxfp4_grouped_autotune_configs(), key=["MAX_M_PER_GROUP", "N", "K", "G"])
@triton.jit
def _mxfp4_grouped_mm_kernel(
    # Pointers
    a_ptr,             # [total_M, K//2] uint8
    b_ptr,             # [G, K//2, N] uint8 (caller passed transpose-of-[G,N,K//2])
    a_scale_ptr,       # [total_M, K//32] uint8
    b_scale_ptr,       # [G, N, K//32] uint8
    c_ptr,             # [total_M, N] bf16
    offsets_ptr,       # [G] int32, cumulative M ends
    # Sizes
    total_M, N, K, G,
    MAX_M_PER_GROUP,   # autotune key only (not used inside)
    # Strides
    stride_am, stride_ak,
    stride_bg, stride_bk, stride_bn,
    stride_asm, stride_ask,
    stride_bsg, stride_bsn, stride_bsk,
    stride_cm, stride_cn,
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
    pid = tl.program_id(0)    # tile id within this expert
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
    a_base = a_ptr + offs_m_global[:, None] * stride_am + offs_k_packed[None, :] * stride_ak
    # B: [G, K//2, N], stored with caller-supplied .transpose(-2,-1) so the
    # K-major view here is what tl.dot_scaled rhs expects.
    b_base = (
        b_ptr + pid_g * stride_bg
        + offs_k_packed[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    as_base = (
        a_scale_ptr
        + offs_m_global[:, None] * stride_asm
        + offs_k_scale[None, :] * stride_ask
    )
    bs_base = (
        b_scale_ptr + pid_g * stride_bsg
        + offs_n[:, None] * stride_bsn
        + offs_k_scale[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    K_packed = K // 2
    for k_off_packed in range(0, K_packed, PACKED_BLOCK_K):
        a_pack = tl.load(a_base + k_off_packed * stride_ak, mask=m_mask[:, None], other=0)
        b_pack = tl.load(b_base + k_off_packed * stride_bk)
        scale_off = (k_off_packed * 2) // 32
        a_scale = tl.load(as_base + scale_off * stride_ask, mask=m_mask[:, None], other=0)
        b_scale = tl.load(bs_base + scale_off * stride_bsk)
        acc = tl.dot_scaled(
            a_pack, a_scale, "e2m1",
            b_pack, b_scale, "e2m1",
            acc=acc,
            out_dtype=tl.float32,
        )

    c = acc.to(tl.bfloat16)
    c_base = c_ptr + offs_m_global[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_base, c, mask=mask)


def mxfp4_grouped_mm(
    XQ: torch.Tensor,                       # [total_M, K//2] uint8
    WQ: torch.Tensor,                       # [G, K//2, N] uint8 (K-major, caller transposed)
    x_scale: torch.Tensor,                  # [total_M, K//32] uint8
    w_scale: torch.Tensor,                  # [G, N, K//32] uint8
    offsets: torch.Tensor,                  # [G] int32, cumulative M ends
    output: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,  # unused in MXFP4 path
) -> torch.Tensor:
    """ROCm Triton implementation of mslk::f4f4bf16_grouped_mm (MXFP4 path)."""
    if XQ.dtype != torch.uint8:
        XQ = XQ.view(torch.uint8)
    if WQ.dtype != torch.uint8:
        WQ = WQ.view(torch.uint8)
    if x_scale.dtype != torch.uint8:
        x_scale = x_scale.view(torch.uint8)
    if w_scale.dtype != torch.uint8:
        w_scale = w_scale.view(torch.uint8)
    assert offsets.dtype == torch.int32 and offsets.dim() == 1

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
    grid = lambda META: (
        triton.cdiv(total_M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        G,
    )

    _mxfp4_grouped_mm_kernel[grid](
        XQ, WQ, x_scale, w_scale, output, offsets,
        total_M, N, K, G, max_m_per_group,
        XQ.stride(0), XQ.stride(1),
        WQ.stride(0), WQ.stride(1), WQ.stride(2),
        x_scale.stride(0), x_scale.stride(1),
        w_scale.stride(0), w_scale.stride(1), w_scale.stride(2),
        output.stride(0), output.stride(1),
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
    XQ: torch.Tensor,                       # [total_M, K//2] uint8 contiguous
    WQ: torch.Tensor,                       # [G, N, K//2] uint8
    x_scale: torch.Tensor,                  # [total_M, K//32] or [G, M, K//32] uint8
    w_scale: torch.Tensor,                  # [G, N, K//32] uint8
    M_sizes: torch.Tensor,                  # [G] int64
    output: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,        # unused (NVFP4 only)
    starting_row_after_padding: Optional[torch.Tensor] = None,  # CUTLASS-specific; ignored on AMD
    use_mx: bool = True,
    offsets_override: Optional[torch.Tensor] = None,    # AMD-only: pre-computed int32 cumulative M ends; skips cumsum
) -> torch.Tensor:
    """ROCm Triton implementation of mslk::f4f4bf16_grouped_stacked (MXFP4 path).


    """
    assert use_mx, "AMD path only supports MXFP4 (use_mx=True). NVFP4 path not implemented."

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

    grid = lambda META: (
        triton.cdiv(total_M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        G,
    )

    # Stride remap: kernel reads B as [G, K//2, N] with (stride_bg, stride_bk, stride_bn).
    # We have [G, N, K//2] with (stride_g, stride_n, stride_k). Re-bind:
    #   stride_bg = WQ.stride(0)
    #   stride_bk = WQ.stride(2)
    #   stride_bn = WQ.stride(1)
    _mxfp4_grouped_mm_kernel[grid](
        XQ, WQ, x_scale, w_scale, output, offsets,
        total_M, N, K, G, max_m_per_group,
        XQ.stride(0), XQ.stride(1),
        WQ.stride(0), WQ.stride(2), WQ.stride(1),  # <-- swapped
        x_scale.stride(0), x_scale.stride(1),
        w_scale.stride(0), w_scale.stride(1), w_scale.stride(2),
        output.stride(0), output.stride(1),
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
    """Triton ROCm implementation of mslk::f4f4bf16.

    Dispatch:
      - ``mxfp4_block_size == 32`` and ``global_scale is None``  → MXFP4 path.
        Routes to native scaled MFMA on gfx950, dequant→FP8 fallback on gfx942.
      - ``mxfp4_block_size == 16`` and ``global_scale is not None`` → NVFP4
        dequant→FP8 path (on both gfx942 and gfx950 — NVFP4's 16-elem E4M3
        block scales aren't compatible with the native MXFP4 MFMA).
    """
    if mxfp4_block_size == 32:
        assert global_scale is None, "MXFP4 path does not use a per-tensor global_scale"
        return mxfp4_gemm(XQ, WQ, x_scale, w_scale, output=output)
    if mxfp4_block_size == 16:
        assert global_scale is not None, "NVFP4 path requires a per-tensor global_scale"
        return nvfp4_dequant_gemm(XQ, WQ, x_scale, w_scale, global_scale, output=output)
    raise ValueError(f"mxfp4_block_size must be 16 (NVFP4) or 32 (MXFP4), got {mxfp4_block_size}")
