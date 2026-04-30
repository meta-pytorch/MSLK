# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# ROCm INT8 GEMM via Triton (BACKLOG-G1 / MSLK-G1)
#
# Implements:
#   i8i8bf16         – static scale variant:  Y = XQ @ WQ.T * scale
#   i8i8bf16_dynamic – dynamic scale variant: Y = XQ @ WQ.T * scale (scale is a tensor)
#
# Kernel design follows torchao's intmm_triton.py
# (pytorch/ao → torchao/kernel/intmm_triton.py,
#  matmul_kernel_with_block_pointers / scaled_matmul_kernel_with_block_pointers)
# adapted to MSLK's dispatch / autotune conventions (fp8_gemm.py / grouped_gemm.py).
#
# Layout: XQ [M, K] row-major, WQ [N, K] row-major (transposed in kernel), Y [M, N].
# Accumulator: int32.  Output: bfloat16.

import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton import Config  # @manual


# ---------------------------------------------------------------------------
# Autotune configs
# ---------------------------------------------------------------------------

# NV configs – same shapes already noted as "good for int8" in fp8_gemm.py
_NV_CONFIGS = [
    Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=3, num_warps=8),
    Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
    Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 128}, num_stages=4, num_warps=4),
    Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=4, num_warps=4),
    Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4),
    Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_stages=4, num_warps=4),
    Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=4, num_warps=4),
    Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 64},  num_stages=4, num_warps=4),
    Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 64},  num_stages=5, num_warps=2),
    # smaller-M / decode shapes
    Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_stages=4, num_warps=4),
    Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_stages=5, num_warps=2),
    Config({"BLOCK_M": 16,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_stages=5, num_warps=2),
]

# AMD configs – tuned for CDNA3 (MI300X) / CDNA3+ (MI350X, gfx950).
# waves_per_eu and matrix_instr_nonkdim follow the grouped_gemm.py AMD pattern.
# BLOCK_K >= 64 recommended on CDNA3+ to utilise the larger LDS.
_AMD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [64, 128, 256]
    for block_n in [64, 128, 256]
    for block_k in [64, 128]
    for num_stages in [1, 2]
    for num_warps, waves_per_eu in [(4, 1), (8, 2), (16, 4)]
    for matrix_instr_nonkdim in [16]
]


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["m_key", "n_key", "k_key"],
)
@triton.jit
def _kernel_int8_gemm(
    # Pointers
    A_ptr,          # [M, K] int8
    B_ptr,          # [N, K] int8  (stored row-major; loaded transposed)
    C_ptr,          # [M, N] bf16
    # Scalar scale (used only when USE_TENSOR_SCALE=False)
    scale_val,
    # Tensor scale pointer (used only when USE_TENSOR_SCALE=True)
    scale_ptr,
    # Problem size
    M,
    N,
    K,
    # Autotune keys (bucketed)
    m_key,
    n_key,
    k_key,
    # Strides
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    # Tile sizes (autotuned)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Whether scale is a device tensor (dynamic) or a compile-time float
    USE_TENSOR_SCALE: tl.constexpr,
) -> None:
    """INT8 × INT8 → BF16 GEMM kernel.

    Computes  C = (A @ B.T) * scale
    where A [M,K] and B [N,K] are int8 row-major,
    accumulation is int32, and the output is bfloat16.

    Follows torchao intmm_triton.py's block-pointer approach.
    scale is either a compile-time float (static) or a float32 scalar tensor (dynamic).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Bounds masks for M / N (K handled per-iteration)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Base pointers for the first tile in K
    A = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    # Accumulate in int32 (matches CUTLASS i8i8bf16 accumulator type)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining

        a = tl.load(A, mask=mask_m[:, None] & k_mask[None, :], other=0)
        b = tl.load(B, mask=mask_n[:, None] & k_mask[None, :], other=0)

        # tl.dot requires the rhs to be [K, N]; transpose b tile in-register.
        acc += tl.dot(a, b.T, out_dtype=tl.int32)

        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # Apply scale and convert to bfloat16
    acc_fp = acc.to(tl.float32)
    if USE_TENSOR_SCALE:
        scale = tl.load(scale_ptr).to(tl.float32)
    else:
        scale = scale_val.to(tl.float32)
    acc_fp = acc_fp * scale

    # Write output
    C = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C, acc_fp.to(tl.bfloat16), mask=out_mask)


# ---------------------------------------------------------------------------
# Python-level dispatch helpers
# ---------------------------------------------------------------------------

def _get_matmul_tune(M: int, N: int, K: int):
    """Bucketed autotune keys – mirrors fp8_gemm.get_matmul_tune."""
    m_key = M if M < 256 else 256 + M // 1024
    return m_key, N, K


def _launch_int8_gemm(
    XQ: torch.Tensor,   # [M, K] int8
    WQ: torch.Tensor,   # [N, K] int8
    scale_val: float,
    scale_tensor: torch.Tensor,  # ignored when use_tensor_scale=False
    use_tensor_scale: bool,
) -> torch.Tensor:
    assert XQ.dtype == torch.int8, f"XQ must be int8, got {XQ.dtype}"
    assert WQ.dtype == torch.int8, f"WQ must be int8, got {WQ.dtype}"
    assert XQ.is_contiguous(), "XQ must be contiguous"
    assert WQ.is_contiguous(), "WQ must be contiguous"
    assert XQ.shape[1] == WQ.shape[1], (
        f"K dimension mismatch: XQ {XQ.shape}, WQ {WQ.shape}"
    )

    M, K = XQ.shape
    N = WQ.shape[0]
    m_key, n_key, k_key = _get_matmul_tune(M, N, K)

    Y = torch.empty((M, N), device=XQ.device, dtype=torch.bfloat16)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _kernel_int8_gemm[grid](
        XQ,
        WQ,
        Y,
        scale_val,
        scale_tensor,
        M,
        N,
        K,
        m_key,
        n_key,
        k_key,
        XQ.stride(0),
        XQ.stride(1),
        WQ.stride(0),
        WQ.stride(1),
        Y.stride(0),
        Y.stride(1),
        USE_TENSOR_SCALE=use_tensor_scale,
    )
    return Y


# ---------------------------------------------------------------------------
# Public API — called by the ROCm op dispatch shim in gemm_ops.cpp
# ---------------------------------------------------------------------------

def i8i8bf16_triton(
    XQ: torch.Tensor,   # [M, K] int8
    WQ: torch.Tensor,   # [N, K] int8
    scale: float,       # single global scale
    split_k: int = 1,   # accepted for API parity with CUDA path; not used
) -> torch.Tensor:
    """ROCm INT8 GEMM with a static (compile-time) global scale.

    Matches the CUDA signature:
        i8i8bf16(Tensor XQ, Tensor WQ, float scale, int split_k=1) -> Tensor
    """
    dummy_scale_tensor = torch.empty((), dtype=torch.float32, device=XQ.device)
    return _launch_int8_gemm(XQ, WQ, scale, dummy_scale_tensor, use_tensor_scale=False)


def i8i8bf16_dynamic_triton(
    XQ: torch.Tensor,   # [M, K] int8
    WQ: torch.Tensor,   # [N, K] int8
    scale: torch.Tensor,  # float32 scalar tensor (device)
    split_k: int = 1,   # accepted for API parity with CUDA path; not used
) -> torch.Tensor:
    """ROCm INT8 GEMM with a dynamic (device-tensor) global scale.

    Matches the CUDA signature:
        i8i8bf16_dynamic(Tensor XQ, Tensor WQ, Tensor scale, int split_k=1) -> Tensor
    """
    assert scale.dtype == torch.float32, f"scale must be float32, got {scale.dtype}"
    return _launch_int8_gemm(XQ, WQ, 0.0, scale, use_tensor_scale=True)


# ---------------------------------------------------------------------------
# torch.library.impl registration — wire torch.ops.mslk.i8i8bf16[_dynamic]
# to the Triton functions on ROCm.  These registrations are only active when
# the mslk C++ library has been loaded (i.e. at runtime, not during import).
# ---------------------------------------------------------------------------

def _register_rocm_ops() -> None:
    """Register Python-dispatch impls for the ROCm i8i8 ops.

    Called after the mslk C++ library is loaded so that
    torch.ops.mslk is populated before we try to impl against it.
    """
    import torch.library  # noqa: F401

    @torch.library.impl("mslk::i8i8bf16", "CUDA")
    def _i8i8bf16_impl(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        scale: float,
        split_k: int = 1,
    ) -> torch.Tensor:
        return i8i8bf16_triton(XQ, WQ, scale, split_k)

    @torch.library.impl("mslk::i8i8bf16_dynamic", "CUDA")
    def _i8i8bf16_dynamic_impl(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        scale: torch.Tensor,
        split_k: int = 1,
    ) -> torch.Tensor:
        return i8i8bf16_dynamic_triton(XQ, WQ, scale, split_k)


# Register immediately — the C++ library loads the schema during import of
# mslk.gemm, so by the time user code imports from here the ops exist.
if torch.version.hip is not None:
    try:
        _register_rocm_ops()
    except Exception:
        # If mslk C++ library isn't loaded yet (e.g. unit-test environments
        # that stub the schema), registration is deferred to the caller.
        pass
