# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 blockwise-scaled GEMM via FlyDSL (ROCm implementation of
``mslk::f8f8bf16_blockwise`` and its ``_preshuffle`` sibling).

Replaces the CK ``DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3`` kernel
(``csrc/gemm/ck/fp8_blockwise_gemm.hip``), which was numerically broken on
gfx942/gfx950. The kernel lives in
``mslk.flydsl.kernels.gemm.gemm_blockscale`` and consumes CK's exact block-scale
layout natively (no host-side scale repacking).

Tensor contract (matches the CUTLASS/CK ``f8f8bf16_blockwise`` op):
  XQ      : [..., M, K]           FP8
  WQ      : [N, K]                FP8   (plain row-major for the base op;
                                         MFMA-preshuffled for the _preshuffle op)
  x_scale : [ceil(M/128), K//128] FP32  (ScaleBlockM=ScaleBlockK=128, K-major)
  w_scale : [N//128,      K//128] FP32  (ScaleBlockN=ScaleBlockK=128, K-major)
  Output  : [..., M, N]           BF16

  Y[m, n] = sum_k (XQ[m, k] * x_scale[m // 128, k // 128])
                * (WQ[n, k] * w_scale[n // 128, k // 128])

Two ops share the kernel:

* ``f8f8bf16_blockwise`` -- default. B is plain ``[N, K]`` and is staged
  HBM->LDS->registers (``b_preshuffled=False``), matching the CK contract with no
  weight preshuffle. This is the right choice for a general op: the base op
  receives a plain weight each call, so a per-call host-side preshuffle (a full
  ``O(N*K)`` rearrange) would be pure overhead.
* ``f8f8bf16_blockwise_preshuffle`` -- B already swizzled into the MFMA layout
  (``mslk.quantize.shuffle.preshuffle_b_mfma``) is loaded HBM->registers
  (``b_preshuffled=True``, no B-LDS -> faster). For callers that shuffle the
  weight once at load time and cache it.
"""

import os
from typing import Optional

import torch
from mslk.flydsl.common import require_flydsl
from mslk.flydsl.jit import run_compiled

# Only the scale-block granularity is fixed by the op; tile_m/tile_n/tile_k are
# chosen per call (fixed default, or autotune when MSLK_AUTOTUNE_ENABLE).
_SCALE_BLOCK = 128

# Default tile when autotuning is disabled; divides every 128-aligned N/K and is
# CUDA-graph-capture-safe (compiled once, then replayed).
_DEFAULT_TILE = (128, 128, 128)

# Candidate tiles swept by autotune (tile_n/tile_k pinned to the 128 scale-block
# granularity). Configs that overflow LDS are rejected at compile and skipped.
_AUTOTUNE_TILES = (
    (64, 128, 128),
    (128, 128, 128),
    (256, 128, 128),
    (128, 256, 128),
    (256, 256, 128),
    (64, 256, 128),
)


def _next_pow2(x: int) -> int:
    """Smallest power of two >= x (x >= 1). Buckets M for the autotune key so
    nearby token counts share one tuned config."""
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def _launch_kernel(
    XQ, WQ, x_scale, w_scale, output, *, m, n, k, tile_m, tile_n, tile_k, b_preshuffled
):
    """Compile (cached) and launch the native blockscale GEMM for one tile
    config. Scales are passed through in CK layout with no repacking."""
    from mslk.flydsl.kernels.gemm.gemm_blockscale import compile_fp8_blockwise_gemm

    launcher = compile_fp8_blockwise_gemm(
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_m=_SCALE_BLOCK,
        scale_block_n=_SCALE_BLOCK,
        scale_block_k=_SCALE_BLOCK,
        out_dtype="bf16",
        b_preshuffled=b_preshuffled,
    )
    run_compiled(
        launcher,
        output,
        XQ.contiguous().view(torch.int8),
        WQ.contiguous().view(torch.int8),
        x_scale.contiguous(),
        w_scale.contiguous(),
        m,
        n,
        k,
        torch.cuda.current_stream(),
    )
    return output


# Tile autotune (opt-in via MSLK_AUTOTUNE_ENABLE): select the best tile once per
# shape bucket, cache it, then always take the low-overhead direct-launch path.
_TILE_CACHE: dict = {}


def _bench_tile_ms(
    XQ, WQ, x_scale, w_scale, output, *, m, n, k, tile_m, tile_n, tile_k, b_preshuffled
) -> Optional[float]:
    """Compile + time one tile config. Returns ms/iter, or None if the config
    cannot be compiled/run for this shape (e.g. LDS overflow)."""
    try:
        for _ in range(10):  # warmup + first-call compile
            _launch_kernel(
                XQ,
                WQ,
                x_scale,
                w_scale,
                output,
                m=m,
                n=n,
                k=k,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                b_preshuffled=b_preshuffled,
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        rep = 30
        start.record()
        for _ in range(rep):
            _launch_kernel(
                XQ,
                WQ,
                x_scale,
                w_scale,
                output,
                m=m,
                n=n,
                k=k,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                b_preshuffled=b_preshuffled,
            )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / rep
    except Exception:
        return None


def _select_tile(XQ, WQ, x_scale, w_scale, output, m, n, k, b_preshuffled):
    """Pick (and cache) the fastest valid tile for this shape bucket."""
    key = (_next_pow2(m), n, k, b_preshuffled)
    cached = _TILE_CACHE.get(key)
    if cached is not None:
        return cached

    best_tile = None
    best_ms = float("inf")
    for tile_m, tile_n, tile_k in _AUTOTUNE_TILES:
        # tile_n must tile N and tile_k must tile K; tile_m tiles M via masking.
        if n % tile_n != 0 or k % tile_k != 0:
            continue
        ms = _bench_tile_ms(
            XQ,
            WQ,
            x_scale,
            w_scale,
            output,
            m=m,
            n=n,
            k=k,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            b_preshuffled=b_preshuffled,
        )
        if ms is not None and ms < best_ms:
            best_ms = ms
            best_tile = (tile_m, tile_n, tile_k)

    if best_tile is None:
        best_tile = _DEFAULT_TILE
    _TILE_CACHE[key] = best_tile
    return best_tile


# ---- Public dispatch ----


def _dispatch_blockwise(
    XQ, WQ, x_scale, w_scale, block_m, block_n, block_k, *, b_preshuffled
) -> torch.Tensor:
    require_flydsl()

    assert (
        block_m == _SCALE_BLOCK and block_n == _SCALE_BLOCK and block_k == _SCALE_BLOCK
    ), (
        f"Only block size {_SCALE_BLOCK} is supported, got "
        f"({block_m}, {block_n}, {block_k})"
    )
    assert XQ.dim() >= 2, f"XQ must be at least 2D, got {XQ.dim()}D"
    assert WQ.dim() == 2, f"WQ must be [N, K], got {WQ.shape}"
    assert x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32, (
        "Scales must be float32."
    )
    assert XQ.dtype == WQ.dtype, f"XQ/WQ dtype mismatch: {XQ.dtype} vs {WQ.dtype}"

    # Flatten any leading batch dims into M (matches the CK/CUTLASS op).
    lead_shape = tuple(XQ.shape[:-1])
    k = XQ.shape[-1]
    m = 1
    for s in lead_shape:
        m *= s
    n = WQ.shape[0]

    output = torch.empty((*lead_shape, n), dtype=torch.bfloat16, device=XQ.device)
    if m == 0 or n == 0:
        return output
    if k == 0:
        return output.zero_()

    assert WQ.shape[1] == k, f"K mismatch: XQ K={k}, WQ K={WQ.shape[1]}"
    assert n % _SCALE_BLOCK == 0, f"N ({n}) must be a multiple of {_SCALE_BLOCK}"
    assert k % _SCALE_BLOCK == 0, f"K ({k}) must be a multiple of {_SCALE_BLOCK}"

    XQ2d = XQ.reshape(m, k)
    out2d = output.reshape(m, n)

    if os.environ.get("MSLK_AUTOTUNE_ENABLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        tile_m, tile_n, tile_k = _select_tile(
            XQ2d, WQ, x_scale, w_scale, out2d, m, n, k, b_preshuffled
        )
    else:
        tile_m, tile_n, tile_k = _DEFAULT_TILE

    _launch_kernel(
        XQ2d,
        WQ,
        x_scale,
        w_scale,
        out2d,
        m=m,
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        b_preshuffled=b_preshuffled,
    )
    return output


def matmul_f8f8bf16_blockwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """FP8 blockwise-scaled GEMM (FlyDSL), the ROCm impl of
    ``mslk::f8f8bf16_blockwise``. Plain row-major ``WQ`` (no preshuffle)."""
    return _dispatch_blockwise(
        XQ, WQ, x_scale, w_scale, block_m, block_n, block_k, b_preshuffled=False
    )


def matmul_f8f8bf16_blockwise_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """FP8 blockwise-scaled GEMM (FlyDSL), the ROCm impl of
    ``mslk::f8f8bf16_blockwise_preshuffle``. ``WQ`` must already be swizzled into
    the MFMA B layout via ``mslk.quantize.shuffle.preshuffle_b_mfma``; the kernel
    loads it HBM->registers (no B-LDS)."""
    return _dispatch_blockwise(
        XQ, WQ, x_scale, w_scale, block_m, block_n, block_k, b_preshuffled=True
    )


# This module registers nothing: mslk::f8f8bf16_blockwise[_preshuffle] are
# registered (lazily) in mslk/gemm/__init__.py, keeping FlyDSL an opt-in dep.
