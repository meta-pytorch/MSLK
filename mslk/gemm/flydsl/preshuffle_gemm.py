# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL preshuffle GEMM for FP8 rowwise scaling (gfx950).

Provides single and batched FP8 preshuffle GEMM via FlyDSL. Also registers
as the ROCm implementation of the ``mslk`` rowwise FP8 ops on gfx950, and
provides the one ``mslk/gemm/__init__.py`` dispatches the preshuffle ops
(``f8f8{bf16,f16}_rowwise_preshuffle``, CK-preshuffled weights) to.
"""

import weakref
from dataclasses import dataclass
from math import prod
from typing import Optional

import torch
from mslk.utils.device import is_gfx950
from torch import Tensor


# ---------------------------------------------------------------------------
# Tile configuration for the preshuffle GEMM kernel
# ---------------------------------------------------------------------------


@dataclass
class KernelConfig:
    """Tile and launch parameters for one FlyDSL preshuffle GEMM variant."""

    tile_m: int  # M-dimension tile size (rows of activation per workgroup)
    tile_n: int  # N-dimension tile size (columns of weight per workgroup)
    tile_k: int  # K-dimension tile size (reduction loop step)
    lds_stage: int  # number of LDS pipeline stages (software pipelining depth)
    use_cshuffle_epilog: int = 0  # 1 to use cross-lane shuffle in the epilogue
    use_async_copy: int = 0  # 1 to use async global→LDS copy instructions
    waves_per_eu: int = 0  # occupancy hint; 0 = let the compiler decide
    xcd_swizzle: int = 0  # XCD (cross-chiplet die) swizzle pattern index


DEFAULT_CONFIGS_GFX950: dict[int, KernelConfig] = {
    -1: KernelConfig(64, 128, 256, 2, xcd_swizzle=1, waves_per_eu=2),
    -2: KernelConfig(16, 64, 512, 2, waves_per_eu=1),
    -3: KernelConfig(32, 64, 512, 2, xcd_swizzle=1),
    -4: KernelConfig(64, 64, 128, 2, xcd_swizzle=1),
    -5: KernelConfig(128, 256, 128, 2, xcd_swizzle=1, waves_per_eu=2),
    -6: KernelConfig(64, 128, 128, 2, xcd_swizzle=1),
}

# Profile-guided overrides: (m_upper_bound, n, k) -> KernelConfig
# Entries are checked in order; first match where m <= m_upper_bound wins.
# Each (N, K) was swept over M in 1 .. 8192 (every tile_m/tile_n/tile_k that
# computes correctly, x waves_per_eu, xcd_swizzle, async copy, cshuffle
# epilogue), and the last bucket per (N, K) covers every larger M.
_M_ANY = 1 << 62
# fmt: off
_SHAPE_OVERRIDES_GFX950: list[tuple[int, int, int, KernelConfig]] = [
    # N=1024, K=4096: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (1, 1024, 4096, KernelConfig(16, 64, 256, 2, use_async_copy=1, waves_per_eu=2)),
    (16, 1024, 4096, KernelConfig(16, 64, 256, 2, use_async_copy=1)),
    (32, 1024, 4096, KernelConfig(16, 64, 256, 2, use_async_copy=1, waves_per_eu=2)),
    (64, 1024, 4096, KernelConfig(16, 64, 256, 2, waves_per_eu=2)),
    (128, 1024, 4096, KernelConfig(32, 64, 512, 2, xcd_swizzle=1)),
    (256, 1024, 4096, KernelConfig(64, 64, 256, 2, use_async_copy=1, waves_per_eu=1)),
    (512, 1024, 4096, KernelConfig(64, 64, 256, 2, use_async_copy=1, xcd_swizzle=4)),
    (1024, 1024, 4096, KernelConfig(64, 64, 256, 2)),
    (2048, 1024, 4096, KernelConfig(64, 128, 256, 2, waves_per_eu=2)),
    (4096, 1024, 4096, KernelConfig(64, 128, 128, 2, waves_per_eu=2)),
    (_M_ANY, 1024, 4096, KernelConfig(64, 256, 128, 2)),
    # N=1280, K=8192: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (16, 1280, 8192, KernelConfig(16, 64, 512, 2, use_cshuffle_epilog=1, use_async_copy=1)),
    (32, 1280, 8192, KernelConfig(16, 64, 256, 2)),
    (64, 1280, 8192, KernelConfig(16, 64, 256, 2, waves_per_eu=2, xcd_swizzle=1)),
    (128, 1280, 8192, KernelConfig(32, 64, 512, 2, xcd_swizzle=1)),
    (256, 1280, 8192, KernelConfig(64, 64, 256, 2, xcd_swizzle=1)),
    (512, 1280, 8192, KernelConfig(64, 64, 256, 2, waves_per_eu=2)),
    (1024, 1280, 8192, KernelConfig(64, 128, 512, 2)),
    (2048, 1280, 8192, KernelConfig(64, 128, 128, 2)),
    (6144, 1280, 8192, KernelConfig(128, 128, 128, 2, use_cshuffle_epilog=1, use_async_copy=1, waves_per_eu=2)),
    (_M_ANY, 1280, 8192, KernelConfig(128, 128, 256, 2, use_async_copy=1, waves_per_eu=2, xcd_swizzle=4)),
    # N=2048, K=7168: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (1, 2048, 7168, KernelConfig(16, 64, 512, 2, use_cshuffle_epilog=1, use_async_copy=1)),
    (16, 2048, 7168, KernelConfig(16, 64, 512, 2, use_cshuffle_epilog=1, use_async_copy=1, waves_per_eu=2)),
    (64, 2048, 7168, KernelConfig(16, 64, 256, 2, waves_per_eu=2, xcd_swizzle=1)),
    (128, 2048, 7168, KernelConfig(32, 64, 512, 2, xcd_swizzle=1)),
    (512, 2048, 7168, KernelConfig(64, 64, 256, 2, waves_per_eu=2, xcd_swizzle=1)),
    (1024, 2048, 7168, KernelConfig(64, 128, 256, 2, waves_per_eu=2)),
    (2048, 2048, 7168, KernelConfig(64, 128, 128, 2, waves_per_eu=2, xcd_swizzle=1)),
    (4096, 2048, 7168, KernelConfig(128, 128, 128, 2, use_cshuffle_epilog=1, use_async_copy=1, waves_per_eu=2, xcd_swizzle=4)),
    (_M_ANY, 2048, 7168, KernelConfig(256, 256, 128, 2)),
    # N=4096, K=4096: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (1, 4096, 4096, KernelConfig(16, 64, 256, 2, use_cshuffle_epilog=1, use_async_copy=1)),
    (16, 4096, 4096, KernelConfig(16, 64, 256, 2)),
    (64, 4096, 4096, KernelConfig(16, 64, 256, 2, xcd_swizzle=1)),
    (128, 4096, 4096, KernelConfig(32, 64, 512, 2, xcd_swizzle=1)),
    (256, 4096, 4096, KernelConfig(64, 64, 256, 2, waves_per_eu=2, xcd_swizzle=1)),
    (512, 4096, 4096, KernelConfig(64, 128, 512, 2, waves_per_eu=2, xcd_swizzle=1)),
    (1024, 4096, 4096, KernelConfig(64, 128, 128, 2, xcd_swizzle=1)),
    (2048, 4096, 4096, KernelConfig(128, 128, 128, 2, use_async_copy=1, waves_per_eu=1)),
    (4096, 4096, 4096, KernelConfig(64, 256, 128, 2, waves_per_eu=2)),
    (_M_ANY, 4096, 4096, KernelConfig(64, 256, 128, 2)),
    # N=5120, K=1024: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (1, 5120, 1024, KernelConfig(16, 64, 512, 2, use_cshuffle_epilog=1, waves_per_eu=2)),
    (16, 5120, 1024, KernelConfig(16, 64, 512, 2, use_cshuffle_epilog=1, waves_per_eu=1)),
    (32, 5120, 1024, KernelConfig(16, 64, 512, 2)),
    (128, 5120, 1024, KernelConfig(32, 64, 128, 2, waves_per_eu=2)),
    (256, 5120, 1024, KernelConfig(64, 128, 128, 2, waves_per_eu=2)),
    (1024, 5120, 1024, KernelConfig(64, 128, 128, 2, waves_per_eu=2, xcd_swizzle=1)),
    (2048, 5120, 1024, KernelConfig(64, 128, 256, 2, xcd_swizzle=1)),
    (4096, 5120, 1024, KernelConfig(64, 128, 128, 2)),
    (_M_ANY, 5120, 1024, KernelConfig(64, 256, 128, 2, waves_per_eu=2)),
    # N=7168, K=2048: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (1, 7168, 2048, KernelConfig(16, 64, 256, 2, use_async_copy=1, xcd_swizzle=1)),
    (16, 7168, 2048, KernelConfig(16, 64, 512, 2, waves_per_eu=2)),
    (32, 7168, 2048, KernelConfig(16, 64, 256, 2, waves_per_eu=2, xcd_swizzle=1)),
    (64, 7168, 2048, KernelConfig(16, 128, 256, 2, waves_per_eu=2, xcd_swizzle=1)),
    (128, 7168, 2048, KernelConfig(32, 128, 128, 2, waves_per_eu=2, xcd_swizzle=1)),
    (512, 7168, 2048, KernelConfig(64, 128, 128, 2, waves_per_eu=2, xcd_swizzle=1)),
    (1024, 7168, 2048, KernelConfig(64, 256, 128, 2, xcd_swizzle=1)),
    (_M_ANY, 7168, 2048, KernelConfig(64, 256, 128, 2)),
    # N=7168, K=8192: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (1, 7168, 8192, KernelConfig(16, 64, 512, 2, use_cshuffle_epilog=1, use_async_copy=1, waves_per_eu=2, xcd_swizzle=1)),
    (16, 7168, 8192, KernelConfig(16, 64, 512, 2, use_cshuffle_epilog=1, waves_per_eu=2, xcd_swizzle=1)),
    (32, 7168, 8192, KernelConfig(16, 64, 512, 2, xcd_swizzle=1)),
    (64, 7168, 8192, KernelConfig(32, 64, 512, 2, xcd_swizzle=1)),
    (128, 7168, 8192, KernelConfig(32, 128, 256, 2, use_async_copy=1, xcd_swizzle=1)),
    (256, 7168, 8192, KernelConfig(64, 128, 512, 2, xcd_swizzle=1)),
    (512, 7168, 8192, KernelConfig(64, 128, 128, 2, waves_per_eu=2, xcd_swizzle=1)),
    (1024, 7168, 8192, KernelConfig(128, 128, 128, 2, use_cshuffle_epilog=1, use_async_copy=1, waves_per_eu=2, xcd_swizzle=4)),
    (2048, 7168, 8192, KernelConfig(128, 128, 256, 2, use_async_copy=1, waves_per_eu=2, xcd_swizzle=4)),
    (4096, 7168, 8192, KernelConfig(64, 256, 128, 2)),
    (_M_ANY, 7168, 8192, KernelConfig(128, 256, 128, 2)),
    # N=8192, K=1024: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (1, 8192, 1024, KernelConfig(16, 64, 512, 2, xcd_swizzle=4)),
    (16, 8192, 1024, KernelConfig(16, 64, 512, 2, use_cshuffle_epilog=1, waves_per_eu=1, xcd_swizzle=1)),
    (32, 8192, 1024, KernelConfig(16, 64, 512, 2, xcd_swizzle=1)),
    (128, 8192, 1024, KernelConfig(32, 64, 128, 2, waves_per_eu=2, xcd_swizzle=1)),
    (512, 8192, 1024, KernelConfig(64, 128, 128, 2, xcd_swizzle=1)),
    (1024, 8192, 1024, KernelConfig(64, 256, 128, 2, xcd_swizzle=1)),
    (2048, 8192, 1024, KernelConfig(64, 256, 128, 2, waves_per_eu=2)),
    (_M_ANY, 8192, 1024, KernelConfig(64, 256, 128, 2)),
    # N=8192, K=3584: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (32, 8192, 3584, KernelConfig(16, 64, 256, 2, xcd_swizzle=1)),
    (64, 8192, 3584, KernelConfig(32, 64, 512, 2, waves_per_eu=2, xcd_swizzle=1)),
    (128, 8192, 3584, KernelConfig(32, 64, 256, 2, xcd_swizzle=1)),
    (256, 8192, 3584, KernelConfig(64, 128, 512, 2, waves_per_eu=2, xcd_swizzle=1)),
    (512, 8192, 3584, KernelConfig(64, 128, 128, 2, waves_per_eu=2, xcd_swizzle=1)),
    (_M_ANY, 8192, 3584, KernelConfig(128, 128, 128, 2, use_cshuffle_epilog=1, use_async_copy=1, waves_per_eu=1, xcd_swizzle=4)),
    # N=8192, K=8192: sweep-tuned on gfx950 (FlyDSL 0.4.0, 2026-10-27)
    (1, 8192, 8192, KernelConfig(16, 64, 512, 2, waves_per_eu=1, xcd_swizzle=1)),
    (32, 8192, 8192, KernelConfig(16, 64, 512, 2, xcd_swizzle=1)),
    (64, 8192, 8192, KernelConfig(32, 64, 512, 2, waves_per_eu=2, xcd_swizzle=1)),
    (128, 8192, 8192, KernelConfig(32, 64, 512, 2, use_async_copy=1, xcd_swizzle=1)),
    (256, 8192, 8192, KernelConfig(64, 128, 512, 2, waves_per_eu=2, xcd_swizzle=1)),
    (512, 8192, 8192, KernelConfig(64, 128, 128, 2, xcd_swizzle=1)),
    (1024, 8192, 8192, KernelConfig(128, 128, 128, 2, use_cshuffle_epilog=1, use_async_copy=1, waves_per_eu=2, xcd_swizzle=4)),
    (2048, 8192, 8192, KernelConfig(128, 128, 256, 2, use_async_copy=1, waves_per_eu=2, xcd_swizzle=4)),
    (4096, 8192, 8192, KernelConfig(64, 256, 128, 2)),
    (_M_ANY, 8192, 8192, KernelConfig(128, 256, 128, 2, waves_per_eu=2, xcd_swizzle=1)),
]
# fmt: on


def _heuristic_config_gfx950(m: int, n: int, k: int) -> Optional[KernelConfig]:
    """Tile choice for shapes without a tuned entry, distilled from the sweep
    behind _SHAPE_OVERRIDES_GFX950 (10 N x K pairs, M = 1 .. 8192) and checked
    on 8 held-out N x K pairs: the winners bucket by M much like CK's dispatch,
    tile_k follows K, and async copy wins nearly everywhere. Above M = 512 the
    async winners are 128x128 tiles with the cross-XCD swizzle.
    Returns None when the shape does not divide into the rule's tiles."""

    def tk_for(*prefs: int) -> Optional[int]:
        return next((t for t in prefs if k % t == 0), None)

    wpe, xcd = 0, 1
    if m <= 32:
        tm, tn = 16, 64
        big_or_small_k = k <= 1024 or k >= 7168
        tk = tk_for(512, 256, 128) if big_or_small_k else tk_for(256, 128)
    elif m <= 128:
        tm, tn = 32, 64
        tk = tk_for(128) if k <= 1024 else tk_for(512, 256, 128)
    elif m <= 512:
        tm, tn = 64, (64 if n <= 2048 else 128)
        tk = tk_for(256, 128) if n <= 2048 or k >= 4096 else tk_for(128)
    else:
        tm, tn, wpe, xcd = 128, 128, 2, 4
        tk = tk_for(128)
    while tn > 64 and n % tn:
        tn //= 2
    if tk is None or n % tn:
        return None
    # The A tile is loaded 16 bytes per thread across 256 threads, so
    # tile_m * tile_k must be a multiple of 4096 (16x64x128 is rejected).
    while (tm * tk) % 4096:
        tm *= 2
    return KernelConfig(
        tm, tn, tk, 2, use_async_copy=1, waves_per_eu=wpe, xcd_swizzle=xcd
    )


def select_default_config(m: int, n: int, k: int, batch: int = 1) -> KernelConfig:
    """Select a default FlyDSL tile config based on shape heuristics."""
    for m_upper, n_val, k_val, cfg in _SHAPE_OVERRIDES_GFX950:
        if n == n_val and k == k_val and m <= m_upper:
            return cfg

    # Batched launches multiply the grid, so the per-GEMM M buckets above do
    # not apply; they keep the occupancy-aware pick below.
    if batch == 1:
        cfg = _heuristic_config_gfx950(m, n, k)
        if cfg is not None:
            return cfg

    configs = DEFAULT_CONFIGS_GFX950
    fits = [c for c in configs.values() if n % c.tile_n == 0 and k % c.tile_k == 0]
    if not fits:
        raise RuntimeError(
            f"No FlyDSL preshuffle config fits shape ({m}, {n}, {k}). "
            f"N must be divisible by tile_n and K by tile_k."
        )
    want_tm = min(256, max(16, 1 << (m - 1).bit_length())) if m > 0 else 16

    def _sort_key(c: KernelConfig) -> tuple:
        m_tiles = max(1, -(-m // c.tile_m))
        n_tiles = n // c.tile_n
        low_occupancy = m_tiles * n_tiles * batch < 64
        return (low_occupancy, abs(c.tile_m - want_tm), -c.tile_n, -c.tile_k)

    return min(fits, key=_sort_key)


# ---------------------------------------------------------------------------
# Weight preshuffle
# ---------------------------------------------------------------------------


def flydsl_preshuffle(src: Tensor) -> Tensor:
    """Shuffle FP8 weight tensor into the preshuffle layout this kernel reads.

    For 1-byte types this is the same layout as ``ck_preshuffle(src, 16)`` and
    ``preshuffle_b_mfma(src)``, byte for byte: the 16-row N blocks are laid out
    the same way, and (K//32, 2, ...) with a 16-byte K pack orders K identically
    to CK's (K//64, 4, ...). So weights shuffled for the CK preshuffle ops work
    here unchanged. The difference is only in what each accepts: this needs
    K % 32 == 0 and takes any FP8 dtype, ``ck_preshuffle`` needs K % 64 == 0
    and float8_e4m3fnuz.

    Args:
        src: FP8 weight tensor of shape (N, K).

    Returns:
        Shuffled tensor with same shape and dtype.
    """
    assert src.dim() == 2, f"Expected 2D weight tensor, got {src.dim()}D"
    x_type = src.dtype
    N, K = src.shape
    BN = 16
    BK = 32
    K_pack = 16 // src.element_size()
    assert N % BN == 0, f"N ({N}) must be divisible by {BN}"
    assert K % BK == 0, f"K ({K}) must be divisible by {BK}"
    x_ = src.view(N // BN, BN, K // BK, BK // K_pack, K_pack)
    x_ = x_.permute(0, 2, 3, 1, 4).contiguous()
    x_ = x_.view(N, K).view(x_type)
    return x_


# ---------------------------------------------------------------------------
# Kernel compiler (lazy-loaded)
# ---------------------------------------------------------------------------

_compile_fn = None  # type: ignore[assignment]
_import_done: bool = False


def _get_compile_fn():  # type: ignore[return]
    """Lazy-import the FlyDSL preshuffle kernel compiler."""
    global _compile_fn, _import_done
    if _import_done:
        return _compile_fn
    _import_done = True
    from mslk.flydsl.common import is_flydsl_available

    if not is_flydsl_available():
        return None
    try:
        from mslk.gemm.flydsl._kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

        _compile_fn = compile_preshuffle_gemm_a8
    except Exception:
        pass
    return _compile_fn


def _as_i8(t: Tensor) -> Tensor:
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


# ---------------------------------------------------------------------------
# Single GEMM
# ---------------------------------------------------------------------------


def flydsl_preshuffle_gemm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Optional[Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    tile_k: Optional[int] = None,
    lds_stage: int = 2,
    use_cshuffle_epilog: int = 0,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run FlyDSL preshuffle FP8 rowwise GEMM.

    Args:
        XQ: FP8 activation tensor (..., M, K).
        WQ: FP8 weight tensor (N, K), pre-shuffled via ``flydsl_preshuffle``.
        x_scale: Per-token activation scale (..., M, 1) or (..., M), float32.
        w_scale: Per-channel weight scale (N, 1) or (N,), float32.
        out: Optional pre-allocated output tensor (..., M, N).
        tile_m/tile_n/tile_k: Tile dimensions. If None, auto-selected.
        dtype: Output dtype (bfloat16 or float16).

    Returns:
        Output tensor (..., M, N) in ``dtype``.
    """
    from mslk.flydsl.common import require_flydsl

    require_flydsl()

    m = prod(XQ.shape[:-1])
    k = XQ.shape[-1]
    n = WQ.shape[0]
    output_shape = (*XQ.shape[:-1], n)

    if out is None:
        out = torch.empty(output_shape, dtype=dtype, device=XQ.device)
    elif out.shape != output_shape:
        raise ValueError(f"Expected output shape {output_shape}, got {out.shape}")

    if m == 0 or n == 0:
        return out
    if k == 0:
        return out.zero_()

    compile_fn = _get_compile_fn()
    if compile_fn is None:
        raise RuntimeError("FlyDSL preshuffle kernel compiler not available")

    if tile_m is None or tile_n is None or tile_k is None:
        cfg = select_default_config(m, n, k)
        tile_m = tile_m or cfg.tile_m
        tile_n = tile_n or cfg.tile_n
        tile_k = tile_k or cfg.tile_k
        lds_stage = cfg.lds_stage
        use_cshuffle_epilog = cfg.use_cshuffle_epilog
        use_async_copy = cfg.use_async_copy
        waves_per_eu = cfg.waves_per_eu
        xcd_swizzle = cfg.xcd_swizzle

    if tile_n % 64 != 0:
        # Narrower tiles compile and launch, but leave most of the output
        # unwritten (tile_n=32 was wrong in every run on gfx950, differently
        # each time; 16 gives NaN), so refuse them rather than return garbage.
        raise ValueError(f"tile_n must be a multiple of 64, got {tile_n}")
    if n % tile_n != 0:
        raise RuntimeError(f"N ({n}) not divisible by tile_n ({tile_n})")
    if k % tile_k != 0:
        raise RuntimeError(f"K ({k}) not divisible by tile_k ({tile_k})")

    if "float8" in str(XQ.dtype):
        in_dtype = "fp8"
    elif XQ.dtype == torch.int8:
        in_dtype = "int8"
    else:
        raise ValueError(f"Unsupported input dtype {XQ.dtype}")

    out_dtype = "bf16" if out.dtype == torch.bfloat16 else "fp16"
    wpe = None if waves_per_eu <= 0 else waves_per_eu

    exe = compile_fn(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        lds_stage=lds_stage,
        use_cshuffle_epilog=bool(use_cshuffle_epilog),
        use_async_copy=bool(use_async_copy),
        waves_per_eu=wpe,
        xcd_swizzle=int(xcd_swizzle),
    )

    import flydsl.expr as fx  # pyre-ignore[21]
    from mslk.flydsl.jit import ptr_arg, run_compiled

    def _as_i8(t: Tensor) -> Tensor:
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    out_contig = out.contiguous()
    _dummy_bias = torch.empty(1, dtype=out.dtype, device=out.device)
    run_compiled(
        exe,
        ptr_arg(out_contig.view(-1)),
        ptr_arg(_as_i8(XQ.contiguous()).view(-1)),
        ptr_arg(_as_i8(WQ.contiguous()).view(-1)),
        ptr_arg(x_scale.contiguous().view(-1)),
        ptr_arg(w_scale.contiguous().view(-1)),
        ptr_arg(_dummy_bias),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    if out_contig is not out:
        out.copy_(out_contig)

    return out


# ---------------------------------------------------------------------------
# Batched GEMM
# ---------------------------------------------------------------------------


def flydsl_preshuffle_batched_gemm(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Optional[Tensor] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    tile_k: Optional[int] = None,
    lds_stage: int = 2,
    use_cshuffle_epilog: int = 0,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Batched FP8 preshuffle GEMM via Grid-Z batching.

    Args:
        XQ: FP8 activation tensor (B, M, K).
        WQ: FP8 weight tensor (B, N, K), pre-shuffled via ``flydsl_preshuffle``.
        x_scale: Per-token activation scale (B, M) or (B, M, 1), float32.
        w_scale: Per-channel weight scale (B, N) or (B, 1, N), float32.
        out: Optional pre-allocated output tensor (B, M, N).
        dtype: Output dtype (bfloat16 or float16).
    """
    from mslk.flydsl.common import require_flydsl

    require_flydsl()

    assert XQ.dim() == 3, f"Expected 3D XQ, got {XQ.dim()}D"
    assert WQ.dim() == 3, f"Expected 3D WQ, got {WQ.dim()}D"
    B, M, K = XQ.shape
    N = WQ.shape[1]

    if out is None:
        out = torch.empty(B, M, N, dtype=dtype, device=XQ.device)

    compile_fn = _get_compile_fn()
    if compile_fn is None:
        raise RuntimeError("FlyDSL preshuffle kernel compiler not available")

    if tile_m is None or tile_n is None or tile_k is None:
        cfg = select_default_config(M, N, K, batch=B)
        tile_m = tile_m or cfg.tile_m
        tile_n = tile_n or cfg.tile_n
        tile_k = tile_k or cfg.tile_k
        lds_stage = cfg.lds_stage
        use_cshuffle_epilog = cfg.use_cshuffle_epilog
        use_async_copy = cfg.use_async_copy
        waves_per_eu = cfg.waves_per_eu
        xcd_swizzle = cfg.xcd_swizzle

    if tile_n % 64 != 0:
        # Same limit as flydsl_preshuffle_gemm: narrower tiles leave output unwritten.
        raise ValueError(f"tile_n must be a multiple of 64, got {tile_n}")

    if "float8" in str(XQ.dtype):
        in_dtype = "fp8"
    elif XQ.dtype == torch.int8:
        in_dtype = "int8"
    else:
        raise ValueError(f"Unsupported input dtype {XQ.dtype}")

    out_dtype = "bf16" if dtype == torch.bfloat16 else "fp16"
    wpe = None if waves_per_eu <= 0 else waves_per_eu

    exe = compile_fn(
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        lds_stage=lds_stage,
        use_cshuffle_epilog=bool(use_cshuffle_epilog),
        use_async_copy=bool(use_async_copy),
        waves_per_eu=wpe,
        xcd_swizzle=int(xcd_swizzle),
        batched=True,
    )

    import flydsl.expr as fx  # pyre-ignore[21]
    from mslk.flydsl.jit import ptr_arg, run_compiled

    out_contig = out.contiguous()
    XQ_i8 = _as_i8(XQ.contiguous())
    WQ_i8 = _as_i8(WQ.contiguous())
    xs_contig = x_scale.contiguous()
    ws_contig = w_scale.contiguous()

    dummy_bias = torch.empty(1, dtype=dtype, device=XQ.device)
    stream = fx.Stream(torch.cuda.current_stream())

    run_compiled(
        exe,
        ptr_arg(out_contig.view(-1)),
        ptr_arg(XQ_i8.view(-1)),
        ptr_arg(WQ_i8.view(-1)),
        ptr_arg(xs_contig.view(-1)),
        ptr_arg(ws_contig.view(-1)),
        ptr_arg(dummy_bias),
        M,
        N,
        B,
        stream,
    )
    if out_contig is not out:
        out.copy_(out_contig)

    return out


# ---------------------------------------------------------------------------
# mslk::f8f8{bf16,f16}_rowwise_preshuffle (dispatched from mslk/gemm/__init__.py)
# ---------------------------------------------------------------------------


def is_supported() -> bool:
    """Whether this module can serve the rowwise preshuffle ops on the current
    GPU; ``mslk/gemm/__init__.py`` falls back to CK when it is False."""
    return is_gfx950()


# The kernel needs N % 64 == 0 and K % 128 == 0 on gfx950 (every config's tile_n
# is >= 64; tile_k must be a multiple of 128 for the scaled MFMA), while CK takes
# any shape its layout allows: N % 16 and K % 64. Those shapes are zero-padded
# instead of refused. The layout is (N/16, K/64, 4, 16, 16) blocks, so whole zero
# blocks appended along N and K form a valid layout for the padded shape: the
# padded K adds zeros to every dot product and the padded N columns are sliced
# off.
_PAD_N, _PAD_K = 64, 128
_padded_cache: dict[int, tuple[weakref.ref, int, Tensor]] = {}


def _evict_padded(ref: weakref.ref, key: int) -> None:
    entry = _padded_cache.get(key)
    if entry is not None and entry[0] is ref:
        del _padded_cache[key]


def _pad_preshuffled(WQ: Tensor, n_pad: int, k_pad: int) -> Tensor:
    """WQ zero-padded to (n_pad, k_pad) in the preshuffle layout, cached per
    weight like the rowwise ops' preshuffle (identity + version, not address)."""
    key = id(WQ)
    entry = _padded_cache.get(key)
    if entry is not None:
        ref, version, padded = entry
        if ref() is WQ and version == WQ._version and padded.shape == (n_pad, k_pad):
            return padded
    n, k = WQ.shape
    blocks = WQ.view(torch.int8).view(n // 16, k // 64, 4, 16, 16)
    padded = blocks.new_zeros(n_pad // 16, k_pad // 64, 4, 16, 16)
    padded[: n // 16, : k // 64] = blocks
    padded = padded.view(n_pad, k_pad).view(WQ.dtype)
    ref = weakref.ref(WQ, lambda r, k=key: _evict_padded(r, k))
    _padded_cache[key] = (ref, WQ._version, padded)
    return padded


def _rowwise_preshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor],
    use_fast_accum: bool,
    dtype: torch.dtype,
) -> Tensor:
    # WQ is in the CK preshuffle layout, which for FP8 is the one this kernel
    # reads (see flydsl_preshuffle), so it goes in as given. The checks are the
    # CK op's contract, so callers see the same errors from either backend.
    torch._check(
        XQ.dtype == WQ.dtype,
        lambda: f"XQ and WQ dtypes must match, got {XQ.dtype} vs {WQ.dtype}",
    )
    torch._check(
        x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32,
        lambda: "Scales must be float32.",
    )
    torch._check(
        use_fast_accum, lambda: "AMD does not support disabling use_fast_accum."
    )
    torch._check(bias is None, lambda: "AMD does not support fused bias.")
    torch._check(XQ.is_cuda and XQ.is_contiguous(), lambda: "XQ must be contiguous")
    torch._check(WQ.is_cuda and WQ.is_contiguous(), lambda: "WQ must be contiguous")
    torch._check(WQ.dim() == 2, lambda: f"WQ must be [N, K], got {WQ.shape}")
    torch._check(
        XQ.shape[-1] == WQ.shape[1],
        lambda: f"K mismatch: XQ {XQ.shape} vs WQ {WQ.shape}",
    )
    n, k = WQ.shape
    if prod(XQ.shape[:-1]) == 0 or n == 0 or k == 0:
        # As CK: zeros of the output shape (k == 0 sums nothing).
        return XQ.new_zeros((*XQ.shape[:-1], n), dtype=dtype)
    # The preshuffle layout's own granularity, which ck_preshuffle and
    # preshuffle_b_mfma enforce too.
    torch._check(
        n % 16 == 0 and k % 64 == 0,
        lambda: f"preshuffled WQ needs N % 16 == 0 and K % 64 == 0, got N={n} K={k}",
    )
    n_pad = -(-n // _PAD_N) * _PAD_N
    k_pad = -(-k // _PAD_K) * _PAD_K
    if (n_pad, k_pad) == (n, k):
        return flydsl_preshuffle_gemm(XQ, WQ, x_scale, w_scale, dtype=dtype)
    xq = XQ
    if k_pad != k:
        # Zero bytes are +0.0 in every FP8 format, so pad the raw bytes.
        xq = torch.nn.functional.pad(XQ.view(torch.int8), (0, k_pad - k))
        xq = xq.view(XQ.dtype)
    ws = w_scale.reshape(-1)
    if n_pad != n:
        ws = torch.nn.functional.pad(ws, (0, n_pad - n))
    out = flydsl_preshuffle_gemm(
        xq, _pad_preshuffled(WQ, n_pad, k_pad), x_scale, ws, dtype=dtype
    )
    return out[..., :n].contiguous() if n_pad != n else out


def matmul_f8f8bf16_rowwise_preshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    use_fast_accum: bool = True,
) -> Tensor:
    """FlyDSL impl of ``mslk::f8f8bf16_rowwise_preshuffle`` on gfx950."""
    return _rowwise_preshuffle(
        XQ, WQ, x_scale, w_scale, bias, use_fast_accum, torch.bfloat16
    )


def matmul_f8f8f16_rowwise_preshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    use_fast_accum: bool = True,
) -> Tensor:
    """FlyDSL impl of ``mslk::f8f8f16_rowwise_preshuffle`` on gfx950."""
    return _rowwise_preshuffle(
        XQ, WQ, x_scale, w_scale, bias, use_fast_accum, torch.float16
    )


# ---------------------------------------------------------------------------
# Register FlyDSL as the ROCm implementation of mslk rowwise FP8 ops on gfx950
# ---------------------------------------------------------------------------
if torch.version.hip is not None and is_gfx950() and hasattr(torch.ops, "mslk"):
    from mslk.flydsl.common import is_flydsl_available

    if is_flydsl_available():
        # Keyed by the weight tensor itself, not its address: a freed weight's
        # address is handed to the next allocation of the same size, which
        # would otherwise be served the old weight's preshuffle. The version
        # counter catches in-place updates, and an entry is dropped once its
        # weight is collected so the cache does not pin the copies.
        _preshuffle_cache: dict[int, tuple[weakref.ref, int, Tensor]] = {}

        def _evict(ref: weakref.ref, key: int) -> None:
            entry = _preshuffle_cache.get(key)
            if entry is not None and entry[0] is ref:
                del _preshuffle_cache[key]

        def _get_preshuffled(WQ: Tensor) -> Tensor:
            key = id(WQ)
            entry = _preshuffle_cache.get(key)
            if entry is not None:
                ref, version, shuf = entry
                if ref() is WQ and version == WQ._version and shuf.shape == WQ.shape:
                    return shuf
            shuf = flydsl_preshuffle(WQ)
            ref = weakref.ref(WQ, lambda r, k=key: _evict(r, k))
            _preshuffle_cache[key] = (ref, WQ._version, shuf)
            return shuf

        def _flydsl_rowwise_impl(
            XQ: Tensor,
            WQ: Tensor,
            x_scale: Tensor,
            w_scale: Tensor,
            bias: Optional[Tensor] = None,
            use_fast_accum: bool = True,
            dtype: torch.dtype = torch.bfloat16,
            output: Optional[Tensor] = None,
        ) -> Tensor:
            WQ_shuf = _get_preshuffled(WQ)
            return flydsl_preshuffle_gemm(
                XQ,
                WQ_shuf,
                x_scale,
                w_scale,
                out=output,
                dtype=dtype,
            )

        if hasattr(torch.ops.mslk, "f8f8bf16_rowwise"):

            @torch.library.impl("mslk::f8f8bf16_rowwise", "CUDA")
            def _f8f8bf16_rowwise_flydsl(
                XQ: Tensor,
                WQ: Tensor,
                x_scale: Tensor,
                w_scale: Tensor,
                bias: Optional[Tensor] = None,
                use_fast_accum: bool = True,
            ) -> Tensor:
                return _flydsl_rowwise_impl(
                    XQ,
                    WQ,
                    x_scale,
                    w_scale,
                    bias,
                    use_fast_accum,
                    dtype=torch.bfloat16,
                )

        if hasattr(torch.ops.mslk, "f8f8bf16_rowwise_out"):

            @torch.library.impl("mslk::f8f8bf16_rowwise_out", "CUDA")
            def _f8f8bf16_rowwise_out_flydsl(
                XQ: Tensor,
                WQ: Tensor,
                x_scale: Tensor,
                w_scale: Tensor,
                output: Tensor,
                bias: Optional[Tensor] = None,
                use_fast_accum: bool = True,
            ) -> None:
                _flydsl_rowwise_impl(
                    XQ,
                    WQ,
                    x_scale,
                    w_scale,
                    bias,
                    use_fast_accum,
                    dtype=output.dtype,
                    output=output,
                )

        if hasattr(torch.ops.mslk, "f8f8f16_rowwise"):

            @torch.library.impl("mslk::f8f8f16_rowwise", "CUDA")
            def _f8f8f16_rowwise_flydsl(
                XQ: Tensor,
                WQ: Tensor,
                x_scale: Tensor,
                w_scale: Tensor,
                bias: Optional[Tensor] = None,
                use_fast_accum: bool = True,
            ) -> Tensor:
                return _flydsl_rowwise_impl(
                    XQ,
                    WQ,
                    x_scale,
                    w_scale,
                    bias,
                    use_fast_accum,
                    dtype=torch.float16,
                )

    # --- batched op ---
    if hasattr(torch.ops.mslk, "f8f8bf16_rowwise_batched"):
        from mslk.flydsl.common import is_flydsl_available as _is_flydsl_batched

        if _is_flydsl_batched():

            @torch.library.impl("mslk::f8f8bf16_rowwise_batched", "CUDA")
            def _f8f8bf16_rowwise_batched_flydsl(
                XQ: Tensor,
                WQ: Tensor,
                x_scale: Tensor,
                w_scale: Tensor,
                bias: Optional[Tensor] = None,
                use_fast_accum: bool = True,
                output: Optional[Tensor] = None,
            ) -> Tensor:
                if bias is not None:
                    raise NotImplementedError(
                        "FlyDSL batched preshuffle GEMM does not support bias"
                    )
                return flydsl_preshuffle_batched_gemm(
                    XQ,
                    WQ,
                    x_scale,
                    w_scale,
                    out=output,
                )
