# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 rowwise-scaled GEMM on plain weights via FlyDSL (ROCm implementation of
``mslk::f8f8bf16_rowwise``, ``f8f8bf16_rowwise_out``, ``f8f8f16_rowwise`` and
``f8f8bf16_rowwise_batched``).

These run the shared grouped GEMM kernel in its ``batched`` layout, in which
each group owns a fixed slab of rows and a slice of the grid's z axis, so
nothing is resolved per group. The batched op is one group per batch; the
others are a batch of one. B is the caller's row-major ``[N, K]`` weight,
staged HBM->LDS->registers. Callers that keep their weights in the MFMA
preshuffle layout use the ``_preshuffle`` ops instead (``preshuffle_gemm.py``).

Tensor contract (matches the CK ops):
  XQ      : [..., M, K]  FP8    ([B, M, K] for the batched op)
  WQ      : [N, K]       FP8    ([B, N, K] for the batched op)
  x_scale : [..., M]     FP32   one scale per row of XQ
  w_scale : [N]          FP32   one scale per row of WQ ([B, N] batched)
  Output  : [..., M, N]  BF16, or FP16 for ``f8f8f16_rowwise``

  out[m, n] = (sum_k XQ[m, k] * WQ[n, k]) * x_scale[m] * w_scale[n]
"""

import functools
from math import prod
from typing import Optional

import torch
from mslk.flydsl.autotune import next_pow2, prune_by_divisibility, tunable
from mslk.gemm.flydsl import grouped_dispatch
from mslk.utils.device import gfx_arch_in, supports_float8_fnuz

# The kernel masks a K tail one 16-byte load at a time and an N tail one
# 8-column store at a time, so shapes only have to land on those boundaries.
_K_ALIGN, _N_ALIGN = 16, 8

# Candidate tiles for autotune. Unlike the grouped ops' rowwise space, tile_m
# goes down to 16: with a few rows the work is reading the weight, and a taller
# tile only multiplies padding. A tile has to give each of the block's 256
# threads whole 16-byte loads of A, which tile_m=16 does only at tile_k=256.
_TILES = tuple(
    dict(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        waves_m=wm,
        waves_n=wn,
        waves_per_eu=wpe,
    )
    for tm in (16, 32, 64, 128, 256)
    for tn in (64, 128, 256)
    for tk in (128, 256)
    for wm, wn in ((1, 4), (2, 2), (4, 1))
    for wpe in (0, 2)
    if tm % (wm * 16) == 0 and tn % (wn * 16) == 0 and (tm * tk) % (256 * 16) == 0
)

# Untuned tile per M bucket: the one closest to the best over 10 (N, K) pairs
# with M = 1 .. 8192 on gfx950, (m_upper, tile_m, tile_n, tile_k, waves_m,
# waves_n, waves_per_eu). All of them fit gfx942's 64 KiB of LDS. No fixed tile
# serves both ends: 128x128x128 takes about 3.7x as long as the small tile at
# M <= 32, where the few rows leave most of a tall tile idle.
_M_BUCKETS = (
    (64, 16, 64, 256, 1, 4, 2),
    (128, 32, 64, 256, 2, 2, 2),
    (512, 64, 64, 256, 2, 2, 0),
    (1024, 128, 128, 128, 4, 1, 0),
    (1 << 62, 128, 128, 128, 2, 2, 2),
)


def _default_tile(*, m_bucket: int, **_: object) -> dict:
    for m_upper, tm, tn, tk, wm, wn, wpe in _M_BUCKETS:
        if m_bucket <= m_upper:
            break
    return dict(
        tile_m=tm, tile_n=tn, tile_k=tk, waves_m=wm, waves_n=wn, waves_per_eu=wpe
    )


@tunable(
    configs=_TILES,
    default=_default_tile,
    key=["m_bucket", "n", "k", "g"],
    prune=prune_by_divisibility({"tile_n": "n", "tile_k": "k"}),
)
def _launch_fp8_rowwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    m_bucket: int,
    n: int,
    k: int,
    g: int,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_m: int,
    waves_n: int,
    waves_per_eu: int = 0,
) -> torch.Tensor:
    """The grouped kernel in its batched layout, one slab per batch.

    A function of its own rather than another tunable over
    ``grouped_dispatch.launch``: FlyDSL names the tuning cache file after the
    wrapped function, and this keeps these entries out of the grouped ops' file.
    """
    return grouped_dispatch.launch(
        XQ,
        WQ,
        x_scale,
        w_scale,
        grouped_dispatch.unused_group_meta(XQ.device),
        out,
        m_bucket,
        n,
        k,
        g,
        False,
        "row",
        "batched",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_m=waves_m,
        waves_n=waves_n,
        waves_per_eu=waves_per_eu,
    )


def _launch(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    batch: int,
    m: int,
    n: int,
    k: int,
) -> None:
    torch._check(
        n % _N_ALIGN == 0 and k % _K_ALIGN == 0,
        lambda: f"N must be a multiple of {_N_ALIGN} and K of {_K_ALIGN}, "
        f"got N={n} K={k}",
    )
    # Bucketing M keeps nearby row counts on one tuned entry.
    _launch_fp8_rowwise(
        XQ.view(batch * m, k),
        WQ.view(batch, n, k),
        x_scale.reshape(batch * m),
        w_scale.reshape(batch, n),
        out.view(batch * m, n),
        next_pow2(m),
        n,
        k,
        batch,
    )


def _check_operands(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    use_fast_accum: bool,
) -> None:
    torch._check(
        XQ.dtype == WQ.dtype,
        lambda: f"XQ and WQ dtypes must match, got {XQ.dtype} vs {WQ.dtype}",
    )
    fp8 = torch.float8_e4m3fnuz if supports_float8_fnuz() else torch.float8_e4m3fn
    torch._check(XQ.dtype == fp8, lambda: f"XQ and WQ must be {fp8}")
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
    torch._check(
        WQ.device == XQ.device
        and x_scale.device == XQ.device
        and w_scale.device == XQ.device,
        lambda: "XQ, WQ and scales must be on the same device",
    )


def _check_output(
    output: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    torch._check(
        tuple(output.shape) == shape,
        lambda: f"output must be {shape}, got {tuple(output.shape)}",
    )
    torch._check(output.dtype == dtype, lambda: f"output must be {dtype}")
    # The kernel writes it in place, so a contiguous copy would be written instead.
    torch._check(output.is_contiguous(), lambda: "output must be contiguous")


def _rowwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    use_fast_accum: bool,
    dtype: torch.dtype,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _check_operands(XQ, WQ, x_scale, w_scale, bias, use_fast_accum)
    torch._check(WQ.dim() == 2, lambda: f"WQ must be [N, K], got {tuple(WQ.shape)}")
    torch._check(
        XQ.dim() >= 1 and XQ.shape[-1] == WQ.shape[1],
        lambda: f"K mismatch: XQ {tuple(XQ.shape)} vs WQ {tuple(WQ.shape)}",
    )
    n, k = WQ.shape
    m = prod(XQ.shape[:-1])
    torch._check(x_scale.numel() == m, lambda: "x_scale must have one value per row")
    torch._check(w_scale.numel() == n, lambda: "w_scale must have one value per column")
    shape = (*XQ.shape[:-1], n)
    if output is None:
        output = XQ.new_empty(shape, dtype=dtype)
    else:
        _check_output(output, shape, dtype)
    if m == 0 or n == 0 or k == 0:
        # A contraction over nothing sums to zero.
        return output.zero_()
    _launch(XQ, WQ, x_scale, w_scale, output, 1, m, n, k)
    return output


@functools.lru_cache(maxsize=1)
def is_supported() -> bool:
    """Whether this module can serve the plain rowwise ops on the current GPU;
    ``mslk/gemm/__init__.py`` falls back to CK when it is False."""
    return gfx_arch_in(("gfx942", "gfx950"))


def matmul_f8f8bf16_rowwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """FlyDSL impl of ``mslk::f8f8bf16_rowwise``."""
    return _rowwise(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, torch.bfloat16)


def matmul_f8f8bf16_rowwise_out(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> None:
    """FlyDSL impl of ``mslk::f8f8bf16_rowwise_out``."""
    _rowwise(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, torch.bfloat16, output)


def matmul_f8f8f16_rowwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """FlyDSL impl of ``mslk::f8f8f16_rowwise``."""
    return _rowwise(XQ, WQ, x_scale, w_scale, bias, use_fast_accum, torch.float16)


def matmul_f8f8bf16_rowwise_batched(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FlyDSL impl of ``mslk::f8f8bf16_rowwise_batched``."""
    _check_operands(XQ, WQ, x_scale, w_scale, bias, use_fast_accum)
    torch._check(
        XQ.dim() == 3 and WQ.dim() == 3,
        lambda: "Inputs must have 3 dimensions, with the first being batch.",
    )
    batch, m, k = XQ.shape
    torch._check(
        WQ.shape[0] == batch and WQ.shape[2] == k,
        lambda: f"XQ {tuple(XQ.shape)} and WQ {tuple(WQ.shape)} do not match",
    )
    n = WQ.shape[1]
    torch._check(
        x_scale.numel() == batch * m, lambda: "x_scale must have one value per row"
    )
    torch._check(
        w_scale.numel() == batch * n, lambda: "w_scale must have one value per column"
    )
    shape = (batch, m, n)
    if output is None:
        output = XQ.new_empty(shape, dtype=torch.bfloat16)
    else:
        _check_output(output, shape, torch.bfloat16)
    if batch == 0 or m == 0 or n == 0 or k == 0:
        return output.zero_()
    _launch(XQ, WQ, x_scale, w_scale, output, batch, m, n, k)
    return output


# This module registers nothing: the ops are registered (lazily) in
# mslk/gemm/__init__.py, keeping FlyDSL an opt-in dependency.
