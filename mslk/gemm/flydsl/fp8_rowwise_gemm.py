# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 rowwise-scaled GEMM on plain weights via FlyDSL (ROCm implementation of
``mslk::f8f8bf16_rowwise``, ``f8f8bf16_rowwise_out``, ``f8f8f16_rowwise`` and
``f8f8bf16_rowwise_batched``).

The kernel, ``mslk.flydsl.kernels.gemm.fp8_rowwise_gemm``, reads the caller's
row-major weights and takes every shape the CK ops do. The batched op runs one
batch per slice of the grid's z axis; the others are a batch of one. Callers
that keep their weights in the MFMA preshuffle layout use the ``_preshuffle``
ops instead (``preshuffle_gemm.py``).

Tensor contract (matches the CK ops):
  XQ      : [..., M, K]  FP8    ([B, M, K] for the batched op)
  WQ      : [N, K]       FP8    ([B, N, K] for the batched op)
  x_scale : [..., M]     FP32   one scale per row of XQ
  w_scale : [N]          FP32   one scale per row of WQ ([B, N] batched)
  Output  : [..., M, N]  BF16, or FP16 for ``f8f8f16_rowwise``

  out[m, n] = (sum_k XQ[m, k] * WQ[n, k]) * x_scale[m] * w_scale[n]
"""

import functools
import itertools
from math import prod
from typing import Optional

import torch
from mslk.flydsl.autotune import next_pow2, tunable
from mslk.flydsl.common import lds_capacity_bytes
from mslk.utils.device import get_gfx_arch_name, gfx_arch_in, supports_float8_fnuz

# Candidate configs for autotune. Small M is bound by how many weight loads
# are in flight, so the space runs down to one-wave blocks of 32x32 with a
# deep pipeline; large M wants big tiles and eight waves. Only configs the
# kernel accepts on this GPU are listed: whole 32x32 MFMA tiles per wave and at
# most eight of them, whole
# 16-byte loads per thread, the LDS budget, and an in-flight count s_waitcnt
# can express. gfx942 stages through registers, which pipelines two deep.
# Configs that need K to be a multiple of 16 (more than two stages) fail to
# compile otherwise, which the autotuner skips.
# Without the target features (e.g. "gfx950:sramecc+:xnack-"), as the LDS
# table is keyed.
_ARCH = get_gfx_arch_name().split(":")[0]
_CDNA4 = _ARCH.startswith("gfx95")
_LDS_BYTES = lds_capacity_bytes(_ARCH)
_STAGES = (2, 3, 4) if _CDNA4 else (2,)
_WAVE_GRIDS = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (2, 4),
    (4, 2),
    (8, 1),
    (1, 8),
)


def _legal(tm: int, tn: int, tk: int, wm: int, wn: int, stages: int) -> bool:
    threads = wm * wn * 64
    loads = (tm + tn) * tk // (threads * 16)
    return (
        tm % (wm * 32) == 0
        and tn % (wn * 32) == 0
        and (tm // wm // 32) * (tn // wn // 32) <= 8
        and (tm * tk) % (threads * 16) == 0
        and (tn * tk) % (threads * 16) == 0
        and stages * (tm + tn) * tk <= _LDS_BYTES
        and (stages - 2) * loads <= 63
    )


_TILES = tuple(
    dict(tile_m=tm, tile_n=tn, tile_k=tk, waves_m=wm, waves_n=wn, stages=st)
    for tm, tn, tk, (wm, wn), st in itertools.product(
        (32, 64, 128, 256), (32, 64, 128, 256), (64, 128, 256), _WAVE_GRIDS, _STAGES
    )
    if _legal(tm, tn, tk, wm, wn, st)
)

# Untuned config per M bucket, (m_upper, tile_m, tile_n, tile_k, waves_m,
# waves_n, stages). On gfx950 each is the config that loses least, in geometric
# mean, to the per-shape best over 10 (N, K) pairs; the worst loss is 1.26x up
# to M = 128 and 1.56x above. gfx942 is not tuned: its buckets are the gfx950
# ones cut to its 64 KiB of LDS and two stages.
_M_BUCKETS_GFX950 = (
    (64, 32, 32, 256, 1, 1, 4),
    (128, 32, 64, 256, 1, 2, 3),
    (512, 64, 64, 128, 2, 1, 4),
    (1024, 128, 64, 128, 4, 1, 3),
    (2048, 128, 128, 128, 4, 2, 4),
    (4096, 128, 128, 128, 4, 1, 2),
    (1 << 62, 256, 256, 128, 8, 1, 2),
)
_M_BUCKETS_GFX942 = (
    (64, 32, 32, 256, 1, 1, 2),
    (128, 32, 64, 256, 1, 2, 2),
    (512, 64, 64, 128, 2, 1, 2),
    (1024, 128, 64, 128, 4, 1, 2),
    (1 << 62, 128, 128, 128, 4, 2, 2),
)


def _default_tile(*_: object, m_bucket: int, k: int, **__: object) -> dict:
    for m_upper, tm, tn, tk, wm, wn, stages in (
        _M_BUCKETS_GFX950 if _CDNA4 else _M_BUCKETS_GFX942
    ):
        if m_bucket <= m_upper:
            break
    # A deeper pipeline moves whole 16-byte chunks straight into LDS, which
    # needs every row to start on one.
    if k % 16:
        stages = 2
    return dict(tile_m=tm, tile_n=tn, tile_k=tk, waves_m=wm, waves_n=wn, stages=stages)


@tunable(
    configs=_TILES,
    default=_default_tile,
    key=["m_bucket", "n", "k", "batch", "out_dtype"],
)
def _launch_fp8_rowwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    m: int,
    m_bucket: int,
    n: int,
    k: int,
    batch: int,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_m: int,
    waves_n: int,
    stages: int,
    waves_per_eu: int = 0,
) -> torch.Tensor:
    """Compile (cached) and launch the kernel for one config.

    ``m_bucket`` only feeds the tuning key, so nearby row counts share a tuned
    config; ``m`` is the actual count. FlyDSL names the tuning cache file after
    this function.
    """
    from mslk.flydsl.jit import run_compiled
    from mslk.flydsl.kernels.gemm.fp8_rowwise_gemm import compile_fp8_rowwise_gemm

    launcher = compile_fp8_rowwise_gemm(
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_m=waves_m,
        waves_n=waves_n,
        stages=stages,
        # 0 means the compiler picks.
        waves_per_eu=None if waves_per_eu <= 0 else waves_per_eu,
        out_dtype=out_dtype,
    )
    # The kernel addresses the operands as flat byte buffers, so FP8 is viewed
    # as int8 for the handoff.
    run_compiled(
        launcher,
        out,
        XQ.view(torch.int8),
        WQ.view(torch.int8),
        x_scale,
        w_scale,
        m,
        batch,
        torch.cuda.current_stream(XQ.device),
    )
    return out


def _run(
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
    # One canonical shape per operand: the compiled launcher fixes each
    # argument's rank, and is shared by every caller with the same N and K. The
    # tuning key and the default are read from keyword arguments, so they are
    # passed by name.
    _launch_fp8_rowwise(
        XQ.view(batch * m, k),
        WQ.view(batch * n, k),
        x_scale.contiguous().view(batch * m),
        w_scale.contiguous().view(batch * n),
        out.view(batch * m, n),
        m=m,
        m_bucket=next_pow2(m),
        n=n,
        k=k,
        batch=batch,
        out_dtype="f16" if out.dtype == torch.float16 else "bf16",
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
    output: torch.Tensor,
    rows: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    # As CK: the leading dims only have to flatten to the rows of XQ.
    torch._check(
        output.dim() >= 1 and output.shape[-1] == n and prod(output.shape[:-1]) == rows,
        lambda: f"output must hold {rows} rows of {n}, got {tuple(output.shape)}",
    )
    torch._check(output.dtype == dtype, lambda: f"output must be {dtype}")
    torch._check(output.device == device, lambda: "output must be on XQ's device")
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
    if output is None:
        output = XQ.new_empty((*XQ.shape[:-1], n), dtype=dtype)
    else:
        _check_output(output, m, n, dtype, XQ.device)
    if m == 0 or n == 0 or k == 0:
        # A contraction over nothing sums to zero.
        return output.zero_()
    _run(XQ, WQ, x_scale, w_scale, output, 1, m, n, k)
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
    if output is None:
        output = XQ.new_empty((batch, m, n), dtype=torch.bfloat16)
    else:
        torch._check(
            tuple(output.shape) == (batch, m, n),
            lambda: f"output must be {(batch, m, n)}, got {tuple(output.shape)}",
        )
        _check_output(output, batch * m, n, torch.bfloat16, XQ.device)
    if batch == 0 or m == 0 or n == 0 or k == 0:
        return output.zero_()
    _run(XQ, WQ, x_scale, w_scale, output, batch, m, n, k)
    return output


# This module registers nothing: the ops are registered (lazily) in
# mslk/gemm/__init__.py, keeping FlyDSL an opt-in dependency.
