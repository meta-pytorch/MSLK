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
ops instead (``preshuffle_gemm.py``), which share this module's operand
checks.

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
from mslk.utils.device import is_gfx942, is_gfx950

# Config used when autotuning is disabled. Picked for the smallest worst case
# against the per-shape best, not the most wins, so an untuned caller has a
# bounded loss. Legal for every supported shape on both architectures: two
# stages is all gfx942, and K not a multiple of 16, can take, and it fits
# gfx942's 64 KiB of LDS.
DEFAULT_TILE = {
    "tile_m": 64,
    "tile_n": 64,
    "tile_k": 256,
    "waves_m": 2,
    "waves_n": 2,
    "stages": 2,
    "waves_per_eu": 0,
}

# Candidates swept by autotune. With few rows the time is the latency of the
# weight loads, which small tiles in one-wave blocks with a deep pipeline hide
# best; with many rows big tiles and eight waves win, so the space spans both.
_TILE_M = (32, 64, 128, 256)
_TILE_N = (32, 64, 128, 256)
_TILE_K = (128, 256)
# The wave grid is explicit rather than a wave count, because its shape is what
# drives LDS read traffic: reads per unit work are waves_m / tile_m + waves_n /
# tile_n, so the grid wants to be proportioned like the tile.
_WAVE_GRIDS = ((1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (2, 4), (4, 2), (8, 1), (1, 8))
# LDS buffers per operand; more than two needs gfx950 and K a multiple of 16.
_STAGES = (2, 3, 4)
# Occupancy target, as a minimum waves-per-EU hint to the register allocator; 0
# leaves the choice to the compiler.
_WAVES_PER_EU = (0, 2)

# Many of these do not divide into whole MFMA tiles per wave and whole 16-byte
# loads per thread, or do not fit LDS or the pipeline the architecture and K
# allow; those raise, which the autotuner reports and skips. Enumerating the
# legal subset here instead would duplicate the kernel's guards and drift from
# them.
_TILES = tuple(
    {
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": tk,
        "waves_m": wm,
        "waves_n": wn,
        "stages": st,
        "waves_per_eu": wpe,
    }
    for tm in _TILE_M
    for tn in _TILE_N
    for tk in _TILE_K
    for (wm, wn) in _WAVE_GRIDS
    for st in _STAGES
    for wpe in _WAVES_PER_EU
)

# A tile that overruns N or K still compiles, masking its tail, but wastes part
# of its work, so prune on both axes; where nothing divides, the full list is
# kept.
_PRUNE = prune_by_divisibility({"tile_n": "n", "tile_k": "k"})
# The output dtype is part of the kernel, so the f16 and bf16 ops tune apart.
_KEY = ["m_bucket", "n", "k", "batch", "out_dtype"]


def _launch_fp8_rowwise(
    XQ,
    WQ,
    x_scale,
    w_scale,
    out,
    m_bucket,
    n,
    k,
    batch,
    out_dtype,
    *,
    tile_m,
    tile_n,
    tile_k,
    waves_m,
    waves_n,
    stages,
    waves_per_eu=0,
):
    """Compile (cached) and launch the kernel for one config.

    ``XQ``/``out`` are the batch's rows stacked, ``[batch * M, K]`` and
    ``[batch * M, N]``. ``m_bucket`` only feeds the autotune key: bucketing M
    keeps nearby token counts on one tuned config. ``n``/``k`` are passed for
    the key and for tile pruning. FlyDSL names the tuning cache file after this
    function.
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
    # The kernel addresses the operands as flat byte buffers; FP8 is viewed as
    # int8 for the handoff, as the grouped path does.
    run_compiled(
        launcher,
        out,
        XQ.view(torch.int8),
        WQ.view(torch.int8),
        x_scale,
        w_scale,
        XQ.shape[0] // batch,
        batch,
        torch.cuda.current_stream(XQ.device),
    )
    return out


_tuned_fp8_rowwise = tunable(
    configs=_TILES, default=DEFAULT_TILE, key=_KEY, prune=_PRUNE
)(_launch_fp8_rowwise)


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
    # argument's rank, and is shared by every caller with the same N and K.
    _tuned_fp8_rowwise(
        XQ.view(batch * m, k),
        WQ.view(batch * n, k),
        x_scale.contiguous().view(batch * m),
        w_scale.contiguous().view(batch * n),
        out.view(batch * m, n),
        next_pow2(m),
        n,
        k,
        batch,
        "f16" if out.dtype == torch.float16 else "bf16",
    )


def check_rowwise_operands(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    use_fast_accum: bool,
) -> None:
    """The CK rowwise ops' operand contract, shared by the plain and preshuffle
    ops.

    The FP8 flavour follows the MFMA the kernels issue, OCP E4M3 on gfx950 and
    the fnuz encoding on gfx942, whatever torch's FP8 default is set to.
    """
    torch._check(
        XQ.dtype == WQ.dtype,
        lambda: f"XQ and WQ dtypes must match, got {XQ.dtype} vs {WQ.dtype}",
    )
    gfx950 = is_gfx950()
    fp8_dtype = torch.float8_e4m3fn if gfx950 else torch.float8_e4m3fnuz

    def fp8_message() -> str:
        arch = "gfx950" if gfx950 else "gfx942"
        msg = f"XQ and WQ must be {fp8_dtype} on {arch}, got {XQ.dtype}"
        if gfx950 and XQ.dtype == torch.float8_e4m3fnuz:
            msg += (
                "; MSLK_ROCM_FORCE_FP8FNUZ_TYPE makes MSLK quantize to fnuz, "
                "which these kernels do not read on gfx950"
            )
        return msg

    torch._check(XQ.dtype == fp8_dtype, fp8_message)
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
    check_rowwise_operands(XQ, WQ, x_scale, w_scale, bias, use_fast_accum)
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
    """Whether this module can serve the plain rowwise ops on the current GPU.

    The kernel is built on MFMA, which the RDNA parts do not have -- and FlyDSL
    reports itself available on those, so having the backend says nothing about
    whether these ops can run. ``mslk/gemm/__init__.py`` falls back to CK when
    it is False.
    """
    return is_gfx950() or is_gfx942()


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
    check_rowwise_operands(XQ, WQ, x_scale, w_scale, bias, use_fast_accum)
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
