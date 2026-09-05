# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 groupwise-scaled GEMM via FlyDSL.

Implements ``mslk::f8f8bf16_groupwise``, the plain (non-grouped) GEMM whose
weights are scaled per 128x128 block and whose activations are scaled per token
per 128 of K. CUDA implements it in CUTLASS. The op is registered in
``mslk/gemm/__init__.py``, which picks this implementation wherever the FlyDSL
backend is opted into and Triton's otherwise.

Two kernels serve it, chosen by architecture.

On gfx950 it is a kernel written for this op alone, built around instructions
that architecture introduced -- the wide ``f8f6f4`` MFMA and ``permlane32_swap``
-- and around the fact that a plain GEMM needs no group resolution at all. See
``mslk.flydsl.kernels.gemm.fp8_groupwise_wide_gemm``.

On gfx942 it is the same kernel as the grouped ops, compiled for a single group
under the ``batched`` layout: that layout gives each group a fixed slab of rows,
so one group is one slab spanning all of M, which is a plain GEMM. The scale
layouts coincide too. Block scaling addresses scale_a as per-group blocks --
group g's block starts at ``m_start * scale_k`` and holds element
``(local_m, k_block)`` at ``local_m + k_block * M_g`` -- and at one group that is
``local_m + k_block * M``, which is exactly the ``[K//128, M]`` this op is
handed. Likewise ``[G, K//128, N//128]`` is ``[K//128, N//128]`` at G = 1.

Tensor contract:
  XQ      : [M, K]               FP8  -- activations
  WQ      : [N, K]               FP8  -- weights, already transposed
  x_scale : [K//128, M]          FP32 -- per (K-group-of-128, row)
  w_scale : [K//128, N//128]     FP32 -- per (K-group-of-128, N-group-of-128)
  Output  : [M, N]               BF16

  out[m, n] = sum_k XQ[m, k] * x_scale[k//128, m]
                  * WQ[n, k] * w_scale[k//128, n//128]
"""

import functools
from collections.abc import Callable

import torch
from mslk.flydsl.autotune import next_pow2, prune_by_divisibility, tunable
from mslk.gemm.flydsl import grouped_dispatch
from mslk.utils.device import is_gfx942, is_gfx950

_SCALE_BLOCK = grouped_dispatch.SCALE_BLOCK

# Config used when autotuning is disabled. Picked for the smallest worst case
# against the per-shape best, not the most wins, so an untuned caller has a
# bounded loss. Legal for every supported shape.
DEFAULT_TILE = {
    "tile_m": 128,
    "tile_n": 64,
    "waves_m": 4,
    "waves_n": 1,
    "waves_per_eu": 0,
}

# Candidates swept by autotune. tile_k is absent: the scales change every 128
# elements of K and a tile spanning more than one scale block would need a fold
# per sub-block, so the kernel fixes it at the block size.
_TILE_M = (32, 64, 128, 256)
# Capped at the scale block, above which a tile would span several B scales.
_TILE_N = (64, 128)
# The wave grid is explicit rather than a wave count, because its shape is what
# drives LDS read traffic: reads per unit work are waves_m / tile_m + waves_n /
# tile_n, so the grid wants to be proportioned like the tile. Both four-wave and
# eight-wave blocks are offered, in each proportion the tiles above can divide.
_WAVE_GRIDS = ((1, 4), (2, 2), (4, 1), (2, 4), (4, 2), (8, 1))
# Occupancy target, as a minimum waves-per-EU hint to the register allocator; 0
# leaves the choice to the compiler.
_WAVES_PER_EU = (0, 2)

# Roughly half of these divide into whole MFMA tiles per wave and whole 16-byte
# loads per lane, and fit LDS; the rest raise, which the autotuner reports and
# skips. Enumerating the legal subset here instead would duplicate the kernel's
# guards and drift from them.
_TILES = tuple(
    {
        "tile_m": tm,
        "tile_n": tn,
        "waves_m": wm,
        "waves_n": wn,
        "waves_per_eu": wpe,
    }
    for tm in _TILE_M
    for tn in _TILE_N
    for (wm, wn) in _WAVE_GRIDS
    for wpe in _WAVES_PER_EU
)

_PRUNE = prune_by_divisibility({"tile_n": "n"})
_KEY = ["m_bucket", "n", "k"]


def _matmul_gfx942(XQ, WQ, x_scale, w_scale, M, N, K):
    """Serve the op from the grouped GEMM kernel, as one batched group."""
    # One group owning every row, so the weights become the single-entry stack
    # the kernel indexes by group. The view is free on a contiguous tensor.
    return grouped_dispatch.dispatch(
        XQ,
        WQ.unsqueeze(0),
        x_scale,
        w_scale,
        grouped_dispatch.unused_group_meta(XQ.device),
        b_preshuffled=False,
        blockscale=True,
        layout="batched",
    )


def _launch_gfx950(
    XQ,
    WQ,
    x_scale,
    w_scale,
    out,
    m_bucket,
    n,
    k,
    *,
    tile_m,
    tile_n,
    waves_m,
    waves_n,
    waves_per_eu=0,
):
    """Compile (cached) and launch the dedicated kernel for one config.

    ``m_bucket`` only feeds the autotune key: bucketing M keeps nearby token
    counts on one tuned config. ``n``/``k`` are passed for the key and for tile
    pruning, and are read back off the operands here.
    """
    from mslk.flydsl.jit import run_compiled
    from mslk.flydsl.kernels.gemm.fp8_groupwise_wide_gemm import (
        compile_groupwise_wide_gemm,
    )

    launcher = compile_groupwise_wide_gemm(
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=_SCALE_BLOCK,
        waves_m=waves_m,
        waves_n=waves_n,
        # 0 means the compiler picks.
        waves_per_eu=None if waves_per_eu <= 0 else waves_per_eu,
    )
    # The kernel addresses the operands as flat byte buffers; FP8 is viewed as
    # int8 for the handoff, as the grouped path does.
    run_compiled(
        launcher,
        out,
        XQ.contiguous().view(torch.int8),
        WQ.contiguous().view(torch.int8),
        x_scale.contiguous(),
        w_scale.contiguous(),
        XQ.shape[0],
        n,
        k,
        torch.cuda.current_stream(),
    )
    return out


_tuned_gfx950 = tunable(configs=_TILES, default=DEFAULT_TILE, key=_KEY, prune=_PRUNE)(
    _launch_gfx950
)


def _matmul_gfx950(XQ, WQ, x_scale, w_scale, M, N, K):
    """Serve the op from the kernel written for it."""
    # The grid covers every row and column of the output, so nothing is left
    # unwritten and the buffer does not have to start zeroed.
    out = torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)
    return _tuned_gfx950(XQ, WQ, x_scale, w_scale, out, next_pow2(M), N, K)


@functools.lru_cache(maxsize=1)
def is_supported() -> bool:
    """Whether this module can serve the op on the current GPU.

    Both kernels below are built on MFMA, which the RDNA parts do not have --
    and FlyDSL reports itself available on those, so having the backend says
    nothing about whether this op can run. Callers pick an implementation with
    this rather than with backend availability; ``mslk/gemm/__init__.py`` falls
    back to Triton when it is False.

    Resolved once rather than per call: the architecture cannot change within a
    process, and reading it costs a device-property lookup that dwarfs a tensor
    attribute access.
    """
    return is_gfx950() or is_gfx942()


@functools.lru_cache(maxsize=1)
def _kernel() -> Callable[..., torch.Tensor]:
    """Which kernel serves this op on the current GPU."""
    if is_gfx950():
        return _matmul_gfx950
    if is_gfx942():
        return _matmul_gfx942
    raise RuntimeError(
        "mslk::f8f8bf16_groupwise on ROCm is implemented for gfx942 and gfx950; "
        "this GPU is neither. Reach it through torch.ops.mslk, which falls back "
        "to Triton elsewhere."
    )


def matmul_f8f8bf16_groupwise(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 groupwise-scaled GEMM -> BF16."""
    assert XQ.ndim == 2, f"XQ must be [M, K], got {XQ.shape}"
    assert WQ.ndim == 2, f"WQ must be [N, K], got {WQ.shape}"
    M, K = XQ.shape
    N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    grouped_dispatch.assert_fp8_operands(XQ, WQ)
    # A scale covers a whole block, so N and K have to span whole ones: a
    # partial block at either end falls outside the count and misindexes the
    # scales from there on. The CUDA implementation fixes the same granularity.
    assert N % _SCALE_BLOCK == 0 and K % _SCALE_BLOCK == 0, (
        f"n ({N}) and k ({K}) must be multiples of the {_SCALE_BLOCK}-element "
        "scale block under block scaling"
    )
    scale_k, scale_n = K // _SCALE_BLOCK, N // _SCALE_BLOCK
    assert x_scale.numel() == scale_k * M, (
        f"x_scale must be [{scale_k}, {M}], got {tuple(x_scale.shape)}"
    )
    assert w_scale.numel() == scale_k * scale_n, (
        f"w_scale must be [{scale_k}, {scale_n}], got {tuple(w_scale.shape)}"
    )

    return _kernel()(XQ, WQ, x_scale, w_scale, M, N, K)


# This module deliberately does not register the op. FlyDSL owns it wherever it
# is opted into and Triton is the fallback, but only one CUDA implementation can
# win, so the choice is arbitrated in mslk/gemm/__init__.py on first call --
# which also keeps registration from importing FlyDSL.
