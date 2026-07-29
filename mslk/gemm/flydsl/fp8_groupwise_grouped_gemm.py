# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 groupwise-scaled grouped GEMM via FlyDSL.

Registers two ops, both backed by the same kernel:

* ``mslk::f8f8bf16_groupwise_grouped`` -- the ROCm implementation of the plain
  op, taking row-major ``[G, N, K]`` weights.
* ``mslk::f8f8bf16_groupwise_grouped_preshuffle`` -- a sibling that consumes
  weights already in the MFMA B-preshuffle layout (see
  ``mslk.quantize.shuffle.preshuffle_b_mfma``). Callers shuffle once at load
  time; the op does no shuffling.

Tensor contract:
  XQ      : [TotalM, K]             FP8  -- all groups concatenated along M
  WQ      : [G, N, K]               FP8  -- per-group weights, MFMA-preshuffled
                                            for the preshuffle op
  x_scale :                         FP32 -- per-token per-128K scales, in the
                                            per-group block layout produced by
                                            quantize_fp8_group(m_sizes=...)
  w_scale : [G, K//128, N//128]     FP32 -- per-group per-block scales
  M_sizes : [G]                     int64 -- rows per group (sum to TotalM)
  Output  : [TotalM, N]             BF16
"""

import os

import torch
from mslk.flydsl.common import is_flydsl_available
from mslk.flydsl.jit import run_compiled
from mslk.utils.device import supports_float8_fnuz

_OP_NAME = "mslk::f8f8bf16_groupwise_grouped_preshuffle"

# Only the scale-block granularity is fixed; tile_m/tile_n/tile_k are chosen per
# call -- either by FlyDSL autotune (MSLK_AUTOTUNE_ENABLE set) or a fixed default.
_SCALE_BLOCK = 128

# Default tile when autotuning is disabled. Valid for any supported shape
# (tile_n=tile_k=128 divide every supported N/K, incl. small N=128). This is the
# CI / no-benchmark path -- matches the CUTLASS heuristic fallback tile.
_DEFAULT_TILE = (128, 128, 128)

# Candidate tile space swept by autotune. tile_n must be a multiple of
# scale_block_n=128 so a tile never straddles a weight scale block, and tile_k is
# pinned to the same granularity. Configs that do not divide a given shape are
# pruned before benchmarking, and ones that overflow LDS are rejected at compile.
_AUTOTUNE_TILES = (
    (64, 128, 128),
    (128, 128, 128),
    (256, 128, 128),
    (64, 256, 128),
    (128, 256, 128),
    (256, 256, 128),
)


def _next_pow2(x: int) -> int:
    """Smallest power of two >= x (x>=1). Buckets TotalM for the autotune key so
    nearby token counts share one tuned config -- matching the CUDA-graph capture
    buckets a server pre-captures, and bounding the pre-warm set."""
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def _launch_kernel(
    XQ, WQ, x_scale, w_scale, m_sizes, output, *, tile_m, tile_n, tile_k, b_preshuffled
):
    """Compile (cached) and launch the grouped GEMM for one tile config. Shared
    by the autotune target and the fixed-config path. Writes into `output`."""
    from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_contiguous import (
        compile_grouped_gemm_blockscale_contiguous,
    )

    TotalM, K = XQ.shape
    G, N, _ = WQ.shape
    # Grid M-extent: host-known upper bound (each group wastes at most one partial
    # tile). The kernel resolves group ownership from M_sizes and self-skips
    # surplus tiles.
    num_m_tiles = TotalM // tile_m + G
    launcher = compile_grouped_gemm_blockscale_contiguous(
        n=N,
        k=K,
        num_groups=G,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=_SCALE_BLOCK,
        scale_block_n=_SCALE_BLOCK,
        out_dtype="bf16",
        b_preshuffled=b_preshuffled,
        blockscale=True,
    )
    # Operands keep their natural shape: argument marshalling packs each memref
    # extent as int32, which a flattened view overflows at 2**31 elements. The
    # kernel addresses them as flat byte buffers regardless. FP8 is viewed as
    # int8 for the handoff.
    run_compiled(
        launcher,
        output,
        XQ.contiguous().view(torch.int8),
        WQ.contiguous().view(torch.int8),
        x_scale.contiguous(),
        w_scale.contiguous(),
        m_sizes,
        TotalM,
        N,
        K,
        G,
        num_m_tiles,
        torch.cuda.current_stream(),
    )
    return output


def _f8f8bf16_groupwise_grouped_preshuffle_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    TotalM = XQ.shape[0]
    N = WQ.shape[1]
    return XQ.new_empty((TotalM, N), dtype=torch.bfloat16)


def _autotune_target(
    XQ,
    WQ,
    x_scale,
    w_scale,
    m_sizes,
    output,
    m_bucket,
    n,
    k,
    b_preshuffled,
    *,
    tile_m,
    tile_n,
    tile_k,
):
    """FlyDSL @autotune benchmarks this per candidate tile. Keyed on
    (m_bucket, n, k, b_preshuffled): m_bucket=nextPow2(TotalM) buckets token
    counts; n/k separate the problem shapes (gate/up vs down-proj want different
    tiles); b_preshuffled distinguishes the two kernels (different B-load path,
    can't share a tuned config). Key args are otherwise passed straight through.
    tile_* arrive as Config kwargs."""
    return _launch_kernel(
        XQ,
        WQ,
        x_scale,
        w_scale,
        m_sizes,
        output,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        b_preshuffled=b_preshuffled,
    )


def _prune_tiles(configs, named_args, **kwargs):
    """Drop tile configs invalid for this shape (tile_n must divide N, tile_k
    must divide K) before benchmarking.

    FlyDSL's Autotuner calls this as ``(configs, sig_args)``; ``**kwargs`` keeps
    it compatible with the Triton-style ``(configs, named_args, **meta)`` form.
    """
    WQ = named_args.get("WQ")
    XQ = named_args.get("XQ")
    if WQ is None or XQ is None:
        return configs
    N = WQ.shape[1]
    K = XQ.shape[1]
    kept = [
        c
        for c in configs
        if N % c.kwargs["tile_n"] == 0 and K % c.kwargs["tile_k"] == 0
    ]
    return kept or configs


# Single autotuner for both B-layout variants, built lazily (flydsl.autotune only
# imports when FlyDSL is present). b_preshuffled is a KEY arg (not a Config kwarg)
# so the two kernels get separate tuned entries in one shared disk cache.
_AUTOTUNER = None


def _get_autotuner():
    global _AUTOTUNER
    if _AUTOTUNER is None:
        from flydsl.autotune import autotune, Config

        configs = [
            Config(tile_m=tm, tile_n=tn, tile_k=tk) for (tm, tn, tk) in _AUTOTUNE_TILES
        ]
        _AUTOTUNER = autotune(
            configs=configs,
            key=["m_bucket", "n", "k", "b_preshuffled"],
            prune_configs_by=_prune_tiles,
        )(_autotune_target)
    return _AUTOTUNER


def _dispatch_grouped_gemm(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
    *,
    b_preshuffled: bool,
) -> torch.Tensor:
    """Shared dispatch for both grouped ops. WQ is already in the layout the
    variant expects (MFMA-preshuffled if b_preshuffled else plain [G,N,K]).

    Tile selection follows the CUTLASS precedent: when MSLK_AUTOTUNE_ENABLE is
    set, FlyDSL autotune benchmarks the candidate tiles on a cache-miss and
    persists the winner (keyed on nextPow2(TotalM) and b_preshuffled); otherwise
    a fixed default tile is used with no benchmarking (the CI / graph-capture-safe
    path).
    """
    assert XQ.ndim == 2, f"XQ must be [TotalM, K], got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be [G, N, K], got {WQ.shape}"
    assert M_sizes.ndim == 1, f"M_sizes must be [G], got {M_sizes.shape}"
    TotalM, K = XQ.shape
    G, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    assert M_sizes.shape[0] == G, f"M_sizes length {M_sizes.shape[0]} must equal G={G}"
    # The MFMA instructions read the operands in the arch's native FP8 format, and
    # the kernel passes them through as raw bytes, so an fnuz/OCP mismatch would
    # be applied with the wrong exponent bias rather than rejected.
    expected_fp8 = (
        torch.float8_e4m3fnuz if supports_float8_fnuz() else torch.float8_e4m3fn
    )
    assert XQ.dtype == expected_fp8, f"XQ must be {expected_fp8}, got {XQ.dtype}"
    assert WQ.dtype == expected_fp8, f"WQ must be {expected_fp8}, got {WQ.dtype}"
    assert M_sizes.dtype == torch.int64, f"M_sizes must be int64, got {M_sizes.dtype}"

    output = torch.empty((TotalM, N), dtype=torch.bfloat16, device=XQ.device)
    if TotalM == 0 or N == 0 or K == 0 or G == 0:
        return output

    if os.environ.get("MSLK_AUTOTUNE_ENABLE"):
        # FlyDSL's Autotuner discards the tuned function's return value, so read
        # the result from the output buffer the kernel wrote into.
        _get_autotuner()(
            XQ,
            WQ,
            x_scale,
            w_scale,
            M_sizes,
            output,
            _next_pow2(TotalM),
            N,
            K,
            b_preshuffled,
        )
        return output

    tile_m, tile_n, tile_k = _DEFAULT_TILE
    assert N % tile_n == 0, f"N={N} must be a multiple of tile_n={tile_n}"
    assert K % tile_k == 0, f"K={K} must be a multiple of tile_k={tile_k}"
    return _launch_kernel(
        XQ,
        WQ,
        x_scale,
        w_scale,
        M_sizes,
        output,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        b_preshuffled=b_preshuffled,
    )


def matmul_f8f8bf16_groupwise_grouped_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """Preshuffled-B grouped groupwise FP8 GEMM (WQ already MFMA-preshuffled)."""
    return _dispatch_grouped_gemm(XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=True)


def matmul_f8f8bf16_groupwise_grouped(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """Plain (non-preshuffled) grouped groupwise FP8 GEMM via FlyDSL.

    Same contract as the preshuffle sibling, but WQ is plain row-major
    ``[G, N, K]``. Uses the shared kernel with ``b_preshuffled=False``, which
    stages B through LDS instead of loading it straight to registers.
    """
    return _dispatch_grouped_gemm(
        XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=False
    )


if (
    is_flydsl_available()
    and torch.version.hip is not None
    and hasattr(torch.ops, "mslk")
):
    # FlyDSL supplies the ROCm implementation of both ops; their schemas are
    # declared in csrc/gemm/gemm_ops.cpp. Skip an op whose schema is missing, as
    # in a python-only build, and tolerate a repeat import rebinding it.
    def _register(op_name, cuda_fn, meta_fn=None) -> None:
        if not hasattr(torch.ops.mslk, op_name.split("::")[1]):
            return
        try:
            torch.library.impl(op_name, "CUDA")(cuda_fn)
            if meta_fn is not None:
                torch.library.impl(op_name, "Meta")(meta_fn)
        except RuntimeError:
            pass

    _register(
        _OP_NAME,
        matmul_f8f8bf16_groupwise_grouped_preshuffle,
        _f8f8bf16_groupwise_grouped_preshuffle_meta,
    )
    _register("mslk::f8f8bf16_groupwise_grouped", matmul_f8f8bf16_groupwise_grouped)
