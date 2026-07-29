# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Shared host dispatch for the FlyDSL grouped FP8 GEMM ops.

One kernel serves every combination of B layout (plain or MFMA-preshuffled) and
scaling scheme (block or rowwise); this module holds the host-side work common to
all of them -- operand marshalling, grid extent, and tile selection -- so each op
module only supplies its own contract checks.

Tile selection is delegated to mslk.flydsl.autotune, which tunes only when
MSLK_AUTOTUNE_ENABLE is set and otherwise uses a fixed default.
"""

import torch
from mslk.flydsl.autotune import next_pow2, prune_by_divisibility, tunable
from mslk.flydsl.jit import run_compiled

# Scale-block granularity for the block-scaling scheme. It also sets the K-loop
# sub-block size under either scheme, so tile_k must be a multiple of it.
SCALE_BLOCK = 128

# Default tile when autotuning is disabled. Valid for any supported shape
# (tile_n = tile_k = 128 divide every supported N/K, including a small N=128).
DEFAULT_TILE = {"tile_m": 128, "tile_n": 128, "tile_k": 128}

# Candidate tiles swept by autotune. tile_k is a multiple of the K-loop sub-block
# size under either scheme. Rowwise scaling additionally allows tile_n below the
# scale block, which block scaling cannot express, so the two schemes sweep
# different sets. Tiles that overflow LDS are rejected at compile time.
_TILE_M = (64, 128, 256)
_TILE_K = (128, 256)


def _tiles(tile_ns):
    return tuple(
        {"tile_m": tm, "tile_n": tn, "tile_k": tk}
        for tm in _TILE_M
        for tn in tile_ns
        for tk in _TILE_K
    )


BLOCKSCALE_TILES = _tiles((128, 256))
ROWWISE_TILES = _tiles((64, 128, 256))

_PRUNE = prune_by_divisibility({"tile_n": "n", "tile_k": "k"})
_KEY = ["m_bucket", "n", "k", "b_preshuffled", "blockscale"]


def launch(
    XQ,
    WQ,
    x_scale,
    w_scale,
    m_sizes,
    out,
    m_bucket,
    n,
    k,
    b_preshuffled,
    blockscale,
    *,
    tile_m,
    tile_n,
    tile_k,
):
    """Compile (cached) and launch the grouped GEMM for one tile config.

    ``m_bucket`` only feeds the autotune key: bucketing total_M keeps nearby token
    counts on one tuned config. ``n``/``k`` are likewise passed for the key and
    for tile pruning, and are read back off the operands here.
    """
    from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_contiguous import (
        compile_grouped_gemm_blockscale_contiguous,
    )

    total_M, K = XQ.shape
    G, N, _ = WQ.shape
    # Grid M-extent: host-known upper bound (each group wastes at most one partial
    # tile). The kernel resolves group ownership from m_sizes and self-skips
    # surplus tiles, so this needs no device sync and holds under graph capture.
    num_m_tiles = total_M // tile_m + G
    launcher = compile_grouped_gemm_blockscale_contiguous(
        n=N,
        k=K,
        num_groups=G,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=SCALE_BLOCK,
        scale_block_n=SCALE_BLOCK,
        out_dtype="bf16",
        b_preshuffled=b_preshuffled,
        blockscale=blockscale,
    )
    # Operands keep their natural shape: argument marshalling packs each memref
    # extent as int32, which a flattened view overflows at 2**31 elements. The
    # kernel addresses them as flat byte buffers regardless. FP8 is viewed as
    # int8 for the handoff.
    run_compiled(
        launcher,
        out,
        XQ.contiguous().view(torch.int8),
        WQ.contiguous().view(torch.int8),
        x_scale.contiguous(),
        w_scale.contiguous(),
        m_sizes,
        total_M,
        N,
        K,
        G,
        num_m_tiles,
        torch.cuda.current_stream(),
    )
    return out


# The two scaling schemes sweep different tiles, so each gets its own tuned entry
# point; the cache key carries the scheme as well, since the kernels differ.
_launch_blockscale = tunable(
    configs=BLOCKSCALE_TILES, default=DEFAULT_TILE, key=_KEY, prune=_PRUNE
)(launch)
_launch_rowwise = tunable(
    configs=ROWWISE_TILES, default=DEFAULT_TILE, key=_KEY, prune=_PRUNE
)(launch)


def dispatch(XQ, WQ, x_scale, w_scale, M_sizes, *, b_preshuffled, blockscale):
    """Allocate the output and run the grouped GEMM with a selected tile.

    Callers validate their own operand contract first; this only handles the
    parts every variant shares.
    """
    total_M, K = XQ.shape
    G, N, _ = WQ.shape

    out = torch.empty((total_M, N), dtype=torch.bfloat16, device=XQ.device)
    if total_M == 0 or N == 0 or K == 0 or G == 0:
        return out

    tuned_launch = _launch_blockscale if blockscale else _launch_rowwise
    return tuned_launch(
        XQ,
        WQ,
        x_scale,
        w_scale,
        M_sizes,
        out,
        next_pow2(total_M),
        N,
        K,
        b_preshuffled,
        blockscale,
    )
