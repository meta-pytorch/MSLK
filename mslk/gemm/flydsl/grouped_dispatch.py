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
DEFAULT_TILE = {"tile_m": 128, "tile_n": 128, "tile_k": 128, "waves_per_eu": 2}

# Candidate tiles swept by autotune. tile_k is a multiple of the K-loop sub-block
# size under either scheme. Rowwise scaling additionally allows tile_n below the
# scale block, which block scaling cannot express, so the two schemes sweep
# different sets. Tiles that overflow LDS are rejected at compile time.
_TILE_M = (64, 128, 256)
_TILE_K = (128, 256)

# Occupancy target, as a minimum waves-per-EU hint to the register allocator; 0
# leaves the choice to the compiler. Preshuffled B is held in registers across a
# whole pair of K tiles to cover HBM latency, which puts it right on the 256-VGPR
# boundary between two waves per SIMD and one, so a shape can land either side of
# it. Only these two are worth sweeping: three waves needs 170 registers and four
# needs 128, which no configuration of this kernel comes close to.
_WAVES_PER_EU = (0, 2)


def _tiles(tile_ns):
    return tuple(
        {"tile_m": tm, "tile_n": tn, "tile_k": tk, "waves_per_eu": wpe}
        for tm in _TILE_M
        for tn in tile_ns
        for tk in _TILE_K
        for wpe in _WAVES_PER_EU
    )


BLOCKSCALE_TILES = _tiles((128, 256))
ROWWISE_TILES = _tiles((64, 128, 256))

# A tile that overruns N or K still compiles, as the tail-masked variant, but it
# wastes part of its work on padding and is not going to win, so prune on both
# axes. When nothing divides, which is the case that needs the padding,
# prune_by_divisibility falls back to the full list and the shape still gets
# tuned.
_PRUNE = prune_by_divisibility({"tile_n": "n", "tile_k": "k"})
# roll_k is deliberately absent: it is fixed policy rather than something that
# varies per call, and a tuning space containing a fully unrolled candidate would
# have to compile one per tile config, at a cost that grows with K.
_KEY = ["m_bucket", "n", "k", "b_preshuffled", "blockscale", "layout"]


def _group_and_n(WQ, group_meta, layout):
    """Group count and total N, which the weights only carry for some layouts.

    Weights are a stack of per-group [N, K] matrices except where the groups
    divide N, in which case they are one [total_N, K] matrix and the group count
    comes from the offsets instead.
    """
    if layout in ("n_offsets", "k_offsets"):
        # One matrix rather than a stack, so the group count comes from the
        # offsets; N is its row count either way.
        return group_meta.shape[0], WQ.shape[0]
    return WQ.shape[0], WQ.shape[1]


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
    layout="sizes",
    roll_k=True,
    *,
    tile_m,
    tile_n,
    tile_k,
    waves_per_eu=0,
):
    """Compile (cached) and launch the grouped GEMM for one tile config.

    ``XQ`` is [total_M, K] with groups packed along M, or the flattened
    [G * expected_m, K] view of the per-group slabs. ``layout`` says which, and
    how ``m_sizes`` encodes the group geometry; see the kernel factory.

    ``m_bucket`` only feeds the autotune key: bucketing total_M keeps nearby token
    counts on one tuned config. ``n``/``k`` are likewise passed for the key and
    for tile pruning, and are read back off the operands here.
    """
    from mslk.flydsl.kernels.gemm.fp8_grouped_gemm import compile_fp8_grouped_gemm

    total_M, K = XQ.shape
    G, N = _group_and_n(WQ, m_sizes, layout)
    if b_preshuffled and (K % tile_k != 0 or N % tile_n != 0):
        raise ValueError(
            f"n ({N}) and k ({K}) must be divisible by tile_n ({tile_n}) and "
            f"tile_k ({tile_k}) for preshuffled B: the MFMA B layout interleaves "
            "both, so a partial tile cannot be masked a load at a time"
        )
    if layout == "k_offsets":
        # Every group produces a whole output, so the grid covers M exactly.
        num_m_tiles = -(-total_M // tile_m)
    elif layout in ("padded", "batched", "n_offsets"):
        # Each group owns a slab, so the M axis only spans a single one. Under
        # n_offsets the group rides the N axis instead of z, but its rows are
        # still one slab.
        num_m_tiles = -(-(total_M // G) // tile_m)
    else:
        # Grid M-extent: host-known upper bound (each group wastes at most one
        # partial tile). The kernel resolves group ownership from m_sizes and
        # self-skips surplus tiles, so this needs no device sync and holds under
        # graph capture.
        num_m_tiles = total_M // tile_m + G
    launcher = compile_fp8_grouped_gemm(
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
        layout=layout,
        roll_k=roll_k,
        # 0 means the compiler picks.
        waves_per_eu=None if waves_per_eu <= 0 else waves_per_eu,
        # Compile the tail-masked variant only when K stops mid-tile, so shapes
        # that divide keep the cheaper unmasked loads. This mirrors how CK picks
        # between its KPadding and Default specialisations on the host.
        k_padding=(K % tile_k != 0),
        # A group's column end is a runtime value when the groups divide N, so
        # the tail mask is always needed there.
        n_padding=(N % tile_n != 0) or layout == "n_offsets",
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
        m_sizes.contiguous(),
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


def dispatch(
    XQ,
    WQ,
    x_scale,
    w_scale,
    M_sizes,
    *,
    b_preshuffled,
    blockscale,
    layout="sizes",
    roll_k=True,
    out=None,
):
    """Allocate the output if needed and run the grouped GEMM with a selected tile.

    Callers validate their own operand contract first; this only handles the
    parts every variant shares. ``XQ``/``out`` are the flattened 2D views in the
    slab layouts, so the shape handling below is common to all of them.

    An allocated output is [total_M, N], which is its shape only where the
    groups divide M or own a slab of it. Where they divide N or K it has a
    different shape, so those layouts pass ``out`` rather than rely on this.
    """
    total_M, K = XQ.shape
    G, N = _group_and_n(WQ, M_sizes, layout)

    if out is None:
        out = torch.empty((total_M, N), dtype=torch.bfloat16, device=XQ.device)
    if total_M == 0 or N == 0 or K == 0 or G == 0:
        # The kernel does not launch, so nothing else writes the output. A
        # contraction over nothing sums to zero, and where the output holds no
        # elements at all this is a no-op.
        return out.zero_()

    # Tune on the shape of one group rather than of the concatenation, so that
    # the key describes the work a block actually does. Only the grouped axis
    # needs normalising; the others are already per-group.
    if layout == "n_offsets":
        m_key, n_key = total_M // G, N // G
    else:
        m_key, n_key = total_M, N
    # The groups divide K, so one group contracts over a fraction of it.
    k_key = K // G if layout == "k_offsets" else K

    tuned_launch = _launch_blockscale if blockscale else _launch_rowwise
    return tuned_launch(
        XQ,
        WQ,
        x_scale,
        w_scale,
        M_sizes,
        out,
        next_pow2(m_key),
        n_key,
        k_key,
        b_preshuffled,
        blockscale,
        layout,
        roll_k,
    )
