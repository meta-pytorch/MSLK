# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Shared host dispatch for the FlyDSL grouped GEMM ops.

One kernel serves every combination of input dtype (fp8 or bf16), B layout
(plain or MFMA-preshuffled) and scaling scheme (block, rowwise or none); this
module holds the host-side work common to all of them -- operand marshalling,
grid extent, and tile selection -- so each op module only supplies its own
contract checks.

Tile selection is delegated to mslk.flydsl.autotune, which tunes only when
MSLK_AUTOTUNE_ENABLE is set and otherwise uses a fixed default.
"""

import functools

import torch
from mslk.flydsl.autotune import next_pow2, prune_by_divisibility, tunable
from mslk.flydsl.jit import run_compiled
from mslk.utils.device import supports_float8_fnuz

# Scale-block granularity for the block-scaling scheme. Where there are scales it
# also sets the K-loop sub-block size, so tile_k must then be a multiple of it.
SCALE_BLOCK = 128

# Default config when autotuning is disabled, for the scaled schemes; the
# unscaled one takes UNSCALED_DEFAULT_TILE below. Valid for any supported shape
# (tile_n = tile_k = 128 divide every supported N/K, including a small N=128).
DEFAULT_TILE = {
    "tile_m": 128,
    "tile_n": 128,
    "tile_k": 128,
    "waves_m": 1,
    "waves_n": 4,
    "waves_per_eu": 2,
}

# The unscaled scheme takes a smaller tile_k, since a 2-byte operand doubles what
# a tile occupies in LDS: 128x128x128 in BF16 needs 2*128*128*2 for the ping-pong
# A plus 128*128*2 for B, which is 98304 bytes against the 65536 a CDNA3
# workgroup has. tile_k=64 brings it to 49152 and fits both CDNA3 and CDNA4. Only
# this scheme can move tile_k; where there are scales it is tied to the scale
# block.
UNSCALED_DEFAULT_TILE = {**DEFAULT_TILE, "tile_k": 64}

# Candidate tiles swept by autotune. Rowwise scaling allows tile_n below the
# scale block, which block scaling cannot express, so the two schemes sweep
# different sets. Tiles that overflow LDS are rejected at compile time.
_TILE_M = (64, 128, 256)
_TILE_K = (128, 256)

# Unscaled operands sweep wider on both axes. tile_k is not pinned to the scale
# block, and a two-byte dtype needs the room: its tile spans twice the LDS of the
# one-byte tile of the same shape, so at tile_k=128 a 128x128 tile already holds
# a whole CU's worth and runs one workgroup at a time. Shrinking tile_k buys that
# back most cheaply, since LDS grows as (tile_m + tile_n) * tile_k while the MFMA
# work grows as tile_m * tile_n * tile_k. tile_m=32 is the other end of the same
# trade, for the decode shapes where a group holds only a few rows.
_UNSCALED_TILE_M = (32, 64, 128, 256)
_UNSCALED_TILE_K = (64, 128, 256)

# Occupancy target, as a minimum waves-per-EU hint to the register allocator; 0
# leaves the choice to the compiler. Preshuffled B is held in registers across a
# whole pair of K tiles to cover HBM latency, which puts it right on the 256-VGPR
# boundary between two waves per SIMD and one, so a shape can land either side of
# it. Only these two are worth sweeping: three waves needs 170 registers and four
# needs 128, which no configuration of this kernel comes close to.
_WAVES_PER_EU = (0, 2)

# How the block's four waves divide the tile, as (waves_m, waves_n). Each wave
# reads its whole slab of both operands out of LDS, so reads per unit work go as
# waves_m / tile_m + waves_n / tile_n, smallest when the grid is proportioned
# like the tile: 1x4 suits a tile four times wider than tall, 2x2 a square one,
# 4x1 a tall one. All three are four waves, so they cost the same in threads,
# LDS and registers and differ only in shape.
_WAVE_GRIDS = ((1, 4), (2, 2), (4, 1))


def _tiles(tile_ns, tile_ms=_TILE_M, tile_ks=_TILE_K, wave_grids=_WAVE_GRIDS):
    # The wave grid has to cut the tile into whole 16x16 MFMA tiles, the same
    # rule the kernel factory enforces. Applying it here keeps the tuning space
    # free of configs that would only be built and thrown away.
    return tuple(
        {
            "tile_m": tm,
            "tile_n": tn,
            "tile_k": tk,
            "waves_m": wm,
            "waves_n": wn,
            "waves_per_eu": wpe,
        }
        for tm in tile_ms
        for tn in tile_ns
        for tk in tile_ks
        for wm, wn in wave_grids
        for wpe in _WAVES_PER_EU
        if tm % (wm * 16) == 0 and tn % (wn * 16) == 0
    )


BLOCKSCALE_TILES = _tiles((128, 256))
ROWWISE_TILES = _tiles((64, 128, 256))
# tile_n stops at 64: the CShuffle epilogue lays 32 lanes across N at 2 columns
# each, so a narrower tile has no store to make.
UNSCALED_TILES = _tiles(
    (64, 128, 256), tile_ms=_UNSCALED_TILE_M, tile_ks=_UNSCALED_TILE_K
)

# A tile that overruns N or K still compiles, as the tail-masked variant, but it
# wastes part of its work on padding and is not going to win, so prune on both
# axes. When nothing divides, which is the case that needs the padding,
# prune_by_divisibility falls back to the full list and the shape still gets
# tuned.
_PRUNE = prune_by_divisibility({"tile_n": "n", "tile_k": "k"})
# roll_k is deliberately absent: it is fixed policy rather than something that
# varies per call, and a tuning space containing a fully unrolled candidate would
# have to compile one per tile config, at a cost that grows with K.
_KEY = ["m_bucket", "n", "k", "g", "b_preshuffled", "scaling", "layout", "in_dtype"]


def assert_fp8_operands(XQ: torch.Tensor, WQ: torch.Tensor) -> None:
    """Reject a mismatched FP8 flavour.

    The MFMA instructions read the operands in the arch's native FP8 format, and
    the kernel passes them through as raw bytes, so an fnuz/OCP mismatch would be
    applied with the wrong exponent bias rather than rejected.

    Every wrapper has to make this check, so it sits beside the dispatch rather
    than in any one of them.
    """
    expected = torch.float8_e4m3fnuz if supports_float8_fnuz() else torch.float8_e4m3fn
    assert XQ.dtype == expected, f"XQ must be {expected}, got {XQ.dtype}"
    assert WQ.dtype == expected, f"WQ must be {expected}, got {WQ.dtype}"


@functools.lru_cache(maxsize=8)
def unused_group_meta(device: torch.device) -> torch.Tensor:
    """Stand-in for the group-metadata operand under the batched layout.

    That layout carries no per-group metadata and the kernel never reads the
    argument, but the launcher's argument list is fixed at compile time. Caching
    keeps a call free of an allocation and holds the address stable, which
    CUDA-graph capture requires.

    Every wrapper that reaches the batched layout needs one, so it lives next to
    the dispatch it is an argument to.
    """
    return torch.zeros((1,), dtype=torch.int32, device=device)


@functools.lru_cache(maxsize=8)
def unused_scales(device: torch.device) -> torch.Tensor:
    """Stand-in for the scale operands where the operands carry no scales.

    Unscaled inputs have none to pass and the kernel emits no load against
    them, but the launcher's argument list is fixed at compile time. Cached for
    the same reasons as `unused_group_meta`: no per-call allocation, and a
    stable address for CUDA-graph capture.
    """
    return torch.zeros((1,), dtype=torch.float32, device=device)


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


#: What one AMD buffer descriptor can address. Its num_records is a 32-bit
#: field, and so is the voffset a buffer_load adds, so nothing beyond this is
#: reachable through one of them.
_BUFFER_LIMIT_BYTES = 1 << 32


def _addressing_plan(total_M, N, K, G, elem_bytes, layout):
    """Decide whether A and D need per-group basing, and reject the unreachable.

    Every operand keeps addressing its whole self while that self still fits one
    descriptor, and switches to its group's base only when it does not, since
    per-group basing carries a runtime cost on every path that uses it.

    B's threshold is exact -- ``G * N * K`` is entirely host-known -- while A's
    and D's is a bound, since the packed layouts keep their group row counts on
    the device.

    Returns ``(group_based_b, group_based_rows)``. Raises where even one group
    would not fit, which no basing can rescue.

    Only extents the host knows are checked. Where the groups are packed along M
    their row counts live on the device, so a per-group extent cannot be
    computed here; bounding it by total_M would reject the many-group shapes the
    basing exists to support. Those get the basing whenever the whole operand is
    too big, and are left to the kernel from there.
    """
    d_rows = total_M // G if layout == "n_offsets" else total_M
    d_whole = d_rows * N * 2 * (G if layout == "k_offsets" else 1)
    a_whole = total_M * K * elem_bytes
    group_based_rows = a_whole >= _BUFFER_LIMIT_BYTES or d_whole >= _BUFFER_LIMIT_BYTES

    # Where the groups divide N or K, B is a single matrix every block reads
    # whole, so there is no group axis to base on and its whole extent is what
    # has to fit.
    b_stacked = layout not in ("n_offsets", "k_offsets")
    b_whole = N * K * elem_bytes * (G if b_stacked else 1)
    group_based_b = b_stacked and b_whole >= _BUFFER_LIMIT_BYTES

    checks = []
    if group_based_b or not b_stacked:
        checks.append(("B (one group's weights)", N * K * elem_bytes))
    else:
        checks.append(("B (the whole weight stack)", b_whole))
    if group_based_rows:
        if layout in ("padded", "batched", "n_offsets"):
            # Every group owns the same fixed slab, so its height is host-known.
            slab_m = total_M // G
            checks += [
                ("A (one group's rows)", slab_m * K * elem_bytes),
                ("D (one group's output)", slab_m * N * 2),
            ]
        elif layout == "k_offsets":
            # Each group produces a whole [M, N] output of its own.
            checks.append(("D (one group's output)", total_M * N * 2))
            # A is the one operand no basing can shrink here: the groups divide
            # the contraction, so a group owns a set of columns of [M, total_K]
            # and its last row sits near the end of the matrix whatever its base
            # is. The whole extent is what has to be reachable.
            checks.append(("A (the whole activation matrix)", a_whole))
    for what, nbytes in checks:
        if nbytes >= _BUFFER_LIMIT_BYTES:
            raise ValueError(
                f"{what} is {nbytes} bytes, which a buffer descriptor cannot "
                f"address (limit {_BUFFER_LIMIT_BYTES}). Shape: total_M={total_M} "
                f"N={N} K={K} G={G} at {elem_bytes} B/element, layout {layout!r}. "
                "Split the call along the axis that is too long."
            )
    return group_based_b, group_based_rows


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
    g,
    b_preshuffled,
    scaling,
    layout="sizes",
    roll_k=True,
    in_dtype="fp8",
    *,
    tile_m,
    tile_n,
    tile_k,
    waves_m=1,
    waves_n=4,
    waves_per_eu=0,
):
    """Compile (cached) and launch the grouped GEMM for one tile config.

    ``XQ`` is [total_M, K] with groups packed along M, or the flattened
    [G * expected_m, K] view of the per-group slabs. ``layout`` says which, and
    how ``m_sizes`` encodes the group geometry; see the kernel factory.

    ``m_bucket`` only feeds the autotune key: bucketing total_M keeps nearby token
    counts on one tuned config. ``n``/``k`` are likewise passed for the key and
    for tile pruning, and are read back off the operands here.

    ``g`` feeds the key alone. The group count changes both the grid, which
    carries one partial tile per group, and whether an operand clears the
    buffer-descriptor limit and has to be re-based per group, so group counts
    do not share a tuned entry.

    ``in_dtype`` names the operand element type the kernel is compiled for.
    ``x_scale``/``w_scale`` may be None where it carries no scales.
    """
    from mslk.flydsl.kernels.gemm.fp8_grouped_gemm import compile_fp8_grouped_gemm

    total_M, K = XQ.shape
    G, N = _group_and_n(WQ, m_sizes, layout)
    group_based_b, group_based_rows = _addressing_plan(
        total_M, N, K, G, XQ.element_size(), layout
    )
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
        waves_m=waves_m,
        waves_n=waves_n,
        scale_block_k=SCALE_BLOCK,
        scale_block_n=SCALE_BLOCK,
        out_dtype="bf16",
        in_dtype=in_dtype,
        b_preshuffled=b_preshuffled,
        scaling=scaling,
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
        group_based_b=group_based_b,
        group_based_rows=group_based_rows,
    )
    # Operands keep their natural shape: argument marshalling packs each memref
    # extent as int32, which a flattened view overflows at 2**31 elements. The
    # kernel addresses them as flat byte buffers regardless, so every input
    # dtype is viewed as int8 for the handoff.
    _no_scales = unused_scales(XQ.device)
    run_compiled(
        launcher,
        out,
        XQ.contiguous().view(torch.int8),
        WQ.contiguous().view(torch.int8),
        _no_scales if x_scale is None else x_scale.contiguous(),
        _no_scales if w_scale is None else w_scale.contiguous(),
        m_sizes.contiguous(),
        total_M,
        N,
        K,
        G,
        num_m_tiles,
        torch.cuda.current_stream(),
    )
    return out


# Each scaling scheme sweeps a different tile set, so each gets its own tuned
# entry point; the cache key carries the scheme as well, since the kernels differ.
_launch_blockscale = tunable(
    configs=BLOCKSCALE_TILES, default=DEFAULT_TILE, key=_KEY, prune=_PRUNE
)(launch)
_launch_rowwise = tunable(
    configs=ROWWISE_TILES, default=DEFAULT_TILE, key=_KEY, prune=_PRUNE
)(launch)
_launch_unscaled = tunable(
    configs=UNSCALED_TILES, default=UNSCALED_DEFAULT_TILE, key=_KEY, prune=_PRUNE
)(launch)


def dispatch(
    XQ,
    WQ,
    x_scale,
    w_scale,
    M_sizes,
    *,
    b_preshuffled,
    scaling,
    layout="sizes",
    roll_k=True,
    in_dtype="fp8",
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
    #
    # M is grouped wherever a group owns a slab of it, which the slab layouts
    # and the N-grouped one all do: a block then sees one slab, not the stack.
    # Packed along M it is not, since the groups differ in height and there is
    # no single per-group M to key on.
    if layout in ("padded", "batched", "n_offsets"):
        m_key = total_M // G
    else:
        m_key = total_M
    # The groups divide N, so one group owns a fraction of the columns.
    n_key = N // G if layout == "n_offsets" else N
    # The groups divide K, so one group contracts over a fraction of it.
    k_key = K // G if layout == "k_offsets" else K

    # Each scheme is free of the constraints the one before it carries: block
    # scaling pins tile_n and tile_k to the scale block, rowwise pins only
    # tile_k, and unscaled operands pin neither.
    tuned_launch = {
        "block": _launch_blockscale,
        "row": _launch_rowwise,
        "none": _launch_unscaled,
    }[scaling]
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
        G,
        b_preshuffled,
        scaling,
        layout,
        roll_k,
        in_dtype,
    )
