# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FP8 rowwise-scaled grouped GEMM via FlyDSL.

Every entry point here is backed by the same kernel as the groupwise sibling in
fp8_groupwise_grouped_gemm.py, compiled for rowwise scaling instead. The
variants differ only in how the caller lays out the groups, which the kernel
takes as a compile-time ``layout``:

* ``mslk::f8f8bf16_rowwise_grouped_stacked`` -- groups packed along M with a
  ``[G]`` int64 row count per group, and row-major ``[G, N, K]`` weights.
* ``mslk::f8f8bf16_rowwise_grouped_dynamic`` -- one fixed ``[G, M, K]`` slab per
  group, of which the first ``zero_start_index_M[g]`` rows hold real tokens.
* ``mslk::f8f8bf16_rowwise_grouped_mm`` -- the torch-native API, where the
  operand ranks pick the grouped axis.
* ``..._preshuffle`` siblings of all three consume weights already in the MFMA
  B-preshuffle layout (see ``mslk.quantize.shuffle.preshuffle_b_mfma``).
  Callers shuffle once at load time; the op does no shuffling. The swizzle
  interleaves N and K across the whole matrix, so it only applies where a group
  owns an entire ``[N, K]``: the mm sibling therefore serves the 2D-3D and
  3D-3D ranks and rejects the two that group along N or K.

All of these are registered on ROCm, and gemm_ops.cpp leaves their slots free
there, so nothing else implements them on this platform.

Rowwise scaling carries one scale per row of A and per column of B, both
constant along K, so they factor out of the reduction and the kernel applies
them in the epilogue.

Tensor contract:
  XQ      : [total_M, K]   FP8   -- all groups concatenated along M
  WQ      : [G, N, K]      FP8   -- per-group weights, MFMA-preshuffled for the
                                    preshuffle op
  x_scale : [total_M]      FP32  -- one scale per row of A
  w_scale : [G, N]         FP32  -- one scale per column of each group's B
  M_sizes : [G]            int64 -- rows per group (sum to total_M)
  Output  : [total_M, N]   BF16

  out[m, n] = (sum_k XQ[m, k] * WQ[g, n, k]) * x_scale[m] * w_scale[g, n]
"""

import torch
from mslk.flydsl.common import require_flydsl
from mslk.gemm.flydsl import grouped_dispatch

# Both checks are common to every FlyDSL GEMM wrapper, so they live beside the
# dispatch; these names are kept for the call sites below.
_assert_fp8_operands = grouped_dispatch.assert_fp8_operands
_unused_group_meta = grouped_dispatch.unused_group_meta


def _dispatch_rowwise_grouped(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
    *,
    b_preshuffled: bool,
) -> torch.Tensor:
    """Shared dispatch for both rowwise ops. WQ is already in the layout the
    variant expects (MFMA-preshuffled if b_preshuffled else plain [G,N,K]).
    """
    # Registration does not probe for FlyDSL, so this is the first point at
    # which it is required.
    require_flydsl()
    assert XQ.ndim == 2, f"XQ must be [total_M, K], got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be [G, N, K], got {WQ.shape}"
    assert M_sizes.ndim == 1, f"M_sizes must be [G], got {M_sizes.shape}"
    total_M, K = XQ.shape
    G, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    assert M_sizes.shape[0] == G, f"M_sizes length {M_sizes.shape[0]} must equal G={G}"
    _assert_fp8_operands(XQ, WQ)
    assert M_sizes.dtype == torch.int64, f"M_sizes must be int64, got {M_sizes.dtype}"
    assert x_scale.numel() == total_M, (
        f"x_scale must hold one scale per row ({total_M}), got {x_scale.numel()}"
    )
    assert w_scale.numel() == G * N, (
        f"w_scale must hold one scale per group column ({G * N}), got {w_scale.numel()}"
    )

    return grouped_dispatch.dispatch(
        XQ,
        WQ,
        x_scale,
        w_scale,
        M_sizes,
        b_preshuffled=b_preshuffled,
        blockscale=False,
    )


def matmul_f8f8bf16_rowwise_grouped_stacked(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """FP8 rowwise-scaled grouped GEMM -> BF16, with plain row-major weights."""
    return _dispatch_rowwise_grouped(
        XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=False
    )


def _dispatch_rowwise_grouped_dynamic(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    zero_start_index_M: torch.Tensor,
    zeroing_output_tensor: bool,
    *,
    b_preshuffled: bool,
) -> torch.Tensor:
    """Shared dispatch for the padded-layout ops.

    Each group owns a fixed slab of ``expected_m`` rows and only the first
    ``zero_start_index_M[g]`` of them hold real tokens, so the caller never has to
    compact tokens into one buffer. The slabs are contiguous, so the kernel sees
    the flattened ``[G * expected_m, ...]`` views and treats the group as a grid
    axis rather than resolving it from row counts.
    """
    # Registration does not probe for FlyDSL, so this is the first point at
    # which it is required.
    require_flydsl()
    assert XQ.ndim == 3, f"XQ must be [G, M, K], got {XQ.shape}"
    assert WQ.ndim == 3, f"WQ must be [G, N, K], got {WQ.shape}"
    assert zero_start_index_M.ndim == 1, (
        f"zero_start_index_M must be [G], got {zero_start_index_M.shape}"
    )
    G, expected_m, K = XQ.shape
    Gw, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    assert Gw == G, f"group mismatch: XQ G={G}, WQ G={Gw}"
    assert zero_start_index_M.shape[0] == G, (
        f"zero_start_index_M length {zero_start_index_M.shape[0]} must equal G={G}"
    )
    _assert_fp8_operands(XQ, WQ)
    assert zero_start_index_M.dtype == torch.int64, (
        f"zero_start_index_M must be int64, got {zero_start_index_M.dtype}"
    )
    assert x_scale.numel() == G * expected_m, (
        f"x_scale must hold one scale per row ({G * expected_m}), got {x_scale.numel()}"
    )
    assert w_scale.numel() == G * N, (
        f"w_scale must hold one scale per group column ({G * N}), got {w_scale.numel()}"
    )

    # Rows past a group's valid count are never written, so they carry whatever the
    # buffer already held. Zero them up front when the caller asks for it, matching
    # the CK implementation's separate zeroing pass. A shape that reduces over
    # nothing is zeroed either way, since zero is then the whole answer.
    alloc = torch.zeros if zeroing_output_tensor else torch.empty
    out = alloc((G, expected_m, N), dtype=torch.bfloat16, device=XQ.device)

    grouped_dispatch.dispatch(
        XQ.contiguous().view(G * expected_m, K),
        WQ,
        x_scale,
        w_scale,
        zero_start_index_M,
        b_preshuffled=b_preshuffled,
        blockscale=False,
        layout="padded",
        out=out.view(G * expected_m, N),
    )
    return out


def matmul_f8f8bf16_rowwise_grouped_dynamic(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    zero_start_index_M: torch.Tensor,
    zeroing_output_tensor: bool = True,
) -> torch.Tensor:
    """Padded-layout rowwise grouped GEMM with plain row-major weights."""
    return _dispatch_rowwise_grouped_dynamic(
        XQ,
        WQ,
        x_scale,
        w_scale,
        zero_start_index_M,
        zeroing_output_tensor,
        b_preshuffled=False,
    )


def matmul_f8f8bf16_rowwise_grouped_dynamic_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    zero_start_index_M: torch.Tensor,
    zeroing_output_tensor: bool = True,
) -> torch.Tensor:
    """Preshuffled-B padded-layout rowwise grouped GEMM (WQ already
    MFMA-preshuffled).

    Loads B straight to registers rather than staging it through LDS, in
    exchange for the caller shuffling the weights once at load time.
    """
    return _dispatch_rowwise_grouped_dynamic(
        XQ,
        WQ,
        x_scale,
        w_scale,
        zero_start_index_M,
        zeroing_output_tensor,
        b_preshuffled=True,
    )


def matmul_f8f8bf16_rowwise_grouped_stacked_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """Preshuffled-B rowwise grouped GEMM (WQ already MFMA-preshuffled).

    Loads B straight to registers rather than staging it through LDS, in
    exchange for the caller shuffling the weights once at load time.
    """
    return _dispatch_rowwise_grouped(
        XQ, WQ, x_scale, w_scale, M_sizes, b_preshuffled=True
    )


def _rowwise_grouped_mm_2d3d(
    XQ, WQ, x_scale, w_scale, offsets, out, b_preshuffled=False
):
    """Groups packed along M, addressed by cumulative offsets rather than sizes.

    Same operand layout as the stacked variant; only the group metadata differs,
    so the kernel decodes the offsets in its resolution loop.

    The groups divide M, leaving each one a whole [N, K] of B to itself, so the
    MFMA swizzle applies here exactly as it does to the stacked variant.
    """
    assert offsets is not None, "2D-3D grouped mm requires offsets for XQ"
    assert offsets.dtype == torch.int32, f"offsets must be int32, got {offsets.dtype}"
    total_M, K = XQ.shape
    G, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    assert offsets.shape[0] == G, f"offsets length {offsets.shape[0]} must equal G={G}"
    assert x_scale.numel() == total_M, (
        f"x_scale must hold one scale per row ({total_M}), got {x_scale.numel()}"
    )
    assert w_scale.numel() == G * N, (
        f"w_scale must hold one scale per group column ({G * N}), got {w_scale.numel()}"
    )
    assert tuple(out.shape) == (total_M, N), (
        f"out must be [total_M, N] = {(total_M, N)}, got {tuple(out.shape)}"
    )
    assert out.is_contiguous(), "out must be contiguous"
    _assert_fp8_operands(XQ, WQ)

    grouped_dispatch.dispatch(
        XQ,
        WQ,
        x_scale,
        w_scale,
        offsets,
        b_preshuffled=b_preshuffled,
        blockscale=False,
        layout="offsets",
        out=out,
    )
    return out


def _rowwise_grouped_mm_3d3d(
    XQ, WQ, x_scale, w_scale, offsets, out, b_preshuffled=False
):
    """Plain batched GEMM: fixed per-group slabs with every row carrying data.

    This is the padded layout with nothing to mask, so the kernel skips both the
    metadata load and the epilogue's row predicate.

    Nothing is grouped along N or K, so B is a whole [N, K] per group and takes
    the MFMA swizzle unchanged.
    """
    assert offsets is None, "3D-3D is a batched GEMM and takes no offsets"
    G, M, K = XQ.shape
    Gw, N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    assert Gw == G, f"group mismatch: XQ G={G}, WQ G={Gw}"
    assert x_scale.numel() == G * M, (
        f"x_scale must hold one scale per row ({G * M}), got {x_scale.numel()}"
    )
    assert w_scale.numel() == G * N, (
        f"w_scale must hold one scale per group column ({G * N}), got {w_scale.numel()}"
    )
    assert tuple(out.shape) == (G, M, N), (
        f"out must be [G, M, N] = {(G, M, N)}, got {tuple(out.shape)}"
    )
    # The kernel writes out in place, so it has to be the caller's buffer; a
    # contiguity fixup would silently write to a copy instead.
    assert out.is_contiguous(), "out must be contiguous"
    _assert_fp8_operands(XQ, WQ)

    grouped_dispatch.dispatch(
        XQ.contiguous().view(G * M, K),
        WQ,
        x_scale,
        w_scale,
        _unused_group_meta(XQ.device),
        b_preshuffled=b_preshuffled,
        blockscale=False,
        layout="batched",
        out=out.view(G * M, N),
    )
    return out


def _rowwise_grouped_mm_3d2d(XQ, WQ, x_scale, w_scale, offsets, out):
    """Groups divide the output's columns rather than its rows.

    Each group still multiplies its own activation slab by its own weights; the
    results are packed side by side into [M, total_N] instead of end to end, so
    the weights arrive as one [total_N, K] matrix whose rows the offsets divide.

    Every group's column count must be a multiple of 8, matching the bound CK
    asserts, so that no store straddles a group boundary; the epilogue's widest
    store covers four columns, so four is what the kernel itself needs. The
    offsets live on the device, so like CK this cannot be checked on the host.
    Unlike CK, which asserts on the device, a violation here leaves the
    straddling columns unwritten rather than aborting: the store that would
    cross the boundary is dropped, so the next group's columns are left to it.
    """
    assert offsets is not None, "3D-2D grouped mm requires offsets for WQ"
    assert offsets.dtype == torch.int32, f"offsets must be int32, got {offsets.dtype}"
    G, M, K = XQ.shape
    total_N, Kw = WQ.shape
    assert Kw == K, f"K mismatch: XQ K={K}, WQ K={Kw}"
    assert offsets.shape[0] == G, f"offsets length {offsets.shape[0]} must equal G={G}"
    assert x_scale.numel() == G * M, (
        f"x_scale must hold one scale per row ({G * M}), got {x_scale.numel()}"
    )
    assert w_scale.numel() == total_N, (
        f"w_scale must hold one scale per column ({total_N}), got {w_scale.numel()}"
    )
    assert tuple(out.shape) == (M, total_N), (
        f"out must be [M, total_N] = {(M, total_N)}, got {tuple(out.shape)}"
    )
    assert out.is_contiguous(), "out must be contiguous"
    _assert_fp8_operands(XQ, WQ)

    grouped_dispatch.dispatch(
        XQ.contiguous().view(G * M, K),
        WQ,
        x_scale,
        w_scale,
        offsets,
        b_preshuffled=False,
        blockscale=False,
        layout="n_offsets",
        out=out,
    )
    return out


def _rowwise_grouped_mm_2d2d(XQ, WQ, x_scale, w_scale, offsets, out):
    """Groups divide the contraction rather than either output axis.

    Every group multiplies the same M rows by the same N columns but over its
    own slice of K, so each produces a whole [M, N] of its own and the output is
    [G, M, N] with nothing packed. The operands are one matrix each, the groups
    taking column slices of them.

    Every group's K length must be a multiple of 16, the vectorised load width.
    The offsets live on the device, so like CK this cannot be checked on the
    host; CK does not check it anywhere, and is silently wrong below it.

    A group whose K slice is empty contributes nothing, and its slab of ``out``
    is left as the caller passed it -- the same as CK, which skips such groups.
    Where every group is empty the whole contraction has length zero, which sums
    to zero, and ``out`` is zeroed rather than left alone.
    """
    assert offsets is not None, "2D-2D grouped mm requires offsets for K"
    assert offsets.dtype == torch.int32, f"offsets must be int32, got {offsets.dtype}"
    M, total_K = XQ.shape
    N, Kw = WQ.shape
    assert Kw == total_K, f"K mismatch: XQ K={total_K}, WQ K={Kw}"
    G = offsets.shape[0]
    assert x_scale.numel() == G * M, (
        f"x_scale must hold one scale per row per group ({G * M}), "
        f"got {x_scale.numel()}"
    )
    assert w_scale.numel() == G * N, (
        f"w_scale must hold one scale per column per group ({G * N}), "
        f"got {w_scale.numel()}"
    )
    assert tuple(out.shape) == (G, M, N), (
        f"out must be [G, M, N] = {(G, M, N)}, got {tuple(out.shape)}"
    )
    assert out.is_contiguous(), "out must be contiguous"
    _assert_fp8_operands(XQ, WQ)

    grouped_dispatch.dispatch(
        XQ,
        WQ,
        x_scale,
        w_scale,
        offsets,
        b_preshuffled=False,
        blockscale=False,
        layout="k_offsets",
        out=out.view(G * M, N),
    )
    return out


def matmul_f8f8bf16_rowwise_grouped_mm(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    offsets: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """Torch-native grouped GEMM, where the operand ranks pick the grouped axis.

    2D-3D groups along M, 3D-2D along N, 2D-2D along K, and 3D-3D is an ordinary
    batched GEMM. ``offsets`` gives the cumulative int32 end of each group along
    whichever axis is grouped, and is None for 3D-3D. ``out`` is written in place
    and returned.
    """
    # Registration does not probe for FlyDSL, so this is the first point at
    # which it is required.
    require_flydsl()
    ranks = (XQ.ndim, WQ.ndim)
    if ranks == (2, 3):
        return _rowwise_grouped_mm_2d3d(XQ, WQ, x_scale, w_scale, offsets, out)
    if ranks == (3, 3):
        return _rowwise_grouped_mm_3d3d(XQ, WQ, x_scale, w_scale, offsets, out)
    if ranks == (3, 2):
        return _rowwise_grouped_mm_3d2d(XQ, WQ, x_scale, w_scale, offsets, out)
    if ranks == (2, 2):
        return _rowwise_grouped_mm_2d2d(XQ, WQ, x_scale, w_scale, offsets, out)
    raise ValueError(
        f"XQ must be 2D or 3D and WQ 2D or 3D, got {XQ.ndim}D and {WQ.ndim}D"
    )


def matmul_f8f8bf16_rowwise_grouped_mm_preshuffle(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    offsets: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """Sibling of the grouped mm taking weights already MFMA-preshuffled.

    Only the rank combinations that leave B whole are served: 2D-3D groups along
    M and 3D-3D groups nothing, so each group owns an entire [N, K] and the
    swizzle applies to it as it stands. Grouping along N or K cuts across the
    axes the swizzle interleaves, so 3D-2D and 2D-2D have no preshuffled form
    and are rejected rather than silently given the plain path.
    """
    # Registration does not probe for FlyDSL, so this is the first point at
    # which it is required.
    require_flydsl()
    ranks = (XQ.ndim, WQ.ndim)
    if ranks == (2, 3):
        return _rowwise_grouped_mm_2d3d(
            XQ, WQ, x_scale, w_scale, offsets, out, b_preshuffled=True
        )
    if ranks == (3, 3):
        return _rowwise_grouped_mm_3d3d(
            XQ, WQ, x_scale, w_scale, offsets, out, b_preshuffled=True
        )
    if ranks in ((3, 2), (2, 2)):
        axis = "N" if ranks == (3, 2) else "K"
        raise ValueError(
            f"{ranks[0]}D-{ranks[1]}D groups along {axis}, which the MFMA "
            f"B-preshuffle layout interleaves across the whole matrix, so a "
            f"group boundary falls inside the swizzle. Use "
            f"matmul_f8f8bf16_rowwise_grouped_mm with plain weights instead"
        )
    raise ValueError(
        f"XQ must be 2D or 3D and WQ 2D or 3D, got {XQ.ndim}D and {WQ.ndim}D"
    )


# This module registers nothing. All six ops are registered in
# mslk/gemm/__init__.py, whose impls import this module on the first call.
# Keeping registration out of here is what lets //mslk:gemm_ops avoid depending
# on //mslk/mslk/gemm:flydsl_ops, and so keeps the FlyDSL wheel out of every
# binary that merely imports mslk.gemm. Shape inference lives in
# mslk/gemm/_meta.py for the same reason.
