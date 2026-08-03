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
* ``..._preshuffle`` siblings of the first two consume weights already in the
  MFMA B-preshuffle layout (see ``mslk.quantize.shuffle.preshuffle_b_mfma``).
  Callers shuffle once at load time; the op does no shuffling.

Only ``_stacked`` and its preshuffle sibling are registered; the others are
reachable as plain functions.

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

import functools

import torch
from mslk.flydsl.common import is_flydsl_available
from mslk.gemm.flydsl import grouped_dispatch
from mslk.utils.device import supports_float8_fnuz

_PRESHUFFLE_OP_NAME = "mslk::f8f8bf16_rowwise_grouped_preshuffle"


def _assert_fp8_operands(XQ: torch.Tensor, WQ: torch.Tensor) -> None:
    """Reject a mismatched FP8 flavour.

    The MFMA instructions read the operands in the arch's native FP8 format, and
    the kernel passes them through as raw bytes, so an fnuz/OCP mismatch would be
    applied with the wrong exponent bias rather than rejected.
    """
    expected = torch.float8_e4m3fnuz if supports_float8_fnuz() else torch.float8_e4m3fn
    assert XQ.dtype == expected, f"XQ must be {expected}, got {XQ.dtype}"
    assert WQ.dtype == expected, f"WQ must be {expected}, got {WQ.dtype}"


@functools.lru_cache(maxsize=8)
def _unused_group_meta(device: torch.device) -> torch.Tensor:
    """Stand-in for the group-metadata operand under the batched layout.

    That layout carries no per-group metadata and the kernel never reads the
    argument, but the launcher's argument list is fixed at compile time. Caching
    keeps a call free of an allocation and holds the address stable, which
    CUDA-graph capture requires.
    """
    return torch.zeros((1,), dtype=torch.int32, device=device)


def _f8f8bf16_rowwise_grouped_preshuffle_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    total_M = XQ.shape[0]
    N = WQ.shape[1]
    return XQ.new_empty((total_M, N), dtype=torch.bfloat16)


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


def matmul_f8f8bf16_rowwise_grouped(
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
    # the CK implementation's separate zeroing pass.
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


def matmul_f8f8bf16_rowwise_grouped_preshuffle(
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


def _rowwise_grouped_mm_2d3d(XQ, WQ, x_scale, w_scale, offsets, out):
    """Groups packed along M, addressed by cumulative offsets rather than sizes.

    Same operand layout as the stacked variant; only the group metadata differs,
    so the kernel decodes the offsets in its resolution loop.
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
    _assert_fp8_operands(XQ, WQ)

    grouped_dispatch.dispatch(
        XQ,
        WQ,
        x_scale,
        w_scale,
        offsets,
        b_preshuffled=False,
        blockscale=False,
        layout="offsets",
        out=out,
    )
    return out


def _rowwise_grouped_mm_3d3d(XQ, WQ, x_scale, w_scale, offsets, out):
    """Plain batched GEMM: fixed per-group slabs with every row carrying data.

    This is the padded layout with nothing to mask, so the kernel skips both the
    metadata load and the epilogue's row predicate.
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
        b_preshuffled=False,
        blockscale=False,
        layout="batched",
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
    ranks = (XQ.ndim, WQ.ndim)
    if ranks == (2, 3):
        return _rowwise_grouped_mm_2d3d(XQ, WQ, x_scale, w_scale, offsets, out)
    if ranks == (3, 3):
        return _rowwise_grouped_mm_3d3d(XQ, WQ, x_scale, w_scale, offsets, out)
    if ranks in ((3, 2), (2, 2)):
        raise NotImplementedError(
            f"grouped mm with XQ {XQ.ndim}D and WQ {WQ.ndim}D groups along "
            f"{'N' if ranks == (3, 2) else 'K'}, which this kernel does not yet "
            "support"
        )
    raise ValueError(
        f"XQ must be 2D or 3D and WQ 2D or 3D, got {XQ.ndim}D and {WQ.ndim}D"
    )


if (
    is_flydsl_available()
    and torch.version.hip is not None
    and hasattr(torch.ops, "mslk")
):
    # FlyDSL supplies the ROCm implementation of both ops; their schemas are
    # declared in csrc/gemm/gemm_ops.cpp, which also leaves the _stacked slot
    # free on ROCm so this binding can take it. Skip an op whose schema is
    # missing, as in a python-only build, and tolerate a repeat import
    # rebinding it.
    def _register(op_name, cuda_fn, meta_fn=None) -> None:
        if not hasattr(torch.ops.mslk, op_name.split("::")[1]):
            return
        try:
            torch.library.impl(op_name, "CUDA")(cuda_fn)
            if meta_fn is not None:
                torch.library.impl(op_name, "Meta")(meta_fn)
        except RuntimeError:
            pass

    # _stacked already has a Meta implementation registered in mslk/gemm/_meta.py.
    _register("mslk::f8f8bf16_rowwise_grouped_stacked", matmul_f8f8bf16_rowwise_grouped)
    _register(
        _PRESHUFFLE_OP_NAME,
        matmul_f8f8bf16_rowwise_grouped_preshuffle,
        _f8f8bf16_rowwise_grouped_preshuffle_meta,
    )
