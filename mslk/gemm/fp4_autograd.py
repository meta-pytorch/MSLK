# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Autograd support for FP4 GEMM custom ops.

Provides differentiable ``torch.autograd.Function`` wrappers for:
  - mslk::f4f4bf16            -> f4f4bf16
  - mslk::f4f4bf16_grouped_mm -> f4f4bf16_grouped_mm
  - mslk::f4f4bf16_ultra_grouped_mm -> f4f4bf16_ultra_grouped_mm

The underlying ops are non-functional (optional in-place output/scale
buffers), so ``torch.library.register_autograd`` cannot be used; callers
should use these wrappers to get a backward pass.

Backward strategy: dequantize packed FP4 inputs to BF16, then compute
gradients via standard BF16 matmuls. FP4 (E2M1) has only 8 representable
magnitudes and cannot carry gradient signal, so this follows the same
FP8-forward / BF16-backward pattern.

Layout note for the grouped ops: ``WQ`` is passed column-major as
``(G, K/2, N)`` (2D-3D) to match ``torch._scaled_grouped_mm``, so ``WQ[g]``
is ``(K/2, N)`` and must be transposed to ``(N, K/2)`` before dequantizing,
since the packed FP4 axis is the *last* axis for the dequant helpers.
"""

from typing import Optional

import torch
from mslk.quantize.triton.legacy.fp4_utils import (
    dequantize_mx4,
    dequantize_nvfp4,
    fp4_to_float,
)
from mslk.quantize.triton.legacy.primitives import _from_blocked

# ---------------------------------------------------------------------------
# Dequantization helpers
# ---------------------------------------------------------------------------


def _as_row_major_packed(xq: torch.Tensor) -> torch.Tensor:
    """Return ``xq`` as a contiguous uint8 view with the packed axis last."""
    return xq.contiguous().view(torch.uint8)


def _dequantize_fp4_to_bf16(
    xq: torch.Tensor,
    scale: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    mxfp4_block_size: int = 32,
) -> torch.Tensor:
    """
    Dequantize packed FP4 tensor to BF16, dispatching MXFP4 vs NVFP4.

    Args:
        xq: Packed FP4 tensor (float4_e2m1fn_x2 / uint8).
        scale: Per-block scale factors (E8M0 for MXFP4, FP8 for NVFP4).
        global_scale: If present, use NVFP4 path; if None, use MXFP4 path.
        mxfp4_block_size: Block size for MXFP4 (ignored for NVFP4).

    Returns:
        BF16 tensor with unpacked shape.
    """
    # NVFP4 path: group_size is always 16
    if global_scale is not None:
        return dequantize_nvfp4(
            _as_row_major_packed(xq), scale, global_scale, group_size=16
        )
    # MXFP4 path
    else:
        return dequantize_mx4(
            _as_row_major_packed(xq), scale, group_size=mxfp4_block_size
        )


def _unblock_fp8_scale(
    scale: torch.Tensor,
    rows: int,
    num_groups: int,
) -> torch.Tensor:
    """Un-swizzle a padded+blocked CUTLASS FP8 scale buffer to ``(rows, num_groups)``.

    The ultra grouped op takes its scales in the same padded+swizzled 128x4
    layout the CUTLASS kernels consume, so a plain reshape would pair blocks
    with the wrong elements.
    """
    unblocked = _from_blocked(scale.reshape(-1).view(torch.uint8), (rows, num_groups))
    return unblocked.view(torch.float8_e4m3fn).to(torch.float32)


def _dequantize_fp4_ultra(
    xq: torch.Tensor,
    scale: torch.Tensor,
    global_scale_inv: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """
    Dequantize NVFP4 with per-token/per-group inverse global scale.

    Ultra grouped MM uses inverse global scales (1/global_scale) per token (X)
    or per group (W), rather than the standard combined global scale.

    Args:
        xq: Packed FP4 tensor in uint8, packed axis last.
        scale: Per-block FP8 scales in padded+swizzled CUTLASS layout, or an
            already-unswizzled ``(rows, num_groups)`` float32 tensor.
        global_scale_inv: Inverse global scale — scalar or [rows] per-token.
        group_size: Elements per scale group (default 16).

    Returns:
        BF16 tensor with unpacked shape.
    """
    xq_u8 = _as_row_major_packed(xq)
    M = xq_u8.shape[0]
    K = xq_u8.shape[-1] * 2
    num_groups = K // group_size

    x_float = fp4_to_float(xq_u8)

    if scale.dtype == torch.float32 and scale.shape == (M, num_groups):
        local_scale = scale
    else:
        local_scale = _unblock_fp8_scale(scale, M, num_groups)

    if global_scale_inv.dim() == 0:
        true_scale = local_scale * global_scale_inv.to(torch.float32)
    else:
        true_scale = local_scale * global_scale_inv.reshape(-1, 1).to(torch.float32)

    x_scaled = (
        x_float.view(M, num_groups, group_size) * true_scale.view(M, num_groups, 1)
    ).view(M, K)

    return x_scaled.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Autograd wrappers
# ---------------------------------------------------------------------------
#
# The underlying ``mslk::f4f4bf16*`` ops are non-functional (they accept
# optional in-place ``output`` / scale buffers), so
# ``torch.library.register_autograd`` cannot be used on them. Instead each op
# is wrapped in a ``torch.autograd.Function`` that saves the packed FP4 inputs
# and computes BF16 gradients by dequantizing them on the backward pass.


class _F4F4BF16(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        XQ,
        WQ,
        x_scale,
        w_scale,
        output=None,
        global_scale=None,
        mxfp4_block_size=32,
    ):
        ctx.save_for_backward(XQ, WQ, x_scale, w_scale)
        ctx.global_scale = global_scale
        ctx.mxfp4_block_size = mxfp4_block_size
        return torch.ops.mslk.f4f4bf16(
            XQ, WQ, x_scale, w_scale, output, global_scale, mxfp4_block_size
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        XQ, WQ, x_scale, w_scale = ctx.saved_tensors

        X_bf16 = _dequantize_fp4_to_bf16(
            XQ, x_scale, ctx.global_scale, ctx.mxfp4_block_size
        )
        W_bf16 = _dequantize_fp4_to_bf16(
            WQ, w_scale, ctx.global_scale, ctx.mxfp4_block_size
        )

        grad_output = grad_output.contiguous()

        # dX = dY @ W  — (M,N) @ (N,K) -> (M,K)
        grad_X = grad_output @ W_bf16
        # dW = dY^T @ X — (N,M) @ (M,K) -> (N,K)
        grad_W = grad_output.t() @ X_bf16

        # XQ, WQ, x_scale, w_scale, output, global_scale, mxfp4_block_size
        return (grad_X, grad_W, None, None, None, None, None)


def f4f4bf16(
    XQ, WQ, x_scale, w_scale, output=None, global_scale=None, mxfp4_block_size=32
):
    """Differentiable wrapper around ``mslk::f4f4bf16`` (BF16 backward)."""
    return _F4F4BF16.apply(
        XQ, WQ, x_scale, w_scale, output, global_scale, mxfp4_block_size
    )


def _group_bounds(offsets: torch.Tensor) -> list[tuple[int, int]]:
    """Convert cumulative ``offsets`` into explicit ``[start, end)`` pairs."""
    ends = [int(v) for v in offsets.cpu().tolist()]
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def _select_group(t: Optional[torch.Tensor], g: int, G: int) -> Optional[torch.Tensor]:
    """Index a per-group tensor if it is stacked over G, else pass it through."""
    if t is None:
        return None
    if t.dim() >= 1 and t.shape[0] == G:
        return t[g]
    return t


class _F4F4BF16GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, XQ, WQ, x_scale, w_scale, offsets, output=None, global_scale=None):
        ctx.save_for_backward(XQ, WQ, x_scale, w_scale, offsets)
        ctx.global_scale = global_scale
        return torch.ops.mslk.f4f4bf16_grouped_mm(
            XQ, WQ, x_scale, w_scale, offsets, output, global_scale
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        XQ, WQ, x_scale, w_scale, offsets = ctx.saved_tensors
        global_scale = ctx.global_scale

        if WQ.dim() != 3:
            # 2D-2D grouped GEMM partitions the K dimension and has no test
            # coverage or in-tree caller; refuse rather than return gradients
            # that silently disagree with the kernel's layout.
            raise NotImplementedError(
                "backward for 2D-2D f4f4bf16_grouped_mm (WQ of rank 2) is not "
                "implemented; only the 2D-3D (MoE) form is supported."
            )

        grad_output = grad_output.contiguous()
        block_size = 32
        bounds = _group_bounds(offsets)
        G = len(bounds)

        grad_X_parts = []
        grad_W_parts = []

        for g, (start, end) in enumerate(bounds):
            xs_g = _select_group(x_scale, g, G)
            ws_g = _select_group(w_scale, g, G)
            gs_g = _select_group(global_scale, g, G)

            # XQ is (total_M, K/2) row-major: rows are the group axis.
            X_g = _dequantize_fp4_to_bf16(XQ[start:end], xs_g, gs_g, block_size)
            # WQ is (G, K/2, N) column-major, so WQ[g].t() is the (N, K/2)
            # row-major operand the dequant helpers expect.
            W_g = _dequantize_fp4_to_bf16(WQ[g].t(), ws_g, gs_g, block_size)

            dY_g = grad_output[start:end]  # (M_g, N)

            # Y_g = X_g @ W_g^T with X_g (M_g, K) and W_g (N, K).
            grad_X_parts.append(dY_g @ W_g)  # (M_g, K)
            # Gradient must match the (K, N) layout of WQ[g].
            grad_W_parts.append(X_g.t() @ dY_g)  # (K, N)

        grad_X = torch.cat(grad_X_parts, dim=0)
        grad_W = torch.stack(grad_W_parts, dim=0)

        # XQ, WQ, x_scale, w_scale, offsets, output, global_scale
        return (grad_X, grad_W, None, None, None, None, None)


def f4f4bf16_grouped_mm(
    XQ, WQ, x_scale, w_scale, offsets, output=None, global_scale=None
):
    """Differentiable wrapper around ``mslk::f4f4bf16_grouped_mm``."""
    return _F4F4BF16GroupedMM.apply(
        XQ, WQ, x_scale, w_scale, offsets, output, global_scale
    )


class _F4F4BF16UltraGroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        XQ,
        WQ,
        x_scale,
        w_scale,
        offsets,
        x_global_scale,
        w_global_scale,
        output=None,
    ):
        ctx.save_for_backward(XQ, WQ, x_scale, w_scale, offsets)
        ctx.x_global_scale = x_global_scale
        ctx.w_global_scale = w_global_scale
        return torch.ops.mslk.f4f4bf16_ultra_grouped_mm(
            XQ, WQ, x_scale, w_scale, offsets, x_global_scale, w_global_scale, output
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        XQ, WQ, x_scale, w_scale, offsets = ctx.saved_tensors
        x_global_scale = ctx.x_global_scale
        w_global_scale = ctx.w_global_scale

        grad_output = grad_output.contiguous()
        group_size = 16

        total_M = XQ.shape[0]
        K = XQ.shape[-1] * 2
        num_groups = K // group_size

        # x_scale is one padded+swizzled buffer covering all total_M rows.
        # Un-swizzle once up front: row slicing is only meaningful afterwards,
        # because the blocked layout interleaves rows in 128-row tiles.
        x_scale_rows = _unblock_fp8_scale(x_scale, total_M, num_groups)

        bounds = _group_bounds(offsets)
        G = len(bounds)

        grad_X_parts = []
        grad_W_parts = []

        for g, (start, end) in enumerate(bounds):
            X_g = _dequantize_fp4_ultra(
                XQ[start:end],
                x_scale_rows[start:end],
                x_global_scale[start:end],
                group_size,
            )  # (M_g, K)

            # WQ is (G, K/2, N) column-major; transpose to (N, K/2).
            W_g = _dequantize_fp4_ultra(
                WQ[g].t(),
                _select_group(w_scale, g, G),
                w_global_scale[g],
                group_size,
            )  # (N, K)

            dY_g = grad_output[start:end]  # (M_g, N)

            # Y_g = X_g @ W_g^T with X_g (M_g, K) and W_g (N, K).
            grad_X_parts.append(dY_g @ W_g)  # (M_g, K)
            grad_W_parts.append(X_g.t() @ dY_g)  # (K, N)

        grad_X = torch.cat(grad_X_parts, dim=0)
        grad_W = torch.stack(grad_W_parts, dim=0)

        # XQ, WQ, x_scale, w_scale, offsets, x_global_scale, w_global_scale, output
        return (grad_X, grad_W, None, None, None, None, None, None)


def f4f4bf16_ultra_grouped_mm(
    XQ, WQ, x_scale, w_scale, offsets, x_global_scale, w_global_scale, output=None
):
    """Differentiable wrapper around ``mslk::f4f4bf16_ultra_grouped_mm``."""
    return _F4F4BF16UltraGroupedMM.apply(
        XQ, WQ, x_scale, w_scale, offsets, x_global_scale, w_global_scale, output
    )
