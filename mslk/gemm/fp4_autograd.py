# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Autograd support for FP4 GEMM custom ops.

Registers torch.library.register_autograd for:
  - mslk::f4f4bf16
  - mslk::f4f4bf16_grouped_mm
  - mslk::f4f4bf16_ultra_grouped_mm
  
Backward strategy: dequantize packed FP4 inputs to BF16, then compute
gradients via standard BF16 matmuls.  FP4 (E2M1, 3 magnitude levels)
cannot carry gradient signal, so this follows the same FP8-forward /
BF16-backward pattern used throughout the repo.
"""

from typing import Optional

import torch

from mslk.quantize.triton.legacy.fp4_utils import dequantize_nvfp4, fp4_to_float
from mslk.quantize.triton.legacy.primitives import _from_blocked

# ---------------------------------------------------------------------------
# Dequantization helpers
# ---------------------------------------------------------------------------


def _dequantize_mxfp4_to_bf16(
    xq: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Dequantize MXFP4 packed tensor to BF16.
    
    Args:
        xq: Packed FP4 tensor [M, K/2] in float4_e2m1fn_x2 / uint8.
        scale: Blocked E8M0 scale tensor (swizzled layout).
        block_size: Number of elements per scale group (default 32).
        
    Returns:
        BF16 tensor [M, K].
    """
    xq_u8 = xq.view(torch.uint8)
    M = xq_u8.shape[0]
    K = xq_u8.shape[1] * 2  # two nibbles per byte

    x_float = fp4_to_float(xq_u8)  # [M, K]

    # Unswizzle scale from blocked layout
    num_groups = K // block_size
    scale_flat = _from_blocked(scale.reshape(-1).view(torch.uint8), (M, num_groups))

    scale_float = torch.exp2(scale_flat.view(torch.uint8).to(torch.float32) - 127.0)

    x_scaled = (
        x_float.view(M, num_groups, block_size) * scale_float.view(M, num_groups, 1)
    ).view(M, K)

    return x_scaled.to(torch.bfloat16)


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
    if global_scale is not None:
        # NVFP4 path: group_size is always 16
        return dequantize_nvfp4(
            xq.view(torch.uint8), scale, global_scale, group_size=16
        )
    else:
        # MXFP4 path
        return _dequantize_mxfp4_to_bf16(xq, scale, block_size=mxfp4_block_size)


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
        xq: Packed FP4 tensor in uint8.
        scale: Per-block FP8 scales (swizzled/blocked layout).
        global_scale_inv: Inverse global scale — scalar or [M] per-token.
        group_size: Elements per scale group (default 16).
        
    Returns:
        BF16 tensor with unpacked shape.
    """
    xq_u8 = xq.view(torch.uint8)
    M = xq_u8.shape[0]
    K = xq_u8.shape[-1] * 2

    x_float = fp4_to_float(xq_u8)

    # Get FP8 local scales
    num_groups = K // group_size
    if scale.dim() >= 2:
        scale_fp8 = scale.reshape(M, -1)[:, :num_groups]
    else:
        scale_fp8 = scale.reshape(-1)

    local_scale = scale_fp8.view(torch.float8_e4m3fn).to(torch.float32)

    if global_scale_inv.dim() == 0:
        true_scale = local_scale * global_scale_inv.to(torch.float32)
    else:
        true_scale = local_scale * global_scale_inv.view(-1, 1).to(torch.float32)

    x_scaled = (
        x_float.view(M, num_groups, group_size) * true_scale.view(M, num_groups, 1)
    ).view(M, K)

    return x_scaled.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# mslk::f4f4bf16
# ---------------------------------------------------------------------------

if hasattr(torch.ops.mslk, "f4f4bf16"):

    def _f4f4bf16_setup_context(ctx, inputs, output):
        XQ, WQ, x_scale, w_scale, output_buf, global_scale, mxfp4_block_size = inputs
        ctx.save_for_backward(XQ, WQ, x_scale, w_scale)
        ctx.global_scale = global_scale
        ctx.mxfp4_block_size = mxfp4_block_size

    def _f4f4bf16_backward(ctx, grad_output):
        XQ, WQ, x_scale, w_scale = ctx.saved_tensors

        X_bf16 = _dequantize_fp4_to_bf16(
            XQ, x_scale, ctx.global_scale, ctx.mxfp4_block_size
        )
        W_bf16 = _dequantize_fp4_to_bf16(
            WQ, w_scale, ctx.global_scale, ctx.mxfp4_block_size
        )

        grad_output = grad_output.contiguous()

        # DGrad: dX = dY @ W  — (M,N) @ (N,K) -> (M,K)
        grad_X = grad_output @ W_bf16

        # WGrad: dW = dY^T @ X — (N,M) @ (M,K) -> (N,K)
        grad_W = grad_output.t() @ X_bf16

        # XQ, WQ, x_scale, w_scale, output, global_scale, mxfp4_block_size
        return (grad_X, grad_W, None, None, None, None, None)

    torch.library.register_autograd(
        "mslk::f4f4bf16",
        _f4f4bf16_backward,
        setup_context=_f4f4bf16_setup_context,
    )


if hasattr(torch.ops.mslk, "f4f4bf16_grouped_mm"):

    def _f4f4bf16_grouped_mm_setup_context(ctx, inputs, output):
        XQ, WQ, x_scale, w_scale, offsets, output_buf, global_scale = inputs
        ctx.save_for_backward(XQ, WQ, x_scale, w_scale, offsets)
        ctx.global_scale = global_scale

    def _f4f4bf16_grouped_mm_backward(ctx, grad_output):
        XQ, WQ, x_scale, w_scale, offsets = ctx.saved_tensors
        global_scale = ctx.global_scale

        grad_output = grad_output.contiguous()

        offsets_list = offsets.cpu().tolist()
        is_w_3d = WQ.dim() == 3
        G = len(offsets_list)
        starts = [0] + offsets_list[:-1]
        ends = offsets_list

        grad_X_parts = []
        grad_W_parts = []

        for g in range(G):
            start, end = int(starts[g]), int(ends[g])

            if is_w_3d:
                xq_g = XQ[start:end]
                wq_g = WQ[g]

                xs_g = (
                    x_scale[g]
                    if x_scale.dim() >= 2 and x_scale.shape[0] == G
                    else x_scale
                )
                ws_g = (
                    w_scale[g]
                    if w_scale.dim() >= 2 and w_scale.shape[0] == G
                    else w_scale
                )

                gs_g = None
                if global_scale is not None:
                    gs_g = (
                        global_scale[g]
                        if global_scale.dim() >= 1 and global_scale.shape[0] == G
                        else global_scale
                    )

                X_g = _dequantize_fp4_to_bf16(xq_g, xs_g, gs_g, 32)
                W_g = _dequantize_fp4_to_bf16(wq_g, ws_g, gs_g, 32)

                dY_g = grad_output[start:end]

                grad_X_parts.append(dY_g @ W_g.t())
                grad_W_parts.append(dY_g.t() @ X_g)
            else:
                # 2D-2D grouped GEMM: XQ is (M, total_K), WQ is (total_K, N),
                # offsets index into K dimension, output is (G, M, N).
                xq_g = XQ[:, start:end] if start < XQ.shape[1] else XQ
                wq_g = WQ[:, start:end] if start < WQ.shape[1] else WQ

                xs_g = (
                    x_scale[g]
                    if x_scale.dim() >= 2 and x_scale.shape[0] == G
                    else x_scale
                )
                ws_g = (
                    w_scale[g]
                    if w_scale.dim() >= 2 and w_scale.shape[0] == G
                    else w_scale
                )

                gs_g = None
                if global_scale is not None:
                    gs_g = (
                        global_scale[g]
                        if global_scale.dim() >= 1 and global_scale.shape[0] == G
                        else global_scale
                    )

                X_g = _dequantize_fp4_to_bf16(xq_g, xs_g, gs_g, 32)
                W_g = _dequantize_fp4_to_bf16(wq_g, ws_g, gs_g, 32)

                # Output is (G, M, N) — index by group to get (M, N) slice.
                dY_g = grad_output[g]

                grad_X_parts.append(dY_g @ W_g)
                grad_W_parts.append(dY_g.t() @ X_g)

        if is_w_3d:
            grad_X = torch.cat(grad_X_parts, dim=0)
            grad_W = torch.stack(grad_W_parts, dim=0)
        else:
            grad_X = torch.cat(grad_X_parts, dim=1)
            grad_W = torch.cat(grad_W_parts, dim=1)

        # 7 inputs: XQ, WQ, x_scale, w_scale, offsets, output, global_scale
        return (grad_X, grad_W, None, None, None, None, None)

    torch.library.register_autograd(
        "mslk::f4f4bf16_grouped_mm",
        _f4f4bf16_grouped_mm_backward,
        setup_context=_f4f4bf16_grouped_mm_setup_context,
    )


if hasattr(torch.ops.mslk, "f4f4bf16_ultra_grouped_mm"):

    def _f4f4bf16_ultra_setup_context(ctx, inputs, output):
        (
            XQ,
            WQ,
            x_scale,
            w_scale,
            offsets,
            x_global_scale,
            w_global_scale,
            output_buf,
        ) = inputs
        ctx.save_for_backward(XQ, WQ, x_scale, w_scale, offsets)
        ctx.x_global_scale = x_global_scale
        ctx.w_global_scale = w_global_scale

    def _f4f4bf16_ultra_backward(ctx, grad_output):
        XQ, WQ, x_scale, w_scale, offsets = ctx.saved_tensors
        x_global_scale = ctx.x_global_scale
        w_global_scale = ctx.w_global_scale

        grad_output = grad_output.contiguous()

        offsets_list = offsets.cpu().tolist()
        G = len(offsets_list)
        starts = [0] + offsets_list[:-1]
        ends = offsets_list

        grad_X_parts = []
        grad_W_parts = []

        for g in range(G):
            start, end = int(starts[g]), int(ends[g])

            xq_g = XQ[start:end]
            wq_g = WQ[g] if WQ.dim() == 3 else WQ

            xs_g = x_scale[start:end] if x_scale.dim() == 2 else x_scale
            ws_g = (
                w_scale[g] if w_scale.dim() >= 2 and w_scale.shape[0] == G else w_scale
            )

            if x_global_scale is not None and x_global_scale.numel() > 0:
                x_gs_g = x_global_scale[start:end]
                X_g = _dequantize_fp4_ultra(xq_g, xs_g, x_gs_g)
            else:
                X_g = _dequantize_fp4_to_bf16(xq_g, xs_g, None, 32)

            if w_global_scale is not None and w_global_scale.numel() > 0:
                w_gs_g = w_global_scale[g]
                W_g = _dequantize_fp4_ultra(wq_g, ws_g, w_gs_g)
            else:
                W_g = _dequantize_fp4_to_bf16(wq_g, ws_g, None, 32)

            dY_g = grad_output[start:end]

            # Forward computes Y_g = X_g @ W_g where W_g is stored as (K, N)
            # Therefore: grad_X = dY @ W_g^T, grad_W = dY^T @ X_g.
            grad_X_parts.append(dY_g @ W_g.t())
            grad_W_parts.append(dY_g.t() @ X_g)

        grad_X = torch.cat(grad_X_parts, dim=0)
        grad_W = (
            torch.stack(grad_W_parts, dim=0)
            if WQ.dim() == 3
            else torch.cat(grad_W_parts, dim=0)
        )

        # XQ, WQ, x_scale, w_scale, offsets, x_global_scale, w_global_scale, output
        return (grad_X, grad_W, None, None, None, None, None, None)

    torch.library.register_autograd(
        "mslk::f4f4bf16_ultra_grouped_mm",
        _f4f4bf16_ultra_backward,
        setup_context=_f4f4bf16_ultra_setup_context,
    )
