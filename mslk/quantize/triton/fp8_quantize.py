# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import logging
import os
from typing import Dict, Optional, Tuple

import torch
import triton  # @manual

import triton.language as tl  # @manual
from mslk.utils.triton.fp8_utils import get_fp8_constants
from triton import Config  # @manual

logger: logging.Logger = logging.getLogger(__name__)

running_on_github: bool = os.getenv("GITHUB_ENV") is not None

try:
    # pyre-ignore[21]
    from triton.fb.compat import disable_bufferops  # @manual
except ModuleNotFoundError:
    # Ensure we can call disable_bufferops if compat is not included (e.g. opensource)
    # TODO(njriasan): Remove when we integrate triton.fb.compat into every Triton
    # version.
    from contextlib import contextmanager

    @contextmanager
    def disable_bufferops(_unused: bool):
        yield None


@triton.autotune(
    configs=[
        Config({"BLOCK_SIZE": 512}),
        Config({"BLOCK_SIZE": 1024}),
        Config({"BLOCK_SIZE": 2048}),
        Config({"BLOCK_SIZE": 4096}),
        Config({"BLOCK_SIZE": 8192}),
    ],
    key=["K"],
)
@triton.jit
def _kernel_quantize_fp8_row(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    zero_start_index_M,
    B,
    M,
    N,
    K,
    K_fp8,  # used when padding
    stride_ab,
    stride_am,
    stride_an,
    stride_ak,
    stride_ob,
    stride_om,
    stride_on,
    stride_ok,
    stride_zb,
    stride_zm,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    JAGGED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Quantize and scale each row.

    Scale per row i is computed as MAX_FP8 / max(abs(A[i, :]))

    Kernel naively iterates through  matrix with [1, BLOCK_SIZE] tiles
    in a max pass then scale/quantize pass.

    Todo:
        * Better tiling schemes.

    Args:
        A (Tensor): higher precision input tensor of 4 dimension.
        A_scale (Tensor): [B * M * N] reciprocal scale tensor per row.
        A_fp8 (Tensor): fp8 scaled tensor. A_fp8 = A / a_scale
        scale_ub (Tensor): [1] Maximum value allowed for scale.
        B (int): Size of dimenion 0
        M (int): Size of dimenion 1
        N (int): Size of dimenion 2
        K (int): Size of dimenion 3 (input row size)
        K_fp8 (int): Size of dimenion 3 for A_fp8 (output row size, can be >= K)
        stride_ab (int): Stride of b dimension of A.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_ob (int): Stride of b dimension of output.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_zb (int): Stride of b dimension of jagged index.
        stride_zm (int): Stride of m dimension of jagged index.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whethar to apply scale_ub.
        JAGGED (bool): Whether to use jagged indexing.
        BLOCK_SIZE (int): Block size for reduction.
        USE_INT64 (bool): Whether to use int64 indexing for large inputs.
    """
    pid = tl.program_id(0)
    # Use int64 indexing for large inputs. This is slower, but
    # needed to avoid index overflows.
    if USE_INT64:
        pid = pid.to(tl.int64)
    n_offset = tl.arange(0, BLOCK_SIZE)
    a_offset_base = (
        pid // (M * N) * stride_ab
        + (pid % (M * N)) // N * stride_am
        + (pid % (M * N)) % N * stride_an
    )
    a_fp8_offset_base = (
        pid // (M * N) * stride_ob
        + (pid % (M * N)) // N * stride_om
        + (pid % (M * N)) % N * stride_on
    )

    K_in = K

    if JAGGED:
        z_offset_base = pid // (M * N) * stride_zb + (pid % (M * N)) // N * stride_zm
        group_rows = tl.load(zero_start_index_M + z_offset_base)
        current_row = pid % N
        # If this row is empty, dont process any of it.
        if current_row >= group_rows:
            K_in = 0

    # Calculate max.
    cur_max = 0.0
    for _k in range(0, tl.cdiv(K_in, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        tile_max = tl.max(tl.abs(a))
        cur_max = tl.maximum(tile_max, cur_max)
        n_offset += BLOCK_SIZE

    # Clamp max value appropriately.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        cur_max = tl.clamp(cur_max, EPS, ub)
    else:
        cur_max = tl.maximum(cur_max, EPS)
    # Scale and quantize.
    a_scale = MAX_FP8 / cur_max
    tl.store(A_scale + pid, 1.0 / a_scale)
    n_offset = tl.arange(0, BLOCK_SIZE)

    # Write quantized values for the first K elements (from A), and pad the rest with zeros up to K_fp8
    for _k in range(0, tl.cdiv(K_fp8, BLOCK_SIZE)):
        # Load from A if in range, else 0 (we're going all the way to K_fp8)
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        # For elements >= K, a will be 0
        a_fp8 = a * a_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)

        # Store the full new row in its place (for elements >= K, a_fp8 is already 0)
        tl.store(
            A_fp8 + a_fp8_offset_base + n_offset * stride_ok,
            a_fp8,
            mask=n_offset < K_fp8,
        )
        n_offset += BLOCK_SIZE


def triton_quantize_fp8_row(
    a: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    zero_start_index_M: Optional[torch.Tensor] = None,
    align_rows_to: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call the triton quantize fp8 row kernel to quantize a tensor to fp8 with row-wise scalings.

    Args:
        a (Tensor): higher precision input tensor of 4 dimension.
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        align_rows_to: Pad rows to align to this value. Useful for downstream kernels accepting specific sizes (e.g., multiple of 16)

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: reciprocal scale tensor per row.
    """
    if scale_ub is not None and scale_ub.device != a.device:
        raise Exception("'scale_ub' must be on the same device as 'a'")
    if zero_start_index_M is not None and zero_start_index_M.device != a.device:
        raise Exception("'zero_start_index_M' must be on the same device as 'a'")

    assert a.dim() <= 4, "Triton only supports up to 4 dimension input tensor."
    a_shape = a.shape
    while a.dim() < 4:
        a = a.unsqueeze(0)
    if zero_start_index_M is not None:
        # There should be one value of zero_start_index_M per NxK matrix.
        zero_start_index_M = zero_start_index_M.view(a.shape[0], a.shape[1])
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    num_rows = a.numel() // a.shape[-1]
    a_scale = torch.empty((num_rows), dtype=torch.float32, device=a.device)
    # If align_rows_to is provided, pad the last dimension to be a multiple of it
    if align_rows_to is not None:
        last_dim = a.shape[-1]
        padded_last_dim = (
            (last_dim + align_rows_to - 1) // align_rows_to
        ) * align_rows_to
        a_fp8 = torch.empty(
            (*a.shape[:-1], padded_last_dim), device=a.device, dtype=pt_dtype
        )
        a_shape = torch.Size((*a_shape[:-1], padded_last_dim))
    else:
        a_fp8 = torch.empty(a.shape, device=a.device, dtype=pt_dtype)

    # If input tensor is sufficiently large, we need to use int64 indexing.
    use_int64 = a.numel() > (2**31 - 1)
    grid = (num_rows,)
    # Pick a conservative value for inference shapes for disabling BufferOps.
    should_disable_bufferops = torch.version.hip is not None and a_shape[0] < 32
    with disable_bufferops(should_disable_bufferops):
        with torch.cuda.device(a.device.index):
            _kernel_quantize_fp8_row[grid](
                a,
                a_scale,
                a_fp8,
                scale_ub,
                zero_start_index_M,
                a.shape[0],
                a.shape[1],
                a.shape[2],
                a.shape[3],
                a_fp8.shape[3],
                a.stride(0),
                a.stride(1),
                a.stride(2),
                a.stride(3),
                a_fp8.stride(0),
                a_fp8.stride(1),
                a_fp8.stride(2),
                a_fp8.stride(3),
                (
                    zero_start_index_M.stride(0)
                    if zero_start_index_M is not None
                    else None
                ),
                (
                    zero_start_index_M.stride(1)
                    if zero_start_index_M is not None
                    else None
                ),
                TL_FP8_DTYPE=tl_dtype,
                MAX_FP8=max_fp8,
                EPS=eps,
                CLAMP_MAX=scale_ub is not None,
                JAGGED=zero_start_index_M is not None,
                USE_INT64=use_int64,
            )

    return a_fp8.view(a_shape), a_scale.view(a_shape[:-1])


@triton.autotune(
    configs=[
        Config({"BLOCK_SIZE": 512}),
        Config({"BLOCK_SIZE": 1024}),
        Config({"BLOCK_SIZE": 2048}),
        Config({"BLOCK_SIZE": 4096}),
        Config({"BLOCK_SIZE": 8192}),
    ],
    key=["K"],
)
@triton.jit
def _kernel_quantize_fp8_packed_row(
    A,
    A_fp8,
    packed_scale,
    scale_ub,
    zero_start_index_M,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_an,
    stride_ak,
    stride_ob,
    stride_om,
    stride_on,
    stride_ok,
    packed_scale_stride,
    stride_zb,
    stride_zm,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    JAGGED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Quantize and scale each row.

    Scale per row i is computed as MAX_FP8 / max(abs(A[i, :]))

    Kernel naively iterates through  matrix with [1, BLOCK_SIZE] tiles
    in a max pass then scale/quantize pass.

    Todo:
        * Better tiling schemes.

    Args:
        A (Tensor): higher precision input tensor of 4 dimension.
        packed_scale (Tensor): [B * M * N] reciprocal scale tensor per row.
        A_fp8 (Tensor): fp8 scaled tensor. A_fp8 = A / a_scale
        scale_ub (Tensor): [1] Maximum value allowed for scale.
        B (int): Size of dimenion 0
        M (int): Size of dimenion 1
        N (int): Size of dimenion 2
        K (int): Size of dimenion 3
        stride_ab (int): Stride of b dimension of A.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_ob (int): Stride of b dimension of output.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        stride_ok (int): Stride of k dimension of output.
        packed_scale_stride (int): Stride of the packed scale, indexing into a_fp8.
        stride_zb (int): Stride of b dimension of jagged index.
        stride_zm (int): Stride of m dimension of jagged index.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whethar to apply scale_ub.
        JAGGED (bool): Whether to use jagged indexing.
        BLOCK_SIZE (int): Block size for reduction.
        USE_INT64 (bool): Whether to use int64 indexing for large inputs.
    """
    pid = tl.program_id(0)
    # Use int64 indexing for large inputs. This is slower, but
    # needed to avoid index overflows.
    if USE_INT64:
        pid = pid.to(tl.int64)
    n_offset = tl.arange(0, BLOCK_SIZE)
    a_offset_base = (
        pid // (M * N) * stride_ab
        + (pid % (M * N)) // N * stride_am
        + (pid % (M * N)) % N * stride_an
    )
    a_fp8_offset_base = (
        pid // (M * N) * stride_ob
        + (pid % (M * N)) // N * stride_om
        + (pid % (M * N)) % N * stride_on
    )

    K_in = K

    if JAGGED:
        z_offset_base = pid // (M * N) * stride_zb + (pid % (M * N)) // N * stride_zm
        group_rows = tl.load(zero_start_index_M + z_offset_base)
        current_row = pid % N
        # If this row is empty, dont process any of it.
        if current_row >= group_rows:
            K_in = 0

    # Calculate max.
    cur_max = 0.0
    for _k in range(0, tl.cdiv(K_in, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        tile_max = tl.max(tl.abs(a))
        cur_max = tl.maximum(tile_max, cur_max)
        n_offset += BLOCK_SIZE

    # Clamp max value appropriately.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        cur_max = tl.clamp(cur_max, EPS, ub)
    else:
        cur_max = tl.maximum(cur_max, EPS)
    # Scale and quantize.
    a_scale = MAX_FP8 / cur_max

    (fp8_0, fp8_1, fp8_2, fp8_3) = tl.inline_asm_elementwise(
        asm="""
        {
            // $4 is the input register
            .reg .b32 input;
            mov.b32 input, $4;
            mov.b32 $0, $4;
            shr.b32 $1, $4, 8;
            shr.b32 $2, $4, 16;
            shr.b32 $3, $4, 24;
        }
            """,
        constraints=("=r,=r,=r,=r," "r"),
        # Let's pass in 1 uint32 value per iteration, containing 8 packed int4 values
        args=[1.0 / a_scale],
        dtype=(
            tl.uint8,
            tl.uint8,
            tl.uint8,
            tl.uint8,
        ),
        is_pure=True,
        pack=1,
    )

    # There are some compiler issues with FP8 pointers
    packed_scale_ptr = packed_scale.to(tl.pointer_type(tl.uint8))
    tl.store(packed_scale_ptr + pid * packed_scale_stride, fp8_0)
    tl.store(packed_scale_ptr + pid * packed_scale_stride + 1, fp8_1)
    tl.store(packed_scale_ptr + pid * packed_scale_stride + 2, fp8_2)
    tl.store(packed_scale_ptr + pid * packed_scale_stride + 3, fp8_3)

    n_offset = tl.arange(0, BLOCK_SIZE)

    for _k in range(0, tl.cdiv(K, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        a_fp8 = a * a_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        tl.store(
            A_fp8 + a_fp8_offset_base + n_offset * stride_ok,
            a_fp8,
            mask=n_offset < K,
        )

        n_offset += BLOCK_SIZE


def triton_quantize_fp8_packed_row(
    a: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    zero_start_index_M: Optional[torch.Tensor] = None,
    return_only_packed: Optional[bool] = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Call the triton quantize fp8 row kernel to quantize a tensor to fp8 with row-wise scalings.

    This packs the FP32 scale at the end of each row, so the fp8 scaled tensor and the reciprocal scale tensor per row are contiguous in memory.

    Args:
        a (Tensor): higher precision input tensor of 4 dimension.
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        return_only_packed (bool): Only return the packed tensor, do not unpack results if True
    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: reciprocal scale tensor per row.
        torch.Tensor: The packed FP8 scaled tensor, with the scale at the end of each row.
    """
    if scale_ub is not None and scale_ub.device != a.device:
        raise Exception("'scale_ub' must be on the same device as 'a'")
    if zero_start_index_M is not None and zero_start_index_M.device != a.device:
        raise Exception("'zero_start_index_M' must be on the same device as 'a'")

    assert a.dim() <= 4, "Triton only supports up to 4 dimension input tensor."
    a_shape = a.shape
    while a.dim() < 4:
        a = a.unsqueeze(0)
    if zero_start_index_M is not None:
        # There should be one value of zero_start_index_M per NxK matrix.
        zero_start_index_M = zero_start_index_M.view(a.shape[0], a.shape[1])
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    num_rows = a.numel() // a.shape[-1]

    # Allocate an extra 4-bytes at the end of each row for the scale.
    a_fp8 = torch.empty(
        (*a.shape[:-1], a.shape[-1] + 4), device=a.device, dtype=pt_dtype
    )

    # create a view of the packed scale
    packed_scale = a_fp8[..., -4:]

    # If input tensor is sufficiently large, we need to use int64 indexing.
    use_int64 = a.numel() > (2**31 - 1)
    grid = (num_rows,)

    with torch.cuda.device(a.device.index):
        _kernel_quantize_fp8_packed_row[grid](
            a,
            a_fp8,
            packed_scale,
            scale_ub,
            zero_start_index_M,
            a.shape[0],
            a.shape[1],
            a.shape[2],
            a.shape[3],
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a.stride(3),
            a_fp8.stride(0),
            a_fp8.stride(1),
            a_fp8.stride(2),
            a_fp8.stride(3),
            packed_scale.stride(2),  # this is the stride that matters
            zero_start_index_M.stride(0) if zero_start_index_M is not None else None,
            zero_start_index_M.stride(1) if zero_start_index_M is not None else None,
            TL_FP8_DTYPE=tl_dtype,
            MAX_FP8=max_fp8,
            EPS=eps,
            CLAMP_MAX=scale_ub is not None,
            JAGGED=zero_start_index_M is not None,
            USE_INT64=use_int64,
        )
    if return_only_packed:
        return None, None, a_fp8.view((*a_shape[:-1], a_shape[-1] + 4))

    # Extract the original shape data without the extra 4 bytes per row
    # The data is still contiguous in memory, so we have to unpack it.
    final_fp8_view = a_fp8[..., :-4].view(a_shape)
    scale_view = a_fp8[..., -4:].reshape((num_rows * 4)).view(torch.float32)

    # the difference with the packed API is that it also
    # returns the full packed tensor as a third return value
    return final_fp8_view, scale_view.view(a_shape[:-1]), a_fp8


@torch.library.custom_op("triton::quantize_fp8_packed_row", mutates_args=())
def quantize_fp8_packed_row(
    a: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    zero_start_index_M: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a to fp8 with row-wise scalings and optionally move to output device.

    Args:
        a (Tensor): Input high precision tensor. Required to have no more than 4 dimension
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.
    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: The reciprocal scale tensor per row.
    """

    if a.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        # ignore the packed tensor here, we aren't testing it
        a_fp8, scale, _ = triton_quantize_fp8_packed_row(
            a, scale_ub, zero_start_index_M, return_only_packed=False
        )
        assert a_fp8 is not None
        assert scale is not None
        return a_fp8, scale
    # else use pytorch implementation.
    if not output_device:
        output_device = a.device

    a_shape = a.shape
    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()
    row_max: torch.Tensor = torch.max(torch.abs(a), dim=-1)[0]
    # Apply clamping.
    if scale_ub is not None:
        row_max = torch.clamp(row_max, min=eps, max=scale_ub.item())
    else:
        # pyre-ignore[6]: Incompatible parameter type [6]
        row_max = torch.clamp(row_max, min=eps)
    a_scale = torch.empty((a.shape[:-1]), dtype=torch.float32, device=output_device)
    a_scale = max_fp8 / row_max.to(torch.float32)  # pyre-ignore
    a_scale[a_scale == float("inf")] = 1.0  # pyre-ignore
    a_fp8 = a * a_scale[..., None]  # pyre-ignore
    # Cast and move data to output device (for cpu weight loading).
    a_fp8 = a_fp8.to(device=output_device, dtype=pt_dtype)
    a_scale = a_scale.to(output_device)  # pyre-ignore
    del a
    return a_fp8, (1 / a_scale).view(a_shape[:-1])  # pyre-ignore


@torch.library.custom_op("triton::quantize_fp8_packed_row_raw", mutates_args=())
def quantize_fp8_packed_row_raw(
    a: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    zero_start_index_M: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Quantize a to fp8 with row-wise scalings and optionally move to output device.

    Identical to quantize_fp8_packed_row, except it only returns the raw packed tensor.

    Args:
        a (Tensor): Input high precision tensor. Required to have no more than 4 dimension
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.
    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: The reciprocal scale tensor per row.
    """

    if a.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        # ignore the packed tensor here, we aren't testing it
        _, _, packed_tensor = triton_quantize_fp8_packed_row(
            a, scale_ub, zero_start_index_M, return_only_packed=True
        )
        return packed_tensor
    else:
        raise Exception(
            "No PyTorch implementation provided for triton::quantize_fp8_packed_row_raw"
        )


@torch.library.custom_op("triton::quantize_fp8_row", mutates_args=())
def quantize_fp8_row(
    a: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    zero_start_index_M: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
    align_rows_to: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a to fp8 with row-wise scalings and optionally move to output device.

    Args:
        a (Tensor): Input high precision tensor. Required to have no more than 4 dimension
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.
        align_rows_to: Pad rows to align to this value. Useful for downstream kernels accepting specific sizes (e.g., multiple of 16)

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: The reciprocal scale tensor per row.
    """

    if a.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        return triton_quantize_fp8_row(
            a,
            scale_ub,
            zero_start_index_M,
            align_rows_to=align_rows_to,
        )
    # else use pytorch implementation.
    if not output_device:
        output_device = a.device

    a_shape = a.shape
    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()
    row_max: torch.Tensor = torch.max(torch.abs(a), dim=-1)[0]
    # Apply clamping.
    if scale_ub is not None:
        row_max = torch.clamp(row_max, min=eps, max=scale_ub.item())
    else:
        # pyre-ignore[6]: Incompatible parameter type [6]
        row_max = torch.clamp(row_max, min=eps)
    a_scale = torch.empty((a.shape[:-1]), dtype=torch.float32, device=output_device)
    a_scale = max_fp8 / row_max.to(torch.float32)  # pyre-ignore
    a_scale[a_scale == float("inf")] = 1.0  # pyre-ignore
    a_fp8 = a * a_scale[..., None]  # pyre-ignore
    # Cast and move data to output device (for cpu weight loading).
    a_fp8 = a_fp8.to(device=output_device, dtype=pt_dtype)
    a_scale = a_scale.to(output_device)  # pyre-ignore
    del a
    return a_fp8, (1 / a_scale).view(a_shape[:-1])  # pyre-ignore


@quantize_fp8_row.register_fake
def quantize_fp8_row_meta(
    a: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    zero_start_index_M: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
    align_rows_to: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape function for torch compile."""
    if output_device is None:
        output_device = a.device
    a_shape = a.shape
    dtype = get_fp8_constants()[0]
    fake_scale = torch.empty(a_shape[:-1], device=output_device, dtype=torch.float32)
    if align_rows_to is not None:
        last_dim = a.shape[-1]
        padded_last_dim = (
            (last_dim + align_rows_to - 1) // align_rows_to
        ) * align_rows_to
        fake_out = torch.empty(
            (*a.shape[:-1], padded_last_dim), device=output_device, dtype=dtype
        )
        return fake_out, fake_scale
    else:
        fake_out = torch.empty(a.shape, device=output_device, dtype=dtype)
        return fake_out, fake_scale


@triton.autotune(
    configs=[
        Config({"BLOCK_SIZE": 512}),
        Config({"BLOCK_SIZE": 1024}),
        Config({"BLOCK_SIZE": 2048}),
        Config({"BLOCK_SIZE": 4096}),
        Config({"BLOCK_SIZE": 8192}),
    ],
    key=["N"],
)
@triton.jit
def _kernel_scale_fp8_row(
    A,
    x_scale,
    w_scale,
    scaled_out,
    M,
    N,
    stride_am,
    stride_an,
    stride_om,
    stride_on,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Scale each row of A by x_scale and each column of A by w_scale.

    Args:
        A (Tensor): [m, n] Input tensor to scale.
        x_scale (Tensor): [m] Row-wise scale tensor.
        w_scale (Tensor): [n] Col-wise scale tensor.
        scaled_out (Tensor): [m, n] Output tensor.
        M (int): Number of rows.
        N (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        BLOCK_SIZE (int): Block size for data loads.
    """
    pid = tl.program_id(0)
    n_offset = tl.arange(0, BLOCK_SIZE)
    # Load activation scale for this row.
    row_scale = tl.load(x_scale + pid)

    # Iterate over chunks of the row and apply scales.
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(
            A + pid * stride_am + n_offset * stride_an, mask=n_offset < N, other=0.0
        )
        col_scale = tl.load(w_scale + n_offset)
        scaled_a = a * row_scale * col_scale
        tl.store(
            scaled_out + pid * stride_om + n_offset * stride_on,
            scaled_a,
            mask=n_offset < N,
        )
        n_offset += BLOCK_SIZE


def scale_fp8_row(
    a: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Apply only rowwise scaling to a tensor. Useful when combining with kernels
    that do not support fused rowwise scaling.

    Args:
        a (Tensor): Input floating point tensor to be scaled.
        x_scale (Tensor): Row-wise activation scale tensor.
        w_scale (Tensor): Col-wise weight scale tensor.
    """
    if a.device == torch.device("cpu"):
        # On CPU we'll just use native pytorch to scale.
        return a * x_scale[:, None] * w_scale[None, :]

    if x_scale.device != a.device:
        raise Exception("'x_scale' must be on the same device as 'a'")
    if w_scale.device != a.device:
        raise Exception("'w_scale' must be on the same device as 'a'")

    # Otherwise, use a fast triton kernel to implement.
    # We'll parallelize over rows.
    num_rows = a.shape[0]
    scaled_out = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    grid = (num_rows,)
    with torch.cuda.device(a.device.index):
        _kernel_scale_fp8_row[grid](
            a,
            x_scale,
            w_scale,
            scaled_out,
            a.shape[0],
            a.shape[1],
            a.stride(0),
            a.stride(1),
            scaled_out.stride(0),
            scaled_out.stride(1),
        )

    return scaled_out


@triton.jit
def _kernel_quantize_fp8_block(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    M,
    K,
    stride_am,
    stride_ak,
    stride_om,
    stride_ok,
    stride_a_scale_m,
    stride_a_scale_k,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_MAJOR: tl.constexpr,
) -> None:
    """Quantize and scale each [BLOCK_M, BLOCK_K] block.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(A[i:i+BLOCK_M, j:j+BLOCK_K])))

    Kernel naively iterates through  matrix with [BLOCK_M, BLOCK_K] tiles.

    Todo:
        * Better tiling and ordering schemes.

    Args:
        A (Tensor): [M, K] higher precision input tensor.
        A_scale (Tensor): [cdiv(M, BLOCK_M), cdiv(K, BLOCK_K)] reciprocal scale tensor per block.
        A_fp8 (Tensor): [M, K] fp8 scaled tensor. A_fp8 = A * a_scale
        scale_ub (Tensor): [1] Maximum allowed value for scale.
        M (int): Number of rows.
        K (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_a_scale_m (int): Stride of m dimension of A_scale.
        stride_a_scale_k (int): Stride of k dimension of A_scale.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whether to apply scale_ub.
        BLOCK_M (int): Block size for M dimension of A_scale and kernel.
        BLOCK_K (int): Block size for K dimension of A_scale and kernel.
        K_MAJOR (bool): Whether output scales should be K major (True) or MN major (False).
    """
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    block_m = pid // grid_k
    block_k = pid % grid_k
    rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_offset = rm[:, None] * stride_am + rk[None, :] * stride_ak
    out_offset = rm[:, None] * stride_om + rk[None, :] * stride_ok
    a_mask = (rm < M)[:, None] & (rk < K)[None, :]
    a_block = tl.load(A + a_offset, mask=a_mask, other=0.0)

    block_max = tl.max(tl.abs(a_block))
    # Apply appropriate clamping.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        block_max = tl.clamp(block_max, EPS, ub)
    else:
        block_max = tl.maximum(block_max, EPS)
    scale = MAX_FP8 / block_max

    # Write in transposed order if specified.
    if K_MAJOR:
        scale_offset = block_m * stride_a_scale_m + block_k * stride_a_scale_k
    else:
        scale_offset = block_k * stride_a_scale_m + block_m * stride_a_scale_k
    tl.store(A_scale + scale_offset, 1.0 / scale)
    a_fp8 = a_block * scale
    # Clamp A to fp8 range to make sure there's no overflow.
    # This is required for AMD. Nvidia's default saturation
    # handles it, but it's nice to have anyway.
    a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8)
    a_fp8.to(TL_FP8_DTYPE)
    tl.store(A_fp8 + out_offset, a_fp8, mask=a_mask)


def triton_quantize_fp8_block(
    x: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
    scale_ub: Optional[torch.Tensor] = None,
    k_major: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k])))

    Args:
        x (torch.Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.
        scale_ub: Maximum allowed value for scale.
        k_major (bool): Whether output scales should be K major (True) or MN major (False).

    Returns:
        torch.Tensor : [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block
        if k_major is True, otherwise [cdiv(K, block_k), cdiv(M, block_M)].
    """
    assert (
        x.device != torch.device("cpu")
    ), "Blockwise quantization not support on cpu, please use row-wise quantization instead."

    if scale_ub is not None and scale_ub.device != x.device:
        raise Exception("'scale_ub' must be on the same device as 'a'")

    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    M, K = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_k = triton.cdiv(K, block_k)
    if k_major:
        x_scale = torch.empty((grid_m, grid_k), device=x.device, dtype=torch.float32)
    else:
        x_scale = torch.empty((grid_k, grid_m), device=x.device, dtype=torch.float32)
    x_fp8 = torch.empty((M, K), device=x.device, dtype=pt_dtype)

    _kernel_quantize_fp8_block[(grid_m * grid_k,)](
        x,
        x_scale,
        x_fp8,
        scale_ub,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        # pyre-ignore[6]: Incompatible parameter type [6]
        TL_FP8_DTYPE=tl_dtype,
        # pyre-ignore[6]: Incompatible parameter type [6]
        MAX_FP8=max_fp8,
        # pyre-ignore[6]: Incompatible parameter type [6]
        EPS=eps,
        # pyre-ignore[6]: Incompatible parameter type [6]
        CLAMP_MAX=scale_ub is not None,
        # pyre-ignore[6]: Incompatible parameter type [6]
        BLOCK_M=block_m,
        # pyre-ignore[6]: Incompatible parameter type [6]
        BLOCK_K=block_k,
        # pyre-ignore[6]: Incompatible parameter type [6]
        K_MAJOR=k_major,
    )

    return x_fp8.view(x_shape), x_scale


@torch.library.custom_op("triton::quantize_fp8_block", mutates_args=())
def quantize_fp8_block(
    x: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
    scale_ub: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
    k_major: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings and optionally move to output device.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k])))

    Args:
        x (Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.
        scale_ub: Maximum allowed value for scale.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.
        k_major (bool): Whether output scales should be K major (True) or MN major (False).

    Returns:
        torch.Tensor: [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block
        if k_major is True, otherwise [cdiv(K, block_k), cdiv(M, block_M)].
    """
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    if x.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        xq, x_scale = triton_quantize_fp8_block(x, block_m, block_k, scale_ub, k_major)
        return xq.view(x_shape), x_scale
    # else use pytorch implementation.
    if not output_device:
        output_device = x.device

    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()

    M, K = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_k = triton.cdiv(K, block_k)

    # Pad x to multiple of block size.
    padded_m = grid_m * block_m
    padded_k = grid_k * block_k
    x_padded = torch.zeros(padded_m, padded_k, dtype=x.dtype, device=x.device)
    x_padded[:M, :K] = x

    # Blockwise max.
    block_max = (
        x_padded.abs().reshape(grid_m, block_m, grid_k, block_k).amax(dim=(1, 3))
    )

    # Apply clamping.
    if scale_ub is not None:
        block_max = torch.clamp(block_max, min=eps, max=scale_ub.item())
    else:
        block_max = torch.clamp(block_max, min=eps)
    x_scale = torch.empty((grid_m, grid_k), dtype=torch.float32, device=output_device)
    x_scale = max_fp8 / block_max.to(torch.float32)  # pyre-ignore
    # pyre-ignore[16]: Undefined attribute [16]
    x_scale[x_scale == float("inf")] = 1.0
    x_fp8 = (
        x_padded
        # pyre-ignore[16]: Undefined attribute [16]
        * x_scale.repeat_interleave(block_m, dim=0).repeat_interleave(block_k, dim=1)
    )[:M, :K]

    # Cast and move data to output device (for cpu weight loading).
    x_fp8 = x_fp8.to(device=output_device, dtype=pt_dtype)
    x_scale = x_scale.to(output_device)  # pyre-ignore
    del x, x_padded
    if not k_major:
        x_scale = x_scale.t().contiguous()
    return x_fp8.view(x_shape), 1 / x_scale  # pyre-ignore


@quantize_fp8_block.register_fake
def quantize_fp8_block_meta(
    a: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
    scale_ub: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
    k_major: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shape function for torch compile."""
    if output_device is None:
        output_device = a.device
    a_shape = a.shape
    dtype = get_fp8_constants()[0]
    fake_out = torch.empty(a.shape, device=output_device, dtype=dtype)
    scale_m = triton.cdiv(a_shape[0], block_m)
    scale_k = triton.cdiv(a_shape[1], block_k)
    scale_out_shape = (
        a_shape[:-2] + (scale_m, scale_k) if k_major else (scale_k, scale_m)
    )
    fake_scale = torch.empty(
        scale_out_shape,
        device=output_device,
        dtype=torch.float32,
    )
    return fake_out, fake_scale


@triton.autotune(
    configs=[
        Config({"GROUP_LOAD": 2}),
        Config({"GROUP_LOAD": 4}),
        Config({"GROUP_LOAD": 8}),
        Config({"GROUP_LOAD": 16}),
        Config({"GROUP_LOAD": 32}),
    ],
    key=["K"],
)
@triton.jit
def _kernel_quantize_fp8_group(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    m_sizes,
    M,
    K,
    stride_am,
    stride_ak,
    stride_om,
    stride_ok,
    stride_a_scale_m,
    stride_a_scale_k,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    USE_INT64: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    USE_M_MAJOR: tl.constexpr,
    G: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
):
    """Quantize and scale each GROUP_SIZE chunk of each row.

    Scale per group i is computed as 1 / (MAX_FP8 / max(abs(A[i:i+GROUP_SIZE])))

    Each kernel thread is responsible for one row and loads and processes a tunable
    number of groups at once.

    Args:
        A (Tensor): [M, K] higher precision input tensor.
        A_scale (Tensor): [M, cdiv(K, GROUP_SIZE)] reciprocal scale tensor per group.
        A_fp8 (Tensor): [M, K] fp8 scaled tensor. A_fp8 = A * a
        scale_ub (Tensor): [1] Maximum allowed value for scale.
        m_sizes (Optional[Tensor]): [G] Number of rows in each group.
        M (int): Number of rows.
        K (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_a_scale_m (int): Stride of m dimension of A_scale.
        stride_a_scale_k (int): Stride of k dimension of A_scale.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whether to apply scale_ub.
        USE_INT64 (bool): Whether to index using int64, which may be needed for large tensors.
        GROUP_SIZE (int): Group size for K dimension of A_scale and kernel.
        USE_M_MAJOR (bool): Whether to use grouped M-major layout for A_scale.
        G (int): Number of groups in A_scale, only relevant when m_sizes is provided.
        GROUP_LOAD (int): Number of groups to load and process simultaneously.
    """
    pid = tl.program_id(0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    # We load group_size * group_load chunks at a time.
    row_offset = pid * stride_am
    out_offset = pid * stride_om
    scale_row_offset = pid * stride_a_scale_m
    k_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE)
    scale_k_offset = tl.arange(0, GROUP_LOAD)
    NUM_GROUPS: tl.constexpr = K // GROUP_SIZE

    # When dealing with an M-major grouped gemm, we need to figure out
    # which group this thread corresponds to and figure out the corresponding
    # scale offset.
    group_offset = 0
    group_cumsum = 0
    group_M = 0
    stop = False
    if USE_M_MAJOR and G > 0:
        # Iterate over groups to both compute the cumulative sum and find which group we are in.
        for i in range(G):
            if not stop:
                group_M = tl.cast(tl.load(m_sizes + i), pid.dtype)
                if (group_cumsum + group_M) <= pid:
                    group_cumsum += group_M
                else:
                    # Indicate we are finished computing cumsum.
                    stop = True

        group_offset = group_cumsum * NUM_GROUPS

    for k in range(0, tl.cdiv(K, (GROUP_LOAD * GROUP_SIZE))):
        # Load groups of the input.
        chunk_offset = k_offset + k * GROUP_LOAD * GROUP_SIZE
        a = tl.load(
            A + row_offset + chunk_offset * stride_ak, mask=chunk_offset < K, other=0.0
        )
        # View loaded chunk as a set of groups.
        a_grouped = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Reduce over groups.
        group_max = tl.max(tl.abs(a_grouped), axis=1)
        # Apply clamping if specified.
        if CLAMP_MAX:
            ub = tl.load(scale_ub)
            group_max = tl.clamp(group_max, EPS, ub)
        else:
            group_max = tl.maximum(group_max, EPS)
        # Scale and quantize.
        a_scale = MAX_FP8 / group_max
        scale_chunk_offset = scale_k_offset + k * GROUP_LOAD

        if USE_M_MAJOR and G > 0:
            tl.store(
                A_scale
                + group_offset
                + (pid - group_cumsum) * stride_a_scale_k
                + (scale_chunk_offset * group_M),
                1.0 / a_scale,
                mask=scale_chunk_offset < NUM_GROUPS,
            )
        else:
            if USE_M_MAJOR:
                tl.store(
                    A_scale
                    + pid * stride_a_scale_k
                    + scale_chunk_offset * stride_a_scale_m,
                    1.0 / a_scale,
                    mask=scale_chunk_offset < NUM_GROUPS,
                )
            else:
                tl.store(
                    A_scale + scale_row_offset + scale_chunk_offset * stride_a_scale_k,
                    1.0 / a_scale,
                    mask=scale_chunk_offset < NUM_GROUPS,
                )
        # Apply scale to input.
        a_fp8 = a_grouped * a_scale[:, None]
        # Clamp to FP8 range to avoid overflow
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        # Write to output.
        tl.store(
            A_fp8 + out_offset + chunk_offset * stride_ok,
            tl.ravel(a_fp8),
            mask=chunk_offset < K,
        )


def triton_quantize_fp8_group(
    x: torch.Tensor,
    group_size: int = 128,
    scale_ub: Optional[torch.Tensor] = None,
    m_sizes: Optional[torch.Tensor] = None,
    k_major: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with group-wise scalings.

    Scale per group i is computed as 1 / (MAX_FP8 / max(abs(x[i:i+group_size])))

    Args:
        x (torch.Tensor): [M, K] higher precision input tensor.
        group_size (int): Group size for M dimension of scale.
        scale_ub: Maximum allowed value for scale.
        m_sizes: Optional input for grouped gemm to specify the number of rows in each group.
        k_major (bool): Whether output scales should be K major (True) or MN major (False).

    Returns:
        torch.Tensor: [M, K] fp8 scaled tensor.
        torch.Tensor: [M, cdiv(K, group_size)] reciprocal scale tensor per group.
    """
    assert x.device != torch.device(
        "cpu"
    ), "Triton groupwise quantization not supported on cpu."

    if scale_ub is not None and scale_ub.device != x.device:
        raise Exception("'scale_ub' must be on the same device as 'a'")
    if m_sizes is not None and m_sizes.device != x.device:
        raise Exception("'m_sizes' must be on the same device as 'a'")

    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    M, K = x.shape
    k_groups = triton.cdiv(K, group_size)
    if k_major:
        x_scale = torch.empty((M, k_groups), device=x.device, dtype=torch.float32)
    else:
        x_scale = torch.empty((k_groups, M), device=x.device, dtype=torch.float32)
    x_fp8 = torch.empty((M, K), device=x.device, dtype=pt_dtype)
    _kernel_quantize_fp8_group[(M,)](
        x,
        x_scale,
        x_fp8,
        scale_ub,
        m_sizes,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        TL_FP8_DTYPE=tl_dtype,
        MAX_FP8=max_fp8,
        EPS=eps,
        CLAMP_MAX=scale_ub is not None,
        USE_INT64=x.numel() > (2**32 - 1),
        GROUP_SIZE=group_size,
        USE_M_MAJOR=m_sizes is not None or k_major is False,
        G=m_sizes.numel() if m_sizes is not None else 0,
    )
    return x_fp8.view(x_shape), x_scale


def quantize_fp8_group(
    x: torch.Tensor,
    group_size: int = 128,
    scale_ub: Optional[torch.Tensor] = None,
    m_sizes: Optional[torch.Tensor] = None,
    k_major: bool = True,
    use_triton: bool = True,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with group-wise scalings and optionally move to output device.

    Scale per group i is computed as 1 / (MAX_FP8 / max(abs(x[i:i+group_size])))

    Args:
        x (Tensor): [M, K] higher precision input tensor.
        group_size (int): Group size for M dimension of scale.
        scale_ub: Maximum allowed value for scale.
        m_sizes: Optional input for grouped gemm to specify the number of rows in each group.
        k_major (bool): Whether output scales should be K major (True) or MN major (False).
        This is needed because some kernels like cutlass require a special layout for scales.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.

    Returns:
        torch.Tensor: [M, K] fp8 scaled tensor.
        torch.Tensor: [M, cdiv(K, group_size)] reciprocal scale tensor per group.
    """
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    if x.device == torch.device("cpu"):
        logger.info("Triton does not support cpu, falling back to torch ops.")
        use_triton = False
    if use_triton:
        xq, x_scale = triton_quantize_fp8_group(
            x, group_size, scale_ub, m_sizes, k_major
        )
        return xq.view(x_shape), x_scale
    # else use pytorch implementation.
    if not output_device:
        output_device = x.device

    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()

    M, K = x.shape
    assert (
        K % group_size == 0
    ), "K must be divisible by group_size for cpu implementation."
    assert m_sizes is None, "m_sizes is not supported for cpu implementation."
    k_groups = triton.cdiv(K, group_size)
    # View input as colleciton of groups for reduction.
    x_grouped = x.view(M, k_groups, group_size).to(torch.float32)
    # Reduce over groups.
    group_max = x_grouped.abs().amax(dim=2)
    # Apply clamping.
    group_max = (
        torch.clamp(group_max, min=eps, max=scale_ub.item())
        if scale_ub
        else torch.clamp(group_max, min=eps)
    )
    x_scale = torch.empty((M, k_groups), dtype=torch.float32, device=output_device)
    x_scale = max_fp8 / group_max  # pyre-ignore
    # pyre-ignore[16]: Undefined attribute [16]
    x_scale[x_scale == float("inf")] = 1.0
    # pyre-ignore[16]: Undefined attribute [16]
    x_fp8 = x.view(-1, k_groups, group_size) * x_scale.unsqueeze(2)
    # Cast and move data to output device (for cpu weight loading).
    x_fp8 = x_fp8.to(device=output_device, dtype=pt_dtype)
    x_scale = x_scale.to(output_device)  # pyre-ignore
    if not k_major:
        x_scale = x_scale.t().contiguous()
    return x_fp8.view(x_shape), 1 / x_scale  # pyre-ignore


@triton.autotune(
    configs=[Config({"BLOCK_M": 16, "BLOCK_K": 512, "NUM_STAGES": 2})],
    key=["M", "K"],
)
@triton.jit
def _kernel_dequantize_fp8_row(
    xq_ptr,
    x_scale_ptr,
    x_dequant_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_xdqm,
    stride_xdqk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    """
    Kernel to dequantize FP8 tensor to BF16 tensor.
    Args:
        xq_ptr (tl.constexpr): Pointer to FP8 tensor.
        x_scale_ptr (tl.constexpr): Pointer to FP8 scale tensor.
        x_dequant_ptr (tl.constexpr): Pointer to BF16 tensor.
        M (tl.constexpr): M dimension of input tensor.
        K (tl.constexpr): K dimension of input tensor (along which scales are applied)
        BLOCK_SIZE (tl.constexpr): Block size for the K dimension.
    """
    pid = tl.program_id(axis=0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    scales = tl.load(x_scale_ptr + offs_m)

    for _k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        xq = tl.load(
            xq_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask,
        )
        x_dq = xq * scales[:, None]
        tl.store(
            x_dequant_ptr
            + offs_m[:, None] * stride_xdqm
            + offs_k[None, :] * stride_xdqk,
            x_dq,
            mask=mask,
        )
        offs_k += BLOCK_K


def dequantize_fp8_row(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Rowwise Dequantize FP8 tensor to BF16 tensor along last axis.

    Args:
        xq (torch.Tensor): FP8 tensor to be dequantized.
        x_scale (torch.Tensor): FP8 scale tensor.

    Returns:
        torch.Tensor: Dequantized BF16 tensor.
    """

    assert (
        xq.is_contiguous() and x_scale.is_contiguous()
    ), "Input tensors must be contiguous"
    x_dequant = torch.empty_like(xq, dtype=torch.bfloat16)

    # Reshape to 2-d array keeping last dim only.
    K = xq.shape[-1]
    xq = xq.reshape(-1, K)
    M = xq.shape[0]
    use_int64 = xq.numel() > 2**31

    def grid(meta: Dict[str, int]) -> Tuple[int]:
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    with torch.cuda.device(xq.device.index):
        _kernel_dequantize_fp8_row[grid](
            xq,
            x_scale,
            x_dequant,
            M,
            K,
            xq.stride(0),
            xq.stride(1),
            xq.stride(0),  # Use squashed stride.
            xq.stride(1),
            USE_INT64=use_int64,
        )
    return x_dequant


@triton.autotune(
    configs=[Config({"BLOCK_M": 16, "BLOCK_K": 512, "NUM_STAGES": 2})],
    key=["M", "K"],
)
@triton.jit
def _kernel_dequantize_fp8_packed_row(
    xq_ptr,
    x_scale_ptr,
    x_dequant_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_xdqm,
    stride_xdqk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    """
    Kernel to dequantize FP8 tensor to BF16 tensor.
    Args:
        xq_ptr (tl.constexpr): Pointer to FP8 tensor.
        x_scale_ptr (tl.constexpr): Pointer to FP8 scale tensor.
        x_dequant_ptr (tl.constexpr): Pointer to BF16 tensor.
        M (tl.constexpr): M dimension of input tensor.
        K (tl.constexpr): K dimension of input tensor (along which scales are applied)
        BLOCK_SIZE (tl.constexpr): Block size for the K dimension.
    """
    pid = tl.program_id(axis=0)
    if USE_INT64:
        pid = pid.to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    scales = tl.load(x_scale_ptr + offs_m)

    for _k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

        xq = tl.load(
            xq_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask,
            other=0.0,
        )
        x_dq = xq * scales[:, None]

        tl.store(
            x_dequant_ptr
            + offs_m[:, None] * stride_xdqm
            + offs_k[None, :] * stride_xdqk,
            x_dq,
            mask=mask,
        )
        offs_k += BLOCK_K


def dequantize_fp8_packed_row(
    xq: torch.Tensor,
) -> torch.Tensor:
    """
    Rowwise Dequantize FP8 tensor to BF16 tensor along last axis.

    Args:
        xq (torch.Tensor): Packed FP8 tensor to be dequantized. The last 4 bytes of each row is the FP32 scale for that row.

    Returns:
        torch.Tensor: Dequantized BF16 tensor.
    """

    # Create a view of the packed tensors, get the scale and actual xq tensor
    # This makes it much easier to write the kernel
    orig_shape = (*xq.shape[:-1], xq.shape[-1] - 4)
    actual_xq = xq[..., :-4].view(orig_shape)

    assert xq.is_contiguous(), "Input tensors must be contiguous"
    x_dequant = torch.empty(orig_shape, dtype=torch.bfloat16, device=xq.device)

    # Calculate number of rows when flattened
    num_rows = actual_xq.numel() // actual_xq.shape[-1]

    # TODO: we take a perf hit from these reshapes, can we do better?
    # It's hard to skip this reshape, we can't create a int32/float32 view because of alignment issues
    scale_view = xq[..., -4:].reshape((num_rows * 4)).view(torch.float32)
    scale_view = scale_view.view(orig_shape[:-1])

    # Reshape to 2-d array keeping last dim only.
    K = actual_xq.shape[-1]
    actual_xq = actual_xq.reshape(-1, K)
    M = actual_xq.shape[0]
    use_int64 = actual_xq.numel() > 2**31

    def grid(meta: Dict[str, int]) -> Tuple[int]:
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    with torch.cuda.device(actual_xq.device.index):
        _kernel_dequantize_fp8_packed_row[grid](
            actual_xq,
            scale_view,
            x_dequant,
            M,
            K,
            actual_xq.stride(0),
            actual_xq.stride(1),
            x_dequant.stride(-2),  # Use squashed stride.
            x_dequant.stride(-1),
            USE_INT64=use_int64,
        )

    return x_dequant


@triton.jit
def _kernel_quantize_fp8_tensor(
    A,
    A_fp8,
    global_max_ptr,
    blocks_done_ptr,
    scale_ready_ptr,
    scale_out_ptr,
    N,
    num_sms,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Fused persistent kernel that finds global max and quantizes.

    Uses a persistent kernel approach where we launch exactly num_sms blocks,
    guaranteeing all blocks run concurrently and avoiding deadlocks.
    Each block processes multiple chunks of the input in a loop.

    Args:
        A (Tensor): Flattened input tensor.
        A_fp8 (Tensor): Output fp8 tensor.
        global_max_ptr (Tensor): Pointer to global max value (initialized to 0).
        blocks_done_ptr (Tensor): Pointer to atomic counter (initialized to 0).
        scale_ready_ptr (Tensor): Pointer to ready flag (initialized to 0).
        scale_out_ptr (Tensor): Pointer to output scale value.
        N (int): Total number of elements.
        num_sms (int): Number of SMs (equals number of blocks launched).
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maximum expressible value for FP8.
        EPS (float): Epsilon for numerical stability.
        BLOCK_SIZE (int): Block size for processing.
    """
    pid = tl.program_id(0)

    # Phase 1: Each block finds max across all its assigned chunks
    local_max = 0.0
    chunk_id = pid
    num_chunks = tl.cdiv(N, BLOCK_SIZE)

    while chunk_id < num_chunks:
        offset = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + offset, mask=offset < N, other=0.0)
        chunk_max = tl.max(tl.abs(a))
        local_max = tl.maximum(local_max, chunk_max)
        chunk_id += num_sms

    # Atomically update global max using integer atomics on float bits
    local_max_int = local_max.to(tl.float32, bitcast=False).to(tl.int32, bitcast=True)
    tl.atomic_max(global_max_ptr, local_max_int)

    # Increment completed block counter
    old_count = tl.atomic_add(blocks_done_ptr, 1)

    # Last block to finish computes the scale
    if old_count == num_sms - 1:
        global_max_int = tl.load(global_max_ptr)
        global_max_float = global_max_int.to(tl.float32, bitcast=True)
        global_max_float = tl.maximum(global_max_float, EPS)
        scale = tl.div_rn(global_max_float, MAX_FP8)
        tl.store(scale_out_ptr, scale)
        tl.atomic_xchg(scale_ready_ptr, 1)

    # Phase 2: Spin-wait for scale to be ready
    # Safe because all num_sms blocks are guaranteed to be running
    while tl.atomic_add(scale_ready_ptr, 0) == 0:
        pass

    # Load scale and quantize all assigned chunks
    scale = tl.load(scale_out_ptr)
    chunk_id = pid

    while chunk_id < num_chunks:
        offset = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + offset, mask=offset < N, other=0.0)
        a_fp8 = a * tl.div_rn(1.0, scale)
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        tl.store(A_fp8 + offset, a_fp8, mask=offset < N)
        chunk_id += num_sms


def _get_num_sms(device: torch.device) -> int:
    """Get the number of SMs on the current GPU device."""
    return torch.cuda.get_device_properties(device).multi_processor_count


def triton_quantize_fp8_tensor(
    a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation to quantize a tensor to fp8 with a single scale.

    Uses a fused persistent kernel with atomic operations for inter-block
    coordination. By launching exactly num_sms blocks, we guarantee all
    blocks run concurrently, avoiding deadlocks from spin-waiting.

    Args:
        a (Tensor): Input tensor to be quantized.

    Returns:
        torch.Tensor: fp8 quantized tensor.
        torch.Tensor: scalar reciprocal scale tensor (fp32).
    """
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    N = a.numel()

    BLOCK_SIZE = 4096
    # Launch exactly num_sms blocks to guarantee concurrent execution
    num_sms = _get_num_sms(a.device)

    # Allocate synchronization buffers (initialized to 0)
    global_max = torch.zeros(1, device=a.device, dtype=torch.int32)
    blocks_done = torch.zeros(1, device=a.device, dtype=torch.int32)
    scale_ready = torch.zeros(1, device=a.device, dtype=torch.int32)
    scale_out = torch.empty((), device=a.device, dtype=torch.float32)

    # Output tensor matches shape of a but is contiguous.
    a_fp8 = torch.empty_like(a, dtype=pt_dtype)

    with torch.cuda.device(a.device.index):
        _kernel_quantize_fp8_tensor[(num_sms,)](
            a,
            a_fp8,
            global_max,
            blocks_done,
            scale_ready,
            scale_out,
            N,
            num_sms,
            # pyre-ignore[6]: Incompatible parameter type
            TL_FP8_DTYPE=tl_dtype,
            # pyre-ignore[6]: Incompatible parameter type
            MAX_FP8=max_fp8,
            # pyre-ignore[6]: Incompatible parameter type
            EPS=eps,
            # pyre-ignore[6]: Incompatible parameter type
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return a_fp8, scale_out


@torch.library.custom_op("triton::quantize_fp8_tensor", mutates_args=())
def quantize_fp8_tensor(
    a: torch.Tensor,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with a single scale factor across the entire tensor.

    The scale is computed as MAX_FP8 / max(abs(a)) and applied uniformly.
    Handles non-contiguous input tensors and returns a contiguous output.

    Args:
        a (Tensor): Input tensor of any shape. May be non-contiguous.
        use_triton (bool): Whether to use optimized triton kernel.

    Returns:
        torch.Tensor: fp8 quantized tensor (contiguous, same shape as input).
        torch.Tensor: scalar reciprocal scale tensor (fp32).
    """
    if a.device == torch.device("cpu"):
        use_triton = False

    if use_triton:
        a_fp8, reciprocal_scale = triton_quantize_fp8_tensor(a)
        return a_fp8, reciprocal_scale

    # Fallback to PyTorch implementation
    pt_dtype, _, max_fp8, eps = get_fp8_constants()

    tensor_max = torch.max(torch.abs(a)).to(torch.float32)
    tensor_max = torch.clamp(tensor_max, min=eps)

    scale = max_fp8 / tensor_max  # pyre-ignore[58]
    a_scaled = a.to(torch.float32) * scale
    a_scaled = torch.clamp(a_scaled, -max_fp8, max_fp8)
    a_fp8 = a_scaled.to(pt_dtype)

    reciprocal_scale = (1.0 / scale).to(torch.float32)  # pyre-ignore[16]

    return a_fp8, reciprocal_scale


@quantize_fp8_tensor.register_fake
def quantize_fp8_tensor_meta(
    a: torch.Tensor,
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shape function for torch compile."""
    dtype = get_fp8_constants()[0]
    # Preserve memory format (e.g., channels_last_3d) from input tensor
    fake_out = torch.empty_like(a, dtype=dtype)
    fake_scale = torch.empty((), device=a.device, dtype=torch.float32)
    return fake_out, fake_scale


@triton.jit
def _kernel_dequantize_fp8_block(
    xq_ptr,
    x_scale_ptr,
    x_dequant_ptr,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Kernel to dequantize FP8 tensor to BF16 tensor.
    Args:
        xq_ptr (tl.constexpr): Pointer to FP8 tensor.
        x_scale_ptr (tl.constexpr): Pointer to FP8 scale tensor.
        x_dequant_ptr (tl.constexpr): Pointer to BF16 tensor.
        M (tl.constexpr): M dimension of input tensor.
        K (tl.constexpr): K dimension of input tensor.
        BLOCK_M (tl.constexpr): Block size for the M dimension.
        BLOCK_K (tl.constexpr): Block size for the K dimension.
    """
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_K)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs = offs_m[:, None] * K + offs_k[None, :]
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    xq = tl.load(xq_ptr + offs, mask=mask).to(tl.bfloat16)
    x_scale = tl.load(x_scale_ptr + pid_m * k + pid_k)
    x_dequant = xq * x_scale
    tl.store(x_dequant_ptr + offs, x_dequant, mask=mask)


def dequantize_fp8_block(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
) -> torch.Tensor:
    """
    Dequantize FP8 tensor to BF16 tensor.

    Args:
        xq (torch.Tensor): FP8 tensor to be dequantized.
        x_scale (torch.Tensor): FP8 scale tensor.
        block_m (int): Block size for the M dimension.
        block_k (int): Block size for the K dimension.

    Returns:
        torch.Tensor: Dequantized BF16 tensor.
    """

    assert (
        xq.is_contiguous() and x_scale.is_contiguous()
    ), "Input tensors must be contiguous"
    assert xq.dim() == 2 and x_scale.dim() == 2, "Input tensors must have 2 dimensions"
    M, K = xq.size()
    x_dequant = torch.empty_like(xq, dtype=torch.bfloat16)

    def grid(meta: Dict[str, int]) -> Tuple[int, int]:
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(K, meta["BLOCK_K"]),
        )

    with torch.cuda.device(xq.device.index):
        _kernel_dequantize_fp8_block[grid](
            xq,
            x_scale,
            x_dequant,
            M,
            K,
            BLOCK_M=block_m,  # pyre-ignore[6]
            BLOCK_K=block_k,  # pyre-ignore[6]
        )
    return x_dequant
