# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe
# pyre-ignore-all-errors[29]

import logging
import random
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

import pytest
import torch
from mslk.attention import fmha
from mslk.attention.fmha.attn_bias_utils import create_attn_bias
from mslk.attention.fmha.common import AttentionOpBase

from .utils import _devices

logger = logging.getLogger("xformers")


def sample_random_supported_fw(
    inp: fmha.Inputs,
    fw_ops: Sequence[Type[fmha.common.AttentionFwOpBase]],
    seed: Union[int, str],
    op_bw: Type[fmha.common.AttentionBwOpBase],
) -> Type[fmha.common.AttentionFwOpBase]:
    r = random.Random(seed)
    fw_ops = list(fw_ops)
    if op_bw == fmha.cutlass_blackwell.BwOp:
        fw_ops = [fmha.cutlass_blackwell.FwOp, fmha.flash.FwOp]
    if (
        isinstance(inp.attn_bias, fmha.attn_bias.VARLEN_BIASES)
        and inp.attn_bias.q_seqinfo.seqstart.shape[0] > 2
    ):
        fw_ops = [
            op for op in fw_ops if op.VARLEN_LSE_PACKED == op_bw.VARLEN_LSE_PACKED
        ]
    r.shuffle(fw_ops)
    for op in fw_ops:
        if op.supports(inp):
            return op
    raise NotImplementedError(f"Could not find a FW operator for: {inp}")


def generate_test_shapes_B_Mq_Mkv_H_K_Kv(
    op: Type[fmha.AttentionOpBase],
) -> List[Tuple[int, ...]]:  # noqa: C901
    shapes: List[Tuple[int, ...]] = []
    for B in op._TEST_BATCH_SIZES:
        for Mq in [32, 256]:
            for Mkv in [32, 64, 256, 1024]:
                for K in op._TEST_K:
                    shapes.append((B, Mq, Mkv, 1, K, K))
        Mq = 256
        Mkv = 128
        K = 32
        H = 1
        # Weird values of parameters
        for M in [2, 3, 15, 31, 32, 34, 68, 72, 90, 132, 136]:
            shapes.append((B, M, Mkv, H, K, K))
            shapes.append((B, Mq, M, H, K, K))
        Ks = [1, 2, 3, 31, 34, 36, 38, 40, 64, 80, 160, 256 + 2, 256 + 8, 512]
        for _K in Ks:
            if op.SUPPORTED_MIN_K <= _K <= op.SUPPORTED_MAX_K:
                shapes.append((B, Mq, Mkv, H, _K, _K))
        # Different value for K / Kv
        if op.SUPPORTS_DIFFERENT_VALUE_EMBED:
            for _K in [32, 36, 64, 256 + 8]:
                shapes.append((B, Mq, Mkv, H, K, _K))
                shapes.append((B, Mq, Mkv, H, _K, K))
        # Exotic sizes
        for K in op._TEST_K:
            shapes.append((B, 16, 1024, H, K, K))
            shapes.append((B, 1024, 16, H, K, K))
        # Some number of heads
        for H in [3, 5, 12]:
            shapes.append((max(1, B // H), Mq, Mkv, H, K, K))
    # Filter-out not supported shapes
    shapes = [
        shape
        for shape in shapes
        if len(
            op.shape_not_supported_reasons(
                Mq=shape[1], Mkv=shape[2], K=shape[4], Kv=shape[5]
            )
        )
        == 0
    ]
    # Add some random shapes
    if op in [
        fmha.cutlass.FwOp,
        fmha.cutlass.BwOp,
        fmha.cutlass_blackwell.FwOp,
        fmha.cutlass_blackwell.BwOp,
        fmha.flash.BwOp,
        fmha.ck.FwOp,
    ]:
        K_CHOICES = [8 * i for i in range(1, 256 // 8)]
        r = random.Random(0)
        found_count = 0
        while found_count < 200:
            B = r.randint(1, 400)
            Mq = r.randint(1, 500)
            Mkv = r.randint(1, 500)
            H = r.randint(2, 11)
            B = max(B // H, 1)
            K = r.choice(K_CHOICES)
            Kv = r.choice(K_CHOICES)
            if not op.SUPPORTS_DIFFERENT_VALUE_EMBED:
                Kv = K
            if len(op.shape_not_supported_reasons(Mq, Mkv, K, Kv)):
                continue
            found_count += 1
            shapes.append((B, Mq, Mkv, H, K, Kv))
    return shapes


def make_id(
    op: Type[fmha.AttentionOpBase],
    device: torch.device,
    dtype: torch.dtype,
    bias_type: Type[fmha.attn_bias.AttentionBias],
    *shape: int,
) -> str:
    return (
        f"{op.NAME}-{device}-{str(dtype)}-{bias_type.__name__}"
        f"-{'-'.join([str(s) for s in shape])}"
    )


# This temporary working is necessary because the MTIA test collection might not happen
# on the same device as the device the tests are actually executed on. If test collection
# is done on a device without MTIA, the supported masks will contain masks that MTIA support
# and the corresponding tests will get collected. But when it comes time to actually run the
# tests, the mask won't be supported because it is run on an actual MTIA device.
def get_supported_attn_bias_types(op: Type[fmha.AttentionOpBase]) -> Iterable[Any]:
    supported_attn_bias_types = op.SUPPORTED_ATTN_BIAS_TYPES

    try:
        import mtia.host_runtime.torch_mtia.dynamic_library  # noqa  # type: ignore

        supported_attn_bias_types = [
            b
            for b in supported_attn_bias_types
            if not issubclass(
                b,
                (
                    fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
                    fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
                ),
            )
        ]
    except (ImportError, OSError):
        pass

    return supported_attn_bias_types


def _generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(  # noqa: C901
    ops_list: Sequence[Type[fmha.AttentionOpBase]], max_shapes_per_op: int = 65000
):
    r = random.Random(0)
    combination = []
    for op in ops_list:
        op_count = 0
        # Sort list of masks, so it's deterministic across runs
        LIST_MASKS = sorted(get_supported_attn_bias_types(op), key=str)
        for shape in generate_test_shapes_B_Mq_Mkv_H_K_Kv(op):
            has_one = False
            for device in _devices:
                if device not in op.SUPPORTED_DEVICES:
                    continue
                # Sort set of dtypes to make it deterministic across runs
                for dtype in sorted(op.SUPPORTED_DTYPES, key=str):
                    # "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'
                    if dtype in [torch.float8_e4m3fn]:
                        continue
                    bias_type = r.choice(LIST_MASKS)
                    # Avoid using too much memory
                    B, Mq, Mkv, H, K, Kv = shape
                    if bias_type not in [
                        type(None),
                        fmha.attn_bias.LowerTriangularMask,
                    ]:
                        B = min(B, 12)

                        if bias_type in {
                            fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
                            fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask,
                        }:
                            Mq, Mkv = min(Mkv, Mq), max(Mkv, Mq) + 2
                        elif bias_type in {
                            fmha.attn_bias.BlockDiagonalCausalLocalAttentionPaddedKeysMask,
                            fmha.attn_bias.BlockDiagonalLocalAttentionPaddedKeysMask,
                            fmha.attn_bias.BlockDiagonalCausalWithOffsetGappyKeysMask,
                            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
                            fmha.attn_bias.BlockDiagonalPaddedKeysMask,
                            fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
                            fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
                            fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
                            fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
                        }:
                            Mq, Mkv = min(Mkv, Mq), max(Mkv, Mq)
                    new_shape = (B, Mq, Mkv, H, K, Kv)
                    combination.append((op, device, dtype, bias_type, *new_shape))
                    has_one = True
            if has_one:
                op_count += 1
            if op_count > max_shapes_per_op:
                break
        # Some specific shapes for which we want to run without any mask
        bias_type = type(None)
        for shape in (
            # Some strides/dims don't fit on an uint16
            (1, 128, 128, 300, 128, 128),
            (13, 1, 67, 200, 8, 8),
            (1, 1 + 2**16, 4, 1, 8, 8),
            (1, 4, 1 + 2**16, 1, 8, 8),
            # TODO: Some strides don't fit on an uint32
            # Crashes on Flash, Errors on Cutlass
            # (1, 1, 64000, 300, 128, 128)
        ):
            for device in _devices:
                if device not in op.SUPPORTED_DEVICES:
                    continue
                # Sort set of dtypes to make it deterministic across runs
                for dtype in sorted(op.SUPPORTED_DTYPES, key=str):
                    # "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'
                    if dtype in [torch.float8_e4m3fn]:
                        continue
                    combination.append((op, device, dtype, bias_type, *shape))
    return {
        "argvalues": combination,
        "ids": [make_id(*c) for c in combination],
    }


def get_bias_grad(attn_bias: Any, clear: bool = False) -> Optional[torch.Tensor]:
    tensor_with_grad: Optional[torch.Tensor] = None
    if isinstance(attn_bias, torch.Tensor):
        tensor_with_grad = attn_bias
    if tensor_with_grad is not None:
        grad = tensor_with_grad.grad
        if clear:
            tensor_with_grad.grad = None
        return grad
    return None


def create_tensors(  # noqa: C901
    op: Optional[Type[AttentionOpBase]],
    device: Union[str, torch.device],
    dtype: torch.dtype,
    attn_bias_type: Optional[Type[fmha.attn_bias.AttentionBias]],
    B: int,
    q_len: int,
    kv_len: int,
    h: int,
    k: int,
    kv: int,
    *,
    attn_bias_requires_grad: bool = False,
    fmt: str = "BMK",
    g: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    torch.manual_seed(B * q_len + kv_len * k + kv)

    mask_is_bottom_right = attn_bias_type is not None and issubclass(
        attn_bias_type,
        (
            fmha.attn_bias.LowerTriangularFromBottomRightMask,
            fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask,
            fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
            fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            fmha.attn_bias.BlockDiagonalLocalAttentionFromBottomRightGappyKeysMask,
            fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask,
            fmha.attn_bias.LocalAttentionFromBottomRightMask,
            fmha.attn_bias.PagedBlockDiagonalCausalLocalPaddedKeysMask,
        ),
    )
    if mask_is_bottom_right and q_len > kv_len:
        # Bottom-right attention and local-attention masks require q_len <= kv_len
        kv_len = q_len

    if attn_bias_type is not None and issubclass(
        attn_bias_type,
        (
            fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
            fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
        ),
    ):
        page_size_choices = [256, 512]
        if op is not None and issubclass(op, fmha.triton_splitk.FwOp):
            # TODO: enable small pages for flash attention when that's implemented
            page_size_choices.extend([64, 128])
        page_size = random.choice(page_size_choices)
        kv_len_paged = (kv_len + page_size - 1) // page_size * page_size
    else:
        kv_len_paged = kv_len
        page_size = None

    scale = 3
    if fmt == "BMK":
        query = torch.randn((B * h, q_len, k), device=device, dtype=dtype)
        key = torch.randn((B * h, kv_len_paged, k), device=device, dtype=dtype)
        value = torch.randn((B * h, kv_len_paged, kv), device=device, dtype=dtype)
    elif fmt == "BMHK":
        query = torch.randn((B, q_len, h, k), device=device, dtype=dtype)
        key = torch.randn((B, kv_len_paged, h, k), device=device, dtype=dtype)
        value = torch.randn((B, kv_len_paged, h, kv), device=device, dtype=dtype)
    else:
        assert fmt == "BMGHK"
        query = torch.randn((B, q_len, g, h, k), device=device, dtype=dtype)
        key = torch.randn((B, kv_len_paged, g, 1, k), device=device, dtype=dtype)
        value = torch.randn((B, kv_len_paged, g, 1, kv), device=device, dtype=dtype)

    for x in [query, key, value]:
        x.mul_(scale)

    if fmt == "BMGHK":
        # Expand - after the in-place mul
        key = key.expand((B, kv_len_paged, g, h, k))
        value = value.expand((B, kv_len_paged, g, h, k))

    if fmt == "BMK" and not fmha.common._is_bias_type_supported_in_BMK(attn_bias_type):
        attn_bias_type = None
    attn_bias = None
    if attn_bias_type is not None:
        attn_bias = create_attn_bias(
            attn_bias_type,
            batch_size=B,
            num_heads=h,
            num_heads_groups=g,
            q_len=q_len,
            kv_len=kv_len,
            dtype=dtype,
            device=device,
            requires_grad=attn_bias_requires_grad,
            fmt=fmt,
            op=op,
            page_size=page_size,
        )
        if isinstance(
            attn_bias,
            (
                fmha.attn_bias.BlockDiagonalMask,
                fmha.attn_bias.BlockDiagonalGappyKeysMask,
                fmha.attn_bias.BlockDiagonalPaddedKeysMask,
                fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
                fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
            ),
        ):
            query, key, value = [
                x.reshape([1, -1, *x.shape[2:]]) for x in [query, key, value]
            ]

    inputs = fmha.Inputs(query=query, key=key, value=value, attn_bias=attn_bias)
    if op is not None:
        reasons = op.not_supported_reasons(inputs)
        if reasons:
            err_msg = f"{op.NAME}: unsupported ({'/'.join(reasons)})"
            # Ensure we free memory to avoid OOMs
            del query, key, value, attn_bias, inputs
            pytest.skip(err_msg)
    return query, key, value, attn_bias
