# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# @nolint # fbcode

from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple, Union

import torch

from .attn_bias import (
    AttentionBias,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionPaddedKeysMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalLocalAttentionPaddedKeysMask,
    BlockDiagonalMask,
    BlockDiagonalPaddedKeysMask,
    LocalAttentionFromBottomRightMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    bmghk2bmhk,
    bmhk2bhk,
    Context,
    Gradients,
    Inputs,
)

_ERROR_ATOL: Mapping[torch.dtype, float] = {
    torch.float: 3e-4,
    torch.half: 3e-2,
    torch.bfloat16: 3e-2,
}

_ERROR_RTOL: Mapping[torch.dtype, float] = {
    torch.float: 2e-5,
    torch.half: 3e-2,
    torch.bfloat16: 3e-2,
}


def _get_operator(name: str):
    def no_such_operator(*args, **kwargs):
        raise RuntimeError(
            f"No such operator mslk.attention.flash_attn.interface.{name}"
        )

    try:
        import mslk.attention.flash_attn.interface as flash_attn

        return getattr(flash_attn, name)
    except (RuntimeError, ModuleNotFoundError, AttributeError, ImportError):
        return no_such_operator


def _convert_input_format(inp):
    assert inp.query.ndim in (4, 5)
    query, key, value = inp.query, inp.key, inp.value

    attn_bias = inp.attn_bias
    if isinstance(
        attn_bias,
        (
            BlockDiagonalPaddedKeysMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            BlockDiagonalGappyKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            BlockDiagonalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        ),
    ):
        assert attn_bias.k_seqinfo.seqstart.device == inp.query.device
        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqused_k = attn_bias.k_seqinfo.seqlen
        assert seqused_k is not None
    elif isinstance(attn_bias, BlockDiagonalMask):
        assert attn_bias.k_seqinfo.seqstart.device == inp.query.device
        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqused_k = None
    else:
        cu_seqlen_k = None
        cu_seqlen_q = None
        seqused_k = None
        max_seqlen_q = None
        max_seqlen_k = None

    if query.ndim == 5:  # GQA
        query, _ = bmghk2bmhk(query, None, handle_rep_heads=True)
        key, _ = bmghk2bmhk(key, None, handle_rep_heads=True)
        value, _ = bmghk2bmhk(value, None, handle_rep_heads=True)

    # For varlen, fold 4D to 3D
    should_fold = cu_seqlen_k is not None and query.ndim == 4
    if should_fold:
        query, _ = bmhk2bhk(query)
        key, _ = bmhk2bhk(key)
        value, _ = bmhk2bhk(value)

    new_inp = Inputs(
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        p=inp.p,
        scale=inp.scale,
        output_dtype=inp.output_dtype,
        is_partial=inp.is_partial,
    )
    return (
        new_inp,
        cu_seqlen_q,
        max_seqlen_q,
        cu_seqlen_k,
        max_seqlen_k,
        seqused_k,
        None,  # page_table (not supported on SM90)
    )


def _is_causal(attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> bool:
    return isinstance(
        attn_bias,
        (
            LowerTriangularMask,
            BlockDiagonalCausalMask,
            LowerTriangularFromBottomRightMask,
            BlockDiagonalCausalFromBottomRightMask,
            LowerTriangularFromBottomRightLocalAttentionMask,
            BlockDiagonalCausalLocalAttentionMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ),
    )


def _is_bottom_right(attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> bool:
    return isinstance(
        attn_bias,
        (
            LowerTriangularFromBottomRightMask,
            BlockDiagonalCausalFromBottomRightMask,
            LocalAttentionFromBottomRightMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            BlockDiagonalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        ),
    )


def _window_size(
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]],
) -> Tuple[int, int]:
    win_left = -1
    win_right = -1
    if isinstance(
        attn_bias,
        (
            BlockDiagonalCausalLocalAttentionMask,
            BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            LowerTriangularFromBottomRightLocalAttentionMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        ),
    ):
        win_left = attn_bias._window_size - 1
    if isinstance(
        attn_bias,
        (
            BlockDiagonalLocalAttentionPaddedKeysMask,
            LocalAttentionFromBottomRightMask,
        ),
    ):
        win_left = attn_bias.window_left
        win_right = attn_bias.window_right
    return (win_left, win_right)


class FwOp(AttentionFwOpBase):
    OPERATOR = _get_operator("_flash_attn_fwd")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.bfloat16, torch.float16}
    SUPPORTED_MAX_K = 128
    SUPPORTED_MIN_K = 64
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalPaddedKeysMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalLocalAttentionPaddedKeysMask,
        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        LocalAttentionFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    VARLEN_LSE_PACKED = True
    SUPPORTS_PARTIAL = False
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (9, 0)
    CUDA_MAXIMUM_COMPUTE_CAPABILITY = (9, 9)
    NAME = "cuteDSLF-hopper"

    _TEST_K: List[int] = [64, 128]

    ERROR_ATOL: Mapping[torch.dtype, float] = _ERROR_ATOL
    ERROR_RTOL: Mapping[torch.dtype, float] = _ERROR_RTOL

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        if d.query.ndim < 4 or d.key.ndim < 4 or d.value.ndim < 4:
            reasons.append("Only supports BMHK or BMGHK")
        return reasons

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in [64, 128] or Kv not in [64, 128]:
            reasons.append(f"Embed dim {K} not supported")
        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        q_shape = inp.query.shape
        deterministic = torch.are_deterministic_algorithms_enabled()
        (
            inp,
            cu_seqlens_q,
            max_seq_len_q,
            cu_seqlens_k,
            max_seq_len_k,
            seqused_k,
            page_table,
        ) = _convert_input_format(inp)

        window_left, window_right = _window_size(inp.attn_bias)

        if inp.query.numel() > 0 and inp.key.numel() > 0:
            out, lse = cls.OPERATOR(
                q=inp.query,
                k=inp.key,
                v=inp.value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                seqlen_kv=seqused_k,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=max_seq_len_k,
                softmax_scale=inp.scale,
                causal=_is_causal(inp.attn_bias),
                window_left=window_left,
                window_right=window_right,
                bottom_right=_is_bottom_right(inp.attn_bias),
                deterministic=deterministic,
            )
        else:
            out = torch.zeros_like(inp.query)
            if cu_seqlens_q is None:
                assert inp.query.ndim == 4
                B, M, H, K = inp.query.shape
                lse_shape = [B, H, M]
            else:
                assert inp.query.ndim == 3
                M, H, K = inp.query.shape
                lse_shape = [1, H, M]
            lse = torch.zeros(*lse_shape, dtype=torch.float, device=out.device)
        out = out.reshape(q_shape)
        if not needs_gradient:
            return out, None
        return out, Context(out=out, lse=lse)


class BwOp(AttentionBwOpBase):
    __doc__ = FwOp.__doc__

    OPERATOR = _get_operator("_flash_attn_bwd")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_MIN_K = FwOp.SUPPORTED_MIN_K
    # SM90 backward does NOT support varlen (cu_seqlens) or local attention (window_size).
    # Only non-varlen causal masks are supported.
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
    )
    SUPPORTS_ATTN_BIAS_GRAD = False
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    VARLEN_LSE_PACKED = True
    SUPPORTS_PARTIAL = False
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (9, 0)
    CUDA_MAXIMUM_COMPUTE_CAPABILITY = (9, 9)
    NAME = "cuteDSLB-hopper"

    ERROR_ATOL: Mapping[torch.dtype, float] = _ERROR_ATOL
    ERROR_RTOL: Mapping[torch.dtype, float] = _ERROR_RTOL

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        if (
            d.query.ndim not in [4, 5]
            or d.key.ndim not in [4, 5]
            or d.value.ndim not in [4, 5]
        ):
            reasons.append("Only supports BMHK and BMGHK formats")
        if _is_causal(d.attn_bias) and d.key.shape[1] > d.query.shape[1]:
            reasons.append("SM90 causal requires Mq >= Mkv")
        return reasons

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in [64, 128]:
            reasons.append(f"Embed dim {K} not supported")
        elif Mkv != 0 and Mq > Mkv:
            reasons.append(f"Only support Mq ({Mq}) <= Mk ({Mkv})")
        elif Mq < 8:
            reasons.append(f"Only support Mq ({Mq}) >= 8")
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        query_ndim = inp.query.ndim
        assert query_ndim in [4, 5]
        dq_shape, dk_shape, dv_shape = inp.query.shape, inp.key.shape, inp.value.shape
        deterministic = torch.are_deterministic_algorithms_enabled()
        (
            inp,
            cu_seqlens_q,
            max_seq_len_q,
            cu_seqlens_k,
            max_seq_len_k,
            _,
            _,
        ) = _convert_input_format(inp)

        is_varlen = cu_seqlens_q is not None
        if query_ndim == 5:
            grad, _ = bmghk2bmhk(grad)
            ctx.out, _ = bmghk2bmhk(ctx.out)

        if is_varlen:
            grad, _ = bmhk2bhk(grad, handle_mqa=False)
            ctx.out, _ = bmhk2bhk(ctx.out, handle_mqa=False)

        if inp.query.numel() and inp.key.numel():
            grads = Gradients(
                *cls.OPERATOR(
                    dout=grad,
                    q=inp.query,
                    k=inp.key,
                    v=inp.value,
                    out=ctx.out,
                    softmax_lse=ctx.lse,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seq_len_q=max_seq_len_q,
                    max_seq_len_k=max_seq_len_k,
                    causal=_is_causal(inp.attn_bias),
                    bottom_right=_is_bottom_right(inp.attn_bias),
                    deterministic=deterministic,
                )
            )
        else:
            grads = Gradients(
                dq=torch.zeros_like(inp.query),
                dk=torch.zeros_like(inp.key),
                dv=torch.zeros_like(inp.value),
            )

        grads.dq = grads.dq.reshape(dq_shape)
        grads.dk = grads.dk.reshape(dk_shape)
        grads.dv = grads.dv.reshape(dv_shape)
        return grads
