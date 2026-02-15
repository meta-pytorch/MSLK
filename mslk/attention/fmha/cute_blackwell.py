# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
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
    PagedBlockDiagonalCausalLocalPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    bmghk2bmhk,
    bmhk2bhk,
    Context,
    Gradients,
    Inputs,
    InputsFp8,
)

_ERROR_ATOL: Mapping[torch.dtype, float] = {
    torch.float: 3e-4,
    # (FIXME: Temporary Solution) Decided to use the tolerance same as FA4's
    # varlen test
    # (https://github.com/Dao-AILab/flash-attention/blob/main/tests/cute/test_flash_attn_varlen.py#L70-L71)
    # temporarily.
    torch.half: 3e-2,
    torch.bfloat16: 3e-2,
}

_ERROR_RTOL: Mapping[torch.dtype, float] = {
    torch.float: 2e-5,
    # (FIXME: Temporary Solution) Decided to use the tolerance same as FA4's
    # varlen test
    # (https://github.com/Dao-AILab/flash-attention/blob/main/tests/cute/test_flash_attn_varlen.py#L70-L71)
    # temporarily.
    torch.half: 3e-2,
    torch.bfloat16: 3e-2,
}


def _get_operator(name: str):
    def no_such_operator(*args, **kwargs):
        raise RuntimeError(
            f"No such operator mslk.attention.flash_attn.interface.{name}"
        )

    try:
        # type: ignore  # pyre-ignore
        import mslk.attention.flash_attn.interface as flash_attn

        return getattr(flash_attn, name)  # type: ignore  # pyre-ignore
    except (RuntimeError, ModuleNotFoundError):
        return no_such_operator


def _convert_input_format(
    inp: Inputs,
) -> Tuple[
    Inputs,
    Optional[torch.Tensor],
    Optional[int],
    Optional[torch.Tensor],
    Optional[int],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    assert inp.query.ndim in (4, 5)
    query, key, value = inp.query, inp.key, inp.value

    attn_bias = inp.attn_bias
    page_table = None
    if isinstance(attn_bias, (PagedBlockDiagonalPaddedKeysMask)):
        # Paged attention: use page_table + seqused_k, NOT cu_seqlens_k
        # The kernel assertion: "page_table is not supported with cu_seqlens_k"
        assert attn_bias.k_seqinfo.seqstart.device == inp.query.device
        cu_seqlen_k = None  # Don't pass cu_seqlens_k for paged attention
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqused_k = attn_bias.k_seqinfo.seqlen
        assert seqused_k is not None
        page_table = attn_bias.block_tables
    elif isinstance(attn_bias, BlockDiagonalMask):
        assert attn_bias.k_seqinfo.seqstart.device == inp.query.device
        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqused_k = None
    elif isinstance(
        attn_bias,
        (
            BlockDiagonalPaddedKeysMask,
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            BlockDiagonalGappyKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            BlockDiagonalLocalAttentionPaddedKeysMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
            PagedBlockDiagonalGappyKeysMask,
        ),
    ):
        assert attn_bias.k_seqinfo.seqstart.device == inp.query.device
        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        # All these mask types inherit from classes that have seqlen attribute
        seqused_k = attn_bias.k_seqinfo.seqlen
        assert seqused_k is not None
    else:
        cu_seqlen_k = None
        cu_seqlen_q = None
        seqused_k = None
        max_seqlen_q = None
        max_seqlen_k = None

    key_scale = (
        inp.k_fp8_scale_shift if hasattr(inp, "k_fp8_scale_shift") else None
    )  # pyre-fixme[16]
    value_scale = (
        inp.v_fp8_scale_shift if hasattr(inp, "v_fp8_scale_shift") else None
    )  # pyre-fixme[16]

    if query.ndim == 5:  # GQA
        query, _ = bmghk2bmhk(query, None, handle_rep_heads=True)
        key, key_scale = bmghk2bmhk(key, key_scale, handle_rep_heads=True)
        value, value_scale = bmghk2bmhk(value, value_scale, handle_rep_heads=True)

    # For paged attention or varlen, fold 4D to 3D
    should_fold = (
        cu_seqlen_k is not None or page_table is not None
    ) and query.ndim == 4
    if should_fold:
        # Fold to 3D when using varlen or paged attention
        query, _ = bmhk2bhk(query)
        key, key_scale = bmhk2bhk(key, x_scale=key_scale)
        value, value_scale = bmhk2bhk(value, x_scale=value_scale)
    # For paged attention, K/V have shape (num_pages, page_size, heads, dim) - view to that shape
    if isinstance(
        attn_bias,
        (PagedBlockDiagonalGappyKeysMask, PagedBlockDiagonalPaddedKeysMask),
    ):
        num_pages = value.shape[0] // attn_bias.page_size
        key = key.view(num_pages, attn_bias.page_size, *key.shape[1:])
        value = value.view(num_pages, attn_bias.page_size, *value.shape[1:])

    if isinstance(inp, InputsFp8):
        new_inp = InputsFp8(
            query=query,
            key=key.view(torch.float8_e4m3fn),
            value=value.view(torch.float8_e4m3fn),
            attn_bias=attn_bias,
            p=inp.p,
            scale=inp.scale,
            output_dtype=inp.output_dtype,
            is_partial=inp.is_partial,
            deterministic=inp.deterministic,
            # We want k_fp8_scale_shift and v_fp8_scale_shift to be in the same shape as key and value[:-1]
            k_fp8_scale_shift=key_scale.view(key.shape[:-1]),
            v_fp8_scale_shift=value_scale.view(value.shape[:-1]),
            use_fp32_scales=inp.use_fp32_scales,
        )
    else:
        new_inp = Inputs(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=inp.p,
            scale=inp.scale,
            output_dtype=inp.output_dtype,
            is_partial=inp.is_partial,
            deterministic=inp.deterministic,
        )
    return (
        new_inp,
        cu_seqlen_q,
        max_seqlen_q,
        cu_seqlen_k,
        max_seqlen_k,
        seqused_k,
        page_table,
    )


def _is_seqlen_q_le_seqlen_k(
    cu_seqlens_q_py: List[int], cu_seqlens_k_py: List[int]
) -> bool:
    if len(cu_seqlens_q_py) < 2 or len(cu_seqlens_k_py) < 2:
        # The seqlens q and k info does not exist on CPU
        return True
    cu_seqlens_q = torch.as_tensor(cu_seqlens_q_py, dtype=torch.int, device="cpu")
    cu_seqlens_k = torch.as_tensor(cu_seqlens_k_py, dtype=torch.int, device="cpu")
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    return torch.all(seqlens_k >= seqlens_q).item()  # pyre-fixme[7]


def _get_paged_block_tables(
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]],
) -> Optional[torch.Tensor]:
    if isinstance(
        attn_bias,
        (
            PagedBlockDiagonalCausalLocalPaddedKeysMask,
            PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
            PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
            PagedBlockDiagonalGappyKeysMask,
            PagedBlockDiagonalPaddedKeysMask,
        ),
    ):
        return attn_bias.block_tables
    return None


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
            PagedBlockDiagonalCausalLocalPaddedKeysMask,
            PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
            PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
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
            PagedBlockDiagonalCausalLocalPaddedKeysMask,
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
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (10, 0)
    NAME = "cuteDSLF-blackwell"

    _TEST_K: List[int] = [64, 128]

    ERROR_ATOL: Mapping[torch.dtype, float] = _ERROR_ATOL
    ERROR_RTOL: Mapping[torch.dtype, float] = _ERROR_RTOL

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        if isinstance(d.attn_bias, BlockDiagonalCausalMask):
            (
                _,
                cu_seqlens_q,
                _,
                cu_seqlens_k,
                _,
                _,
                _,
            ) = _convert_input_format(d)
            if not _is_seqlen_q_le_seqlen_k(
                d.attn_bias.q_seqinfo.seqstart_py,  # pyre-fixme[16]
                d.attn_bias.k_seqinfo.seqstart_py,  # pyre-fixme[16]
            ):
                reasons.append("seqlens_k must be >= seqlens_q")

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
        elif Mkv != 0 and Mq > Mkv:
            reasons.append(f"Only support Mq ({Mq}) <= Mk ({Mkv})")
        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        q_shape = inp.query.shape
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
                page_table=_get_paged_block_tables(inp.attn_bias),
                num_splits=inp.num_splits,
                deterministic=inp.deterministic,
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


class FwOpDecode(AttentionFwOpBase):
    """Cute Blackwell decode kernel optimized for inference with sequence length 1.

    This operator is specifically designed for the decode phase of autoregressive generation
    where query length is 1.
    """

    OPERATOR = _get_operator("_flash_attn_decode")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.bfloat16, torch.float16}
    SUPPORTED_MAX_K = 128
    SUPPORTED_MIN_K = 64
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalLocalAttentionPaddedKeysMask,
        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalPaddedKeysMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalPaddedKeysMask,
        PagedBlockDiagonalCausalLocalPaddedKeysMask,
        PagedBlockDiagonalGappyKeysMask,
        PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    VARLEN_LSE_PACKED = True
    SUPPORTS_PARTIAL = False
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (10, 0)
    NAME = "cuteDSLF-blackwell-decode"

    _TEST_K: List[int] = [64, 128]

    ERROR_ATOL: Mapping[torch.dtype, float] = _ERROR_ATOL
    ERROR_RTOL: Mapping[torch.dtype, float] = _ERROR_RTOL

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOpDecode, cls).not_supported_reasons(d)
        q_shape = d.query.shape
        if q_shape[-2] > 16:
            reasons.append(f"Max qHeads ({q_shape[-2]}) per KV head is > 16")
        if (
            d.attn_bias is not None
            and hasattr(d.attn_bias, "q_seqinfo")
            and d.attn_bias.q_seqinfo.max_seqlen is not None  # pyre-fixme[16]
            and d.attn_bias.q_seqinfo.max_seqlen != 1  # pyre-fixme[16]
        ):
            reasons.append(
                f"Max Q seq length ({d.attn_bias.q_seqinfo.max_seqlen}) must be "
                "1 for decode kernel"
            )
        return reasons

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in [64, 128]:
            reasons.append(f"Embed dim {K} not supported")
        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        q_shape = inp.query.shape
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

        is_fp8 = isinstance(inp, InputsFp8)
        if is_fp8:
            assert (
                inp.k_fp8_scale_shift is not None
                and inp.v_fp8_scale_shift is not None  # pyre-fixme[16]
            ), (
                "k_fp8_scale_shift and v_fp8_scale_shift must be provided when inp is InputsFp8"
            )

        if inp.query.numel() > 0 and inp.key.numel() > 0:
            out, _ = cls.OPERATOR(
                q=inp.query,
                k=inp.key,
                v=inp.value,
                k_scale_shift=inp.k_fp8_scale_shift
                if is_fp8
                else None,  # pyre-fixme[16]
                v_scale_shift=inp.v_fp8_scale_shift
                if is_fp8
                else None,  # pyre-fixme[16]
                cu_seqlens_q=cu_seqlens_q,  # not used
                cu_seqlens_k=cu_seqlens_k,  # not used
                seqlen_kv=seqused_k,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=max_seq_len_k,
                softmax_scale=inp.scale,
                causal=_is_causal(inp.attn_bias),
                window_left=window_left,
                window_right=window_right,
                bottom_right=_is_bottom_right(inp.attn_bias),  # not used
                page_table=_get_paged_block_tables(inp.attn_bias),
                num_splits=inp.num_splits,
                deterministic=inp.deterministic,
            )
        else:
            out = torch.zeros_like(inp.query)
            if cu_seqlens_q is None:
                assert inp.query.ndim == 4
                B, M, H, K = inp.query.shape
            else:
                assert inp.query.ndim == 3
                M, H, K = inp.query.shape
        out = out.reshape(q_shape)
        assert not needs_gradient, "FwOpDecode does not support gradient computation"
        return out, None


class BwOp(AttentionBwOpBase):
    __doc__ = FwOp.__doc__

    OPERATOR = _get_operator("_flash_attn_bwd")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_MIN_K = FwOp.SUPPORTED_MIN_K
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        LocalAttentionFromBottomRightMask,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    )
    SUPPORTS_ATTN_BIAS_GRAD = False
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    VARLEN_LSE_PACKED = True
    SUPPORTS_PARTIAL = False
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (10, 0)
    NAME = "cuteDSLB-blackwell"

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

            return reasons

        (
            _,
            cu_seqlens_q,
            _,
            cu_seqlens_k,
            _,
            _,
            _,
        ) = _convert_input_format(d)

        if isinstance(d.attn_bias, BlockDiagonalCausalMask):
            if not _is_seqlen_q_le_seqlen_k(
                d.attn_bias.q_seqinfo.seqstart_py,  # pyre-fixme[16]
                d.attn_bias.k_seqinfo.seqstart_py,  # pyre-fixme[16]
            ):
                reasons.append("seqlens_k must be >= seqlens_q")

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
        (
            inp,
            cu_seqlens_q,
            max_seq_len_q,
            cu_seqlens_k,
            max_seq_len_k,
            _,
            _,
        ) = _convert_input_format(inp)

        window_left, window_right = _window_size(inp.attn_bias)

        is_varlen = cu_seqlens_q is not None
        if query_ndim == 5:
            grad, _ = bmghk2bmhk(grad)
            ctx.out, _ = bmghk2bmhk(ctx.out)

        if is_varlen:
            # We cannot use bmhk2bhk for bwd,
            # grad / out should have the same shape as query even if head was copied.
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
                    window_left=window_left,
                    window_right=window_right,
                    bottom_right=_is_bottom_right(inp.attn_bias),
                    deterministic=inp.deterministic,
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
