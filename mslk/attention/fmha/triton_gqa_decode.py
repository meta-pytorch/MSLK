# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import sys
from typing import Any, Iterable, List, Optional, Set, Tuple

import torch

from ._triton.available import is_triton_available
from .attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
)
from .common import AttentionFwOpBase, check_lastdim_alignment_stride1, Context, Inputs
from .utils.op_common import register_operator

if is_triton_available():
    from ._triton.gqa_decode_kernels import get_gqa_decode_kernel
else:
    get_gqa_decode_kernel = None  # type: ignore


def _strides(x: Optional[torch.Tensor], *stride_names: str):
    if x is None:
        return {f"stride_{name}": None for name in stride_names}
    assert x.ndim == len(stride_names)
    return {f"stride_{name}": s for name, s in zip(stride_names, x.stride())}


def _is_decode_inputs(inp: Inputs) -> bool:
    q = inp.query
    if q.ndim == 5:
        return q.shape[1] <= 1
    if q.ndim == 4:
        return q.shape[1] <= 1
    return False


@register_operator
class FwOp(AttentionFwOpBase):
    """
    Triton decode kernel for Grouped Query Attention on ROCm (and CUDA).

    Uses launch grid (batch, num_kv_head_groups) with native GQA head blocking
    instead of the expand + head-swap path in triton_splitk.
    """

    OPERATOR = get_gqa_decode_kernel
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K: int = 256
    SUPPORTED_MAX_Q_HEADS_PER_GROUP: int = 32
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    SUPPORTS_PARTIAL = False
    NAME = "triton_gqa_decodeF"

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if Mq > 1:
            reasons.append(f"decode kernel requires Mq<=1, got Mq={Mq}")
        if K not in {64, 128, 256}:
            reasons.append(f"head dim {K} not supported (supported: 64, 128, 256)")
        return reasons

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:  # noqa: C901
        reasons = super(FwOp, cls).not_supported_reasons(d)
        if (sys.version_info.major, sys.version_info.minor) < (3, 9):
            reasons.append("triton_gqa_decode requires python 3.9 or above")
        if cls.OPERATOR is None:
            reasons.append("triton is not available")
        if not _is_decode_inputs(d):
            reasons.append("only single-token decode queries are supported")
        if d.p != 0.0:
            reasons.append("dropout not supported")

        q, k, v = d.get_qkv_in_bmghk()
        if q.ndim != 5:
            reasons.append("BMGHK (5D) query layout required")
        else:
            _, Mq, G, Hq, Kq = q.shape
            if Hq > cls.SUPPORTED_MAX_Q_HEADS_PER_GROUP:
                reasons.append(
                    f"at most {cls.SUPPORTED_MAX_Q_HEADS_PER_GROUP} query heads per "
                    f"kv group, got {Hq}"
                )
            if k.shape[2] != G or v.shape[2] != G:
                reasons.append("query and key/value group counts must match")

        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
        check_lastdim_alignment_stride1(reasons, "value", d.value, 8)

        attn_bias = d.attn_bias
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
            if d.query.shape[0] != 1:
                reasons.append(
                    f"one formal batch dim expected on query; got {d.query.shape[0]}"
                )
            padding = attn_bias.k_seqinfo.padding
            if padding > 8192:
                reasons.append("key padding exceeds 8192")
        elif isinstance(attn_bias, PagedBlockDiagonalCausalWithOffsetPaddedKeysMask):
            if d.query.shape[0] != 1 and d.query.ndim == 5:
                if d.query.shape[0] != 1:
                    reasons.append("paged decode expects query batch dim 1")
        elif attn_bias is not None:
            reasons.append(f"unsupported attn_bias type {type(attn_bias)}")

        return reasons

    @classmethod
    def _prepare_tensors(
        cls,
        inp: Inputs,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        int,
        int,
        int,
        int,
        int,
        int,
        bool,
        int,
        int,
    ]:
        attn_bias = inp.attn_bias
        assert attn_bias is not None
        q, k, v = inp.get_qkv_in_bmghk()
        is_paged = isinstance(
            attn_bias, PagedBlockDiagonalCausalWithOffsetPaddedKeysMask
        )

        attn_bias.k_seqinfo.to(k.device)
        attn_bias.q_seqinfo.to(q.device)
        seq_len = attn_bias.k_seqinfo.seqlen
        padding = attn_bias.k_seqinfo.padding
        bsz = len(seq_len)

        key = k
        value = v
        # Fold (1, B * Mq, G, H, D) -> (B, Mq, G, H, D) like ck_decoder.
        if q.shape[0] == 1:
            Mq = q.shape[1] // bsz
            q = q[0].unflatten(0, (bsz, Mq))
            if not is_paged:
                multiquery = k.stride(3) == 0 and k.shape[3] == 1
                if multiquery:
                    key = k[0, :, :, :1].unflatten(0, (-1, padding))
                    value = v[0, :, :, :1].unflatten(0, (-1, padding))
                else:
                    key = k[0].unflatten(0, (-1, padding))
                    value = v[0].unflatten(0, (-1, padding))
        else:
            Mq = q.shape[1]

        if not is_paged:
            # Use compact KV when user passed expanded BMGHK (stride 0 on head dim).
            if key.stride(3) == 0 and key.shape[3] > 1:
                key = key[:, :, :, :1, :].contiguous()
                value = value[:, :, :, :1, :].contiguous()

        block_tables = None
        page_size = 0
        kv_blocks_per_row = 0
        if is_paged:
            block_tables = attn_bias.block_tables
            page_size = attn_bias.page_size
            kv_blocks_per_row = block_tables.shape[1]

        B, Mq, G, Hq, D = q.shape
        N_CTX_K = key.shape[1]
        return (
            q,
            key,
            value,
            seq_len,
            block_tables,
            B,
            G,
            Hq,
            Mq,
            N_CTX_K,
            D,
            is_paged,
            page_size,
            kv_blocks_per_row,
        )

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if needs_gradient:
            raise NotImplementedError("backward pass is not supported")

        if inp.query.numel() == 0 or inp.key.numel() == 0:
            return torch.zeros_like(inp.query), None

        (
            q,
            key,
            value,
            seq_len,
            block_tables,
            B,
            G,
            Hq,
            Mq,
            N_CTX_K,
            D,
            is_paged,
            page_size,
            kv_blocks_per_row,
        ) = cls._prepare_tensors(inp)

        out = torch.empty_like(q)
        kernel = cls.OPERATOR()
        grid = (B, G)

        kernel[grid](
            q,
            key,
            value,
            out,
            seq_len,
            block_tables,
            inp.scale_float,
            **_strides(q, "qz", "qm", "qg", "qh", "qk"),
            **_strides(key, "kz", "kn", "kg", "kh", "kk"),
            **_strides(value, "vz", "vn", "vg", "vh", "vk"),
            **_strides(out, "oz", "om", "og", "oh", "ok"),
            **_strides(block_tables, "bt_batch", "bt_page"),
            Z=B,
            G=G,
            Hq=Hq,
            Mq=Mq,
            N_CTX_K=N_CTX_K,
            BLOCK_DMODEL=D,
            USE_PAGED=is_paged,
            PAGE_SIZE=page_size,
            KV_BLOCKS_PER_ROW=kv_blocks_per_row,
        )

        # Restore flattened batch layout expected by callers/tests.
        if inp.query.shape[0] == 1:
            out = out.view(1, B * Mq, G, Hq, D)
        return out, None
