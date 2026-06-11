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


def _decode_query_seqlen(inp: Inputs) -> int:
    q = inp.query
    if q.ndim == 5 and q.shape[0] == 1:
        attn_bias = inp.attn_bias
        if isinstance(
            attn_bias,
            (
                BlockDiagonalCausalWithOffsetPaddedKeysMask,
                PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
            ),
        ):
            bsz = len(attn_bias.k_seqinfo.seqlen)
            if bsz > 0 and q.shape[1] % bsz == 0:
                return q.shape[1] // bsz
    if q.ndim in (4, 5):
        return q.shape[1]
    return q.shape[1] if q.ndim >= 2 else 0


def _is_decode_inputs(inp: Inputs) -> bool:
    return _decode_query_seqlen(inp) <= 1


@register_operator
class FwOp(AttentionFwOpBase):
    """Triton decode-phase kernel for Grouped Query Attention (GQA), targeting ROCm (and CUDA).

    Designed exclusively for single-token (Mq <= 1) decode steps, where each
    query attends over the full KV context accumulated during prefill and prior
    decode steps.

    **GQA layout and launch grid.**
    Inputs are expected in BMGHK layout (batch, seq, kv_groups, heads_per_group,
    head_dim).  The kernel launches a 2-D grid of shape ``(B, G)`` — one program
    per (batch element, KV-head group) — and processes all ``Hq`` query heads
    that share a single KV head inside that program.  This avoids the
    expand-and-head-swap trick used by ``triton_splitk.FwOp`` and keeps KV
    traffic minimal.

    **Padding and paged KV cache.**
    Two attention-bias types are supported:

    * ``BlockDiagonalCausalWithOffsetPaddedKeysMask`` — contiguous KV cache with
      per-sequence padding.  The query tensor is expected with a single formal
      batch dimension (``query.shape[0] == 1``); internally the operator reshapes
      ``(1, B*Mq, G, Hq, D)`` → ``(B, Mq, G, Hq, D)`` and unfolds the key/value
      blocks accordingly.  Key padding must not exceed 8 192 tokens.

    * ``PagedBlockDiagonalCausalWithOffsetPaddedKeysMask`` — paged KV cache.
      ``block_tables`` of shape ``[batch_size, max_num_pages]`` maps each batch
      element to its physical pages; K/V have shape
      ``[1, max_num_pages * page_size, G, H_kv, D]``.  Each Triton tile
      (``BLOCK_N``) must fit within one page (enforced by a static assert).

    **Shape constraints.**
    * Head dimension ``D`` must be one of ``{64, 128, 256}``.
    * At most ``SUPPORTED_MAX_Q_HEADS_PER_GROUP`` (32) query heads per KV group.
    * Query, key, and value last-dim strides must be stride-1 and 8-element
      aligned.
    * No backward pass; no dropout; no int4/fp8 quantisation (use
      ``triton_splitk.FwOp`` for those).
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
            Mq = _decode_query_seqlen(d)
            _, _, G, Hq, Kq = q.shape
            if Mq > 1:
                reasons.append(f"decode kernel requires Mq<=1, got Mq={Mq}")
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
        if is_paged:
            block_tables = attn_bias.block_tables
            page_size = attn_bias.page_size

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
        ) = cls._prepare_tensors(inp)

        out = torch.empty_like(q)
        kernel = cls.OPERATOR()
        grid = (B, G)
        # HQ_BLOCK: smallest power-of-2 >= max(16, Hq).
        # Minimum 16 satisfies tl.dot's MFMA M-dimension requirement on ROCm.
        HQ_BLOCK = 32 if Hq > 16 else 16

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
            **(
                _strides(block_tables, "bt_batch", "bt_page")
                if is_paged
                else {"stride_bt_batch": 0, "stride_bt_page": 0}
            ),
            Z=B,
            G=G,
            Hq=Hq,
            Mq=Mq,
            N_CTX_K=N_CTX_K,
            BLOCK_DMODEL=D,
            USE_PAGED=is_paged,
            PAGE_SIZE=page_size,
            HQ_BLOCK=HQ_BLOCK,
        )

        # Restore flattened batch layout expected by callers/tests.
        if inp.query.shape[0] == 1:
            out = out.view(1, B * Mq, G, Hq, D)
        return out, None
