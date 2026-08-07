# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
from mslk.flydsl.common import require_flydsl

from .attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from .common import AttentionFwOpBase, Context, Inputs
from .utils.op_common import get_operator, register_operator


def _flydsl_decode_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_positions: Optional[torch.Tensor],
    scale: float,
    use_fp8_kv: bool = False,
) -> torch.Tensor:
    from .flydsl.layout_utils import canonicalize_qkv_5d, normalize_seq_positions
    from .flydsl.pa_decode_dense import pa_decode_launch

    q5, k5, v5 = canonicalize_qkv_5d(query, key, value)
    B = q5.shape[0]
    KV_MAX = k5.shape[1]
    seq = normalize_seq_positions(seq_positions, B, KV_MAX, q5.device)

    if use_fp8_kv:
        # Per-call opt-in (inp.quantize_kv_to_fp8): quantize dense f16/bf16 KV to
        # native fp8 and run the paged fp8 decode.  Lossy + per-call quant cost;
        # gfx950 only (MQA + GQA; the adapter pads short contexts).
        from .flydsl.pa_decode_fp8_dispatch import is_fp8_paged_decode_available

        if is_fp8_paged_decode_available():
            from .flydsl.fp8_paged_adapter import fp8_paged_decode_from_dense

            return fp8_paged_decode_from_dense(q5, k5, v5, seq, scale)
    # split_k=0 -> the kernel's auto split-K heuristic (auto_split_k_hp), which fills
    # the GPU with enough KV partitions to hide memory latency.  The previous split_k=1
    # forced a single partition (no parallelism), leaving the single-step decoder ~1.3x
    # slower than CK; auto split-K brings it in line with the split-K decode op.  The
    # kernel combines the partitions internally and returns a single output tensor.
    return pa_decode_launch(q5, k5, v5, seq, scale, split_k=0)


@register_operator
class FwOp(AttentionFwOpBase):
    """FlyDSL dense decode op (gfx942/gfx950).

    FlyDSL is the sole backend: the CK operator path has been removed.  Requires
    FlyDSL (raises via require_flydsl on an unsupported arch).  Supports f16 / bf16
    / f32 KV in a dense padded layout with GQA/MQA.  Keeps the xformers op name and
    API so existing callers are unchanged.
    """

    OPERATOR = get_operator("xformers", "efficient_attention_forward_decoder_ck")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    # pyrefly: ignore [bad-override-mutable-attribute]
    SUPPORTED_MAX_K: int = 256
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "flydsl_decoderF"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:  # noqa: C901
        reasons = super(FwOp, cls).not_supported_reasons(d)

        attn_bias = d.attn_bias
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
            if d.query.shape[0] != 1:
                reasons.append(
                    f"One formal batch element expected; got {d.query.shape[0]}"
                )

            if d.query.shape[-1] > cls.SUPPORTED_MAX_K:
                reasons.append(
                    f"Got head_dim={d.query.shape[-1]}; only head_dim<={cls.SUPPORTED_MAX_K} is supported for now."
                )

            threads_per_warp = 64  # TODO: ideally query the platform here
            required_alignment = 0
            head_dim = d.query.shape[-1]
            for vec_size in (4, 2, 1):
                if head_dim <= vec_size * threads_per_warp:
                    required_alignment = vec_size

            if not required_alignment:
                reasons.append(f"Got head_dim={head_dim} which is too large")

            if head_dim % required_alignment != 0:
                reasons.append(
                    f"Got head_dim={head_dim}; it needs to be divisible by {required_alignment}"
                )

            if d.key.stride(-1) != 1:
                reasons.append("expect keys to have last dim contiguous")

            if d.value.stride(-1) != 1:
                reasons.append("expect values to have last dim contiguous")

            q_starts = attn_bias.q_seqinfo.seqstart_py
            padding = attn_bias.k_seqinfo.padding
            bsz = d.key.shape[1] // padding
            num_queries = d.query.shape[1] // bsz

            if q_starts != list(range(0, 1 + bsz, num_queries)):
                reasons.append("expect to have same num_queries in each batch")
            if bsz != len(q_starts) - 1:
                reasons.append("empty lanes not supported yet")

            if attn_bias.k_seqinfo.padding > 8192:
                reasons.append("key padding exceeds 8192")

        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if needs_gradient:
            raise NotImplementedError("backward pass is not supported")
        attn_bias = inp.attn_bias
        q, k, v = inp.get_qkv_in_bmghk()
        if attn_bias is not None:
            assert isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
            attn_bias.k_seqinfo.to(k.device)
            attn_bias.q_seqinfo.to(q.device)
            padding = attn_bias.k_seqinfo.padding
            seq_positions_gpu = attn_bias.k_seqinfo.seqlen
        else:
            padding = k.shape[1]
            seq_positions_gpu = None

        if attn_bias is not None:
            # key: (1, B * padding, G, 1 if multiquery else Hkv, D)
            # value: like key
            # query: (1, B * q_seqlen, G, Hq, D)
            multiquery = k.stride(3) == 0
            if multiquery:
                key = k[0, :, :, :1].unflatten(0, (-1, padding))
                value = v[0, :, :, :1].unflatten(0, (-1, padding))
            else:
                key = k[0].unflatten(0, (-1, padding))
                value = v[0].unflatten(0, (-1, padding))
            query = q[0].unflatten(0, (key.shape[0], -1))
        else:
            # key: (B, padding, G, 1 if multiquery else Hkv, D)
            # value: like key
            # query: (B, q_seqlen, G, Hq, D)
            key = k
            query = q
            value = v

        if inp.scale is not None:
            qk_scale = inp.scale
        else:
            qk_scale = torch.rsqrt(
                torch.tensor(key.shape[-1], dtype=torch.float32)
            ).item()

        # FlyDSL is the sole decode backend (the CK operator path was removed): it
        # covers f16/bf16 KV across gfx942 + gfx950 and outperforms the old CK
        # kernel (which used no matrix cores) on every measured shape.  The op name
        # and API are unchanged, so callers are unaffected.
        require_flydsl()
        out = _flydsl_decode_forward(
            query=query,
            key=key,
            value=value,
            seq_positions=seq_positions_gpu,
            scale=qk_scale,
            use_fp8_kv=getattr(inp, "quantize_kv_to_fp8", False),
        )
        return out, None
