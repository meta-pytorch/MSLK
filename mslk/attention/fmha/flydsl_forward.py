# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

"""FlyDSL forward FMHA operator.

Wraps ``mslk.attention.flydsl.flydsl_flash_attn_func`` behind ``AttentionFwOpBase``
as a drop-in for ``ck.FwOp``, bf16/f16, head_dim in [1, 256] (non-native dims are
zero-padded to {64, 128, 256}). Supports causal/non-causal self-attention, additive
tensor bias, sliding-window masks, varlen, padded/gappy keys, and paged KV
(page_size a multiple of 64; native paged D 64/128, paged-gappy any built D); all 18
attn_bias types, BMHK and BMGHK. Dense-path cross-attention is rejected via
``not_supported_reasons``.
"""

import math
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple

import torch

from .attn_bias import (
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalMask,
    BlockDiagonalPaddedKeysMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)
from .common import AttentionFwOpBase, Context, Inputs
from .utils.op_common import register_operator

# Causal mask types (LowerTriangularMaskWithTensorBias is causal + tensor bias).
_CAUSAL_BIAS_TYPES = (
    LowerTriangularMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
    LowerTriangularMaskWithTensorBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
)

# Sliding-window (local) mask types carrying a _window_size.
_WINDOW_BIAS_TYPES = (
    LowerTriangularFromBottomRightLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
)

# Varlen-local (windowed) top-left type; drives causal_top_left.
_VARLEN_WINDOW_TOPLEFT_TYPES = (BlockDiagonalCausalLocalAttentionMask,)

# Head dims the kernel builds natively; other dims in [1, 256] are zero-padded up to
# the next value here (padded Q/K add 0 to QK; padded V columns are sliced off O).
_SUPPORTED_HEAD_DIMS = (64, 128, 256)


def _pad_head_dim(d: int) -> int:
    """Smallest natively-buildable head dim >= d (one of _SUPPORTED_HEAD_DIMS)."""
    for hd in _SUPPORTED_HEAD_DIMS:
        if d <= hd:
            return hd
    return d


def _pack_varlen_lse(lse: torch.Tensor, q_seqinfo: Any) -> torch.Tensor:
    """Padded per-batch LSE [B, H, max_seqlen_q] -> packed [1, H, total_q].

    Mirrors ck.FwOp's packed layout so ck.BwOp (VARLEN_LSE_PACKED=True) reads the
    right rows. Padded tail rows (q_row >= seqlen_q) are dropped per batch.
    """
    parts = [
        sl[:, : end - start]
        for sl, (start, end) in zip(lse.unbind(0), q_seqinfo.intervals())
    ]
    return torch.cat(parts, dim=1).unsqueeze(0)


# Varlen (packed cu_seqlens) mask types; q/k/v arrive as [1, total, H, D].
_VARLEN_BIAS_TYPES = (
    BlockDiagonalMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
)

# Paged-KV mask types, driven through the kernel's vLLM paged path (native D 64/128;
# any page_size that is a multiple of 64, remapped to 64-row sub-pages).
_PAGED_BIAS_TYPES = (
    PagedBlockDiagonalPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
)
_PAGED_HEAD_DIMS = (64, 128)
# The generic paged kernel tiles KV at PAGE_SIZE == BLOCK_N == 64; paged-gappy
# reshapes any physical page_size to 64-row sub-pages (see _apply_bmhk).
_KERNEL_PAGE_SIZE = 64

# Paged-gappy: handled by a host gather through block_tables into a tight packing on
# the varlen path (page-size-agnostic, no kernel paged path). Non-causal.
_PAGED_GAPPY_TYPES = (PagedBlockDiagonalGappyKeysMask,)

# Padded/gappy KV: blocks at k_seqinfo.seqstart with valid length k_seqinfo.seqlen,
# gathered into a tight packing on the varlen path. WithOffset causal = bottom-right.
_PADDED_GAPPY_BIAS_TYPES = (
    BlockDiagonalPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
)


def _is_flydsl_available() -> bool:
    try:
        from mslk.flydsl.common import is_flydsl_available

        return is_flydsl_available()
    except Exception:
        return False


@register_operator
class FwOp(AttentionFwOpBase):
    """FlyDSL flash-attention forward (gfx942 generic / gfx950 dualwave)."""

    # Python callable rather than a torch.ops handle (cf. triton_splitk.FwOp).
    OPERATOR = staticmethod(_is_flydsl_available)
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    # Any dim in [1, 256]; non-native dims are padded (see _pad_head_dim).
    SUPPORTED_MAX_K = 256
    SUPPORTED_MIN_K = 1

    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularMaskWithTensorBias,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        PagedBlockDiagonalPaddedKeysMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalGappyKeysMask,
    )

    # Dropout (dense): reuses CK's philox mask and records [seed, offset] in
    # ctx.rng_state so ck.BwOp reproduces it. See _apply_bmhk.
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    SUPPORTS_PARTIAL = False
    # The kernel writes padded per-batch LSE [B, H, max_seqlen_q]; _apply_bmhk
    # repacks varlen LSE to [1, H, total_q] so this matches ck.BwOp (True).
    VARLEN_LSE_PACKED = True
    NAME = "flydslF"

    # Match ck.FwOp's tolerances (same MFMA reduced-precision accumulation).
    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 3e-4,
        torch.half: 6e-3,
        torch.bfloat16: 2.8e-2,
    }
    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 2e-5,
        torch.half: 3e-3,
        torch.bfloat16: 2e-2,
    }

    # 96 is non-native, to exercise head-dim padding.
    _TEST_K: List[int] = [64, 96, 128, 256]

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super().not_supported_reasons(d)

        if not _is_flydsl_available():
            reasons.append("FlyDSL unavailable on this device/arch")

        K = d.query.shape[-1]
        Kv = d.value.shape[-1]
        if not (1 <= K <= 256):
            reasons.append(f"query head_dim={K} out of range [1, 256]")
        if Kv != K:
            reasons.append("value head_dim must equal query head_dim")

        # Paged KV kernel fixes PAGE_SIZE 64 and D in {64, 128}; a larger physical
        # page_size (multiple of 64) is remapped to 64-row sub-pages in _apply_bmhk.
        if isinstance(d.attn_bias, _PAGED_BIAS_TYPES):
            if d.attn_bias.page_size % _KERNEL_PAGE_SIZE != 0:
                reasons.append(
                    f"paged page_size={d.attn_bias.page_size} must be a "
                    f"multiple of {_KERNEL_PAGE_SIZE}"
                )
            if K not in _PAGED_HEAD_DIMS:
                reasons.append(
                    f"paged head_dim={K} unsupported (kernel paged: {_PAGED_HEAD_DIMS})"
                )
        # Paged-gappy uses the generic paged kernel (per-row page lookup). D is
        # padded to {64,128,256}; page_size must be a multiple of the kernel's 64.
        if isinstance(d.attn_bias, _PAGED_GAPPY_TYPES):
            if d.attn_bias.page_size % _KERNEL_PAGE_SIZE != 0:
                reasons.append(
                    f"paged-gappy page_size={d.attn_bias.page_size} must be a "
                    f"multiple of {_KERNEL_PAGE_SIZE}"
                )
        # Dense bottom-right causal cross-attention requires Mq <= Mkv; for Mq > Mkv
        # the leading rows are fully masked (undefined softmax).
        elif isinstance(d.attn_bias, LowerTriangularFromBottomRightMask):
            if d.query.shape[1] > d.key.shape[1]:
                reasons.append("LowerTriangularFromBottomRightMask requires Mq <= Mkv")
        # Other dense masks are self-attention only (varlen/paged/gappy handle
        # cross-length via their own paths).
        elif not isinstance(
            d.attn_bias,
            _VARLEN_BIAS_TYPES
            + _PADDED_GAPPY_BIAS_TYPES
            + _PAGED_BIAS_TYPES
            + _PAGED_GAPPY_TYPES,
        ):
            q_len = d.query.shape[1]
            kv_len = d.key.shape[1]
            if q_len != kv_len:
                reasons.append(
                    f"cross-attention Mq={q_len} != Mkv={kv_len} not supported "
                    "(dense self-attention only)"
                )

        # Dropout: dense self-attention only (the in-kernel mask is applied on the
        # generic dense path; varlen/paged/gappy dropout is not wired).
        if d.p != 0.0 and isinstance(
            d.attn_bias,
            _VARLEN_BIAS_TYPES
            + _PADDED_GAPPY_BIAS_TYPES
            + _PAGED_BIAS_TYPES
            + _PAGED_GAPPY_TYPES,
        ):
            reasons.append("dropout only supported on the dense self-attention path")

        # Native paged KV has no backward: the dualwave route emits no LSE and there
        # is no paged backward op, so a gradient request (return_lse on that route)
        # would fail at run time. Decline so grad cases dispatch elsewhere.
        _needs_grad = (
            d.query.requires_grad or d.key.requires_grad or d.value.requires_grad
        )
        if _needs_grad and isinstance(d.attn_bias, _PAGED_BIAS_TYPES):
            reasons.append("native paged KV does not support gradients (no backward)")

        # Non-gappy paged KV hardcodes 1/sqrt(head_dim); a custom scale is silently
        # ignored (only gappy paged folds sm_scale).
        if d.scale is not None and isinstance(d.attn_bias, _PAGED_BIAS_TYPES):
            reasons.append("custom scale is not supported for native paged KV")

        # Q/K/V must be last-dim contiguous and 16-byte aligned for the kernel's
        # 128-bit vector loads (raw buffer-resource loads do not tolerate otherwise).
        for _name, _t in (("query", d.query), ("key", d.key), ("value", d.value)):
            if _t.stride(-1) != 1:
                reasons.append(f"{_name} last dim must be contiguous (stride==1)")
            elif _t.data_ptr() % 16 != 0:
                reasons.append(f"{_name} base pointer must be 16-byte aligned")

        # Tensor bias must match the query dtype: the kernel loads bias bytes as the
        # query dtype, so an fp32 bias would be reinterpreted as bf16/f16.
        _bias_t: Optional[torch.Tensor] = None
        if isinstance(d.attn_bias, torch.Tensor):
            _bias_t = d.attn_bias
        elif isinstance(d.attn_bias, LowerTriangularMaskWithTensorBias):
            _bias_t = d.attn_bias._bias
        if _bias_t is not None and _bias_t.dtype != d.query.dtype:
            reasons.append(
                f"attn_bias dtype {_bias_t.dtype} must match query dtype "
                f"{d.query.dtype}"
            )

        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if type(inp.attn_bias) not in cls.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError("Unsupported attn_bias type")
        if inp.query.ndim == 5:
            return cls._apply_bmghk(inp, needs_gradient)
        if inp.query.ndim != 4:
            raise NotImplementedError("Unsupported number of dimensions")
        return cls._apply_bmhk(inp, needs_gradient)

    # ------------------------------------------------------------------ BMGHK
    @classmethod
    def _apply_bmghk(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        # Flatten the G group axis into H (as ck.FwOp does): expanded 5-D KV
        # (stride[3]==0) -> as_strided drop, otherwise flatten(2, 3).
        [_, _, G, Hq, _] = inp.query.shape
        if inp.key.stride()[3] == 0:
            ks, kst = inp.key.size(), inp.key.stride()
            key = inp.key.as_strided(
                (ks[0], ks[1], ks[2], ks[4]), (kst[0], kst[1], kst[2], kst[4])
            )
            vs, vst = inp.value.size(), inp.value.stride()
            value = inp.value.as_strided(
                (vs[0], vs[1], vs[2], vs[4]), (vst[0], vst[1], vst[2], vst[4])
            )
        else:
            key = inp.key.flatten(2, 3)
            value = inp.value.flatten(2, 3)
        # A 5-D tensor bias [B, G, H, q, kv] flattens (G, H) to match the folded
        # head axis; mask objects pass through unchanged.
        bias_flat = inp.attn_bias
        if isinstance(bias_flat, torch.Tensor) and bias_flat.ndim == 5:
            bias_flat = bias_flat.flatten(1, 2)
        elif isinstance(bias_flat, LowerTriangularMaskWithTensorBias):
            _bt = bias_flat._bias
            if _bt.ndim == 5:
                bias_flat = LowerTriangularMaskWithTensorBias(_bt.flatten(1, 2))
        flat = Inputs(
            query=inp.query.flatten(2, 3),
            key=key,
            value=value,
            attn_bias=bias_flat,
            p=inp.p,
            scale=inp.scale,
            output_dtype=inp.output_dtype,
        )
        out, ctx = cls._apply_bmhk(flat, needs_gradient)
        out = out.unflatten(2, (G, Hq))
        if ctx is not None:
            # Update the existing context in place so dropout rng_state (and any
            # other fields) survive; rebuilding a Context would drop rng_state and
            # break ck.BwOp for 5-D dropout inputs.
            ctx.lse = ctx.lse.unflatten(1, (G, Hq))
            ctx.out = out
        return out, ctx

    # ------------------------------------------------------------------- BMHK
    @classmethod
    def _apply_bmhk(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        from mslk.attention.flydsl import flydsl_flash_attn_func

        bias = inp.attn_bias
        causal = isinstance(bias, _CAUSAL_BIAS_TYPES)

        # Head-dim padding: non-native dims are zero-padded to the next of
        # {64, 128, 256}. Paged KV is never padded (gated to native D upstream).
        D_native = inp.query.shape[-1]
        Dpad = _pad_head_dim(D_native)
        needs_pad = Dpad != D_native

        def _pad_hd(t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(t, (0, Dpad - D_native)) if needs_pad else t

        kw: dict = {"causal": causal, "num_kv_heads": inp.key.shape[-2]}

        # window_left = _window_size (xformers keeps kv > q - window_size).
        if isinstance(bias, _WINDOW_BIAS_TYPES):
            kw["window_left"] = int(bias._window_size)

        if isinstance(bias, torch.Tensor):
            kw["bias"] = bias
        elif isinstance(bias, LowerTriangularMaskWithTensorBias):
            kw["bias"] = bias._bias

        # Dense/varlen fold sm_scale at build time; when padding, pin the native
        # scale so the padded 1/sqrt(Dpad) does not change the result.
        if inp.scale is not None:
            kw["sm_scale"] = float(inp.scale)
        elif needs_pad:
            kw["sm_scale"] = 1.0 / math.sqrt(D_native)

        kw["return_lse"] = needs_gradient

        # Dropout (dense only): Python-only mask via native torch RNG. This is
        # CUDA-graph-safe and advances per replay, and drops the CK philox
        # dependency. NOTE: backward mask regeneration is not implemented yet --
        # training (needs_gradient) with dropout is rejected below until a
        # flydsl dropout backward that reproduces this mask lands.
        if inp.p != 0.0:
            B = inp.query.shape[0]
            Sq = inp.query.shape[1]
            H = inp.query.shape[2]
            Skv = inp.key.shape[1]
            keep = (
                torch.rand(
                    B, H, Sq, Skv, device=inp.query.device, dtype=torch.float32
                )
                >= inp.p
            )
            # uint8 keep-mask (0/1), half the memory of a bf16/f16 mask. The
            # 1/(1-p) inverted-dropout scale is applied to the output in _run
            # (a constant factor, so it factors out of the per-(q,k) sum).
            kw["dropout_mask"] = keep.to(torch.uint8)

        # Shared tail: pad the head dim, call, unpack LSE, slice padded columns off O,
        # restore input shape (slice/reshape are no-ops when a path didn't pad).
        def _run(q, k, v, pad=True):
            if pad:
                q, k, v = _pad_hd(q), _pad_hd(k), _pad_hd(v)
            result = flydsl_flash_attn_func(q, k, v, **kw)
            out, lse = result if needs_gradient else (result, None)
            out = out[..., :D_native].reshape(inp.query.shape)
            if inp.p != 0.0:
                # Inverted-dropout output scale (the uint8 mask carries no scale).
                out = out * (1.0 / (1.0 - inp.p))
            return out, lse

        def _set_varlen_kw(cu_q, cu_kv, max_q, max_kv, cross):
            kw["cu_seqlens_q"] = cu_q
            kw["cu_seqlens_kv"] = cu_kv
            kw["max_seqlen_q"] = max_q
            kw["max_seqlen_kv"] = max_kv
            kw["cross_seqlen"] = cross

        # Pack q/k/v to flat [total, H, D] for the varlen path.
        def _flatten_qkv():
            q = inp.query.reshape(-1, inp.query.shape[-2], inp.query.shape[-1])
            k = inp.key.reshape(-1, inp.key.shape[-2], inp.key.shape[-1])
            v = inp.value.reshape(-1, inp.value.shape[-2], inp.value.shape[-1])
            return q, k, v

        if isinstance(bias, _PAGED_BIAS_TYPES):
            # Paged KV: Q packed [1, total_q, H, D], K/V the physical page cache.
            # Drives the kernel's varlen+paged path. Native-D only, so pad=False.
            H = inp.query.shape[-2]
            Hkv = inp.key.shape[-2]
            Dk = inp.key.shape[-1]
            q = inp.query.reshape(-1, H, inp.query.shape[-1])
            page_size = int(bias.page_size)
            # The paged kernel fixes PAGE_SIZE == 64. For a larger physical page_size
            # (a multiple of 64), reshape the cache into 64-row sub-pages and expand
            # block_table so each logical page p maps to sub-pages [p*sub, p*sub+sub);
            # this preserves the physical layout while giving the kernel 64-row pages.
            sub = page_size // _KERNEL_PAGE_SIZE
            k = inp.key.reshape(-1, _KERNEL_PAGE_SIZE, Hkv, Dk)
            v = inp.value.reshape(-1, _KERNEL_PAGE_SIZE, Hkv, Dk)
            seqlen_k = bias.k_seqinfo.seqlen.to(q.device)
            # cu_seqlens_kv = cumulative per-seq KV lengths; pages via block_table.
            kseq = torch.nn.functional.pad(
                seqlen_k.to(torch.int32).cumsum(0, dtype=torch.int32), (1, 0)
            )
            _set_varlen_kw(
                bias.q_seqinfo.seqstart.to(q.device),
                kseq,
                int(bias.q_seqinfo.max_seqlen),
                int(bias.k_seqinfo.max_seqlen),
                True,
            )
            bt = bias.block_tables.to(torch.int32).to(q.device)
            if sub > 1:
                bt = (
                    bt.unsqueeze(-1) * sub
                    + torch.arange(sub, dtype=torch.int32, device=q.device)
                ).reshape(bt.shape[0], bt.shape[1] * sub)
            kw["block_table"] = bt
            kw["seqlen_k"] = seqlen_k
            kw["kv_cache_layout"] = "linear"
            out, lse = _run(q, k, v, pad=False)
        elif isinstance(bias, _PADDED_GAPPY_BIAS_TYPES):
            # Padded/gappy KV: pass the ORIGINAL K/V store (no repack) plus a per-seq
            # absolute KV start; the kernel gathers each seq in-place. seqstart and
            # seqlen are already device tensors, so no host sync.
            q, k, v = _flatten_qkv()
            seqlen_kv = bias.k_seqinfo.seqlen.to(torch.int32).to(q.device)
            kcum = torch.nn.functional.pad(
                seqlen_kv.cumsum(0, dtype=torch.int32), (1, 0)
            )
            _set_varlen_kw(
                bias.q_seqinfo.seqstart.to(q.device),
                kcum,
                int(bias.q_seqinfo.max_seqlen),
                int(bias.k_seqinfo.max_seqlen),
                True,
            )
            kw["kv_seqstart"] = bias.k_seqinfo.seqstart.to(torch.int32).to(q.device)
            out, lse = _run(q, k, v)
        elif isinstance(bias, _PAGED_GAPPY_TYPES):
            # Paged-gappy KV: pass the physical page cache as-is; the kernel resolves
            # each row's page (per-row, since a non-aligned logical start straddles
            # pages) from block_table + a per-seq logical start. No host gather.
            # k_seqinfo: seqstart_py = per-seq logical start, seqlen_py = logical end.
            H = inp.query.shape[-2]
            Hkv = inp.key.shape[-2]
            Dk = inp.key.shape[-1]
            q = _pad_hd(inp.query.reshape(-1, H, Dk))
            page_size = int(bias.page_size)
            # The generic paged kernel fixes PAGE_SIZE == BLOCK_N == 64. Reshape the
            # physical cache to 64-row sub-pages and expand block_table so each
            # logical page p maps to sub-pages [p*sub, p*sub+sub); this keeps the
            # in-kernel per-row page lookup valid at any physical page_size. Head dim
            # is zero-padded (last axis) like the other paths.
            sub = page_size // _KERNEL_PAGE_SIZE
            k = _pad_hd(inp.key.reshape(-1, _KERNEL_PAGE_SIZE, Hkv, Dk))
            v = _pad_hd(inp.value.reshape(-1, _KERNEL_PAGE_SIZE, Hkv, Dk))
            # sm_scale (native, pinned above when padding) is folded by the generic
            # paged kernel, so no Q pre-scale is needed.
            bt = bias.block_tables.to(torch.int32).to(q.device)
            bt64 = (
                bt.unsqueeze(-1) * sub
                + torch.arange(sub, dtype=torch.int32, device=q.device)
            ).reshape(bt.shape[0], bt.shape[1] * sub)
            # seqstart is [B+1] (per-seq logical start + trailing sentinel); seqlen
            # is [B] logical ends. Per-seq length = end - start.
            B_kv = bias.k_seqinfo.seqlen.shape[0]
            kstart = bias.k_seqinfo.seqstart[:B_kv].to(torch.int32)
            kend = bias.k_seqinfo.seqlen.to(torch.int32)
            kseqlen = (kend - kstart).to(q.device)
            kcum = torch.nn.functional.pad(kseqlen.cumsum(0, dtype=torch.int32), (1, 0))
            kw["cu_seqlens_q"] = bias.q_seqinfo.seqstart.to(q.device)
            kw["cu_seqlens_kv"] = kcum
            kw["max_seqlen_q"] = int(bias.q_seqinfo.max_seqlen)
            kw["max_seqlen_kv"] = int(bias.k_seqinfo.max_seqlen)
            kw["cross_seqlen"] = True
            kw["block_table"] = bt64
            kw["kv_seqstart"] = kstart.to(q.device)
            kw["kv_cache_layout"] = "linear"
            out, lse = _run(q, k, v, pad=False)
        elif isinstance(bias, _VARLEN_BIAS_TYPES):
            # Packed varlen via per-sequence seqstart offsets; per-block q/k lengths
            # may differ (cross_seqlen).
            q, k, v = _flatten_qkv()
            qseq = bias.q_seqinfo.seqstart.to(q.device)
            kseq = bias.k_seqinfo.seqstart.to(q.device)
            max_q = int(bias.q_seqinfo.max_seqlen)
            max_kv = int(bias.k_seqinfo.max_seqlen)
            # cross_seqlen enables the general (per-block q!=k length) path; it is a
            # correctness superset of the equal-length fast path, so err toward True.
            # Decide WITHOUT a device->host sync (torch.equal(.cpu()) breaks CUDA-graph
            # capture): unequal max => cross; identical seqstart object => not cross;
            # otherwise assume cross (rare distinct-but-equal layouts lose only the
            # equal-length optimization, never correctness).
            cross = (max_q != max_kv) or (qseq is not kseq)
            _set_varlen_kw(qseq, kseq, max_q, max_kv, cross)
            # Top-left-aligned masks need causal_top_left (kernel default is
            # bottom-right); the ...FromBottomRight variant keeps the default.
            if isinstance(
                bias, (BlockDiagonalCausalMask,) + _VARLEN_WINDOW_TOPLEFT_TYPES
            ):
                kw["causal_top_left"] = True
            out, lse = _run(q, k, v)
        else:
            # Dense self-attention.
            out, lse = _run(inp.query, inp.key, inp.value)

        # The kernel returns varlen LSE padded per-batch as [B, H, max_seqlen_q].
        # Repack it to the [1, H, total_q] packed layout (VARLEN_LSE_PACKED=True) so
        # ck.BwOp and automatic dispatch see a consistent LSE for varlen gradients.
        if lse is not None and isinstance(
            bias,
            _VARLEN_BIAS_TYPES
            + _PADDED_GAPPY_BIAS_TYPES
            + _PAGED_BIAS_TYPES
            + _PAGED_GAPPY_TYPES,
        ):
            lse = _pack_varlen_lse(lse, bias.q_seqinfo)

        ctx: Optional[Context] = None
        if needs_gradient:
            if inp.p != 0.0:
                # TODO: implement a flydsl dropout backward that regenerates the
                # Python (torch RNG) mask. Until then the mask cannot be
                # reproduced in the backward (ck.BwOp only knows CK philox), so
                # reject training + dropout rather than return wrong gradients.
                raise NotImplementedError(
                    "flydsl dropout backward is not implemented yet for the "
                    "Python-generated mask; dropout is currently forward/"
                    "inference-only (needs_gradient=False)."
                )
            # No FlyDSL backward yet.
            ctx = Context(lse=lse, out=out, op_bw=None)
        return out, ctx
