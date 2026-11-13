# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AITER flash-attention FwOp/BwOp for MSLK's fmha dispatch.

AITER's ``flash_attn_func`` provides high-performance flash attention on
gfx950 (MI350X).  These ops are NOT in the auto-dispatch priority lists;
they are available for explicit use via ``op=aiter.FwOp`` or via future
OpRouting integration (``mslk/aiter/routing.py``).

First-use latency: the pinned AITER build (``noprebuild``) JIT-compiles
CK/ASM modules with ``hipcc`` on first call (~60-70s per kernel config).
Warm up before CUDA-graph capture.

Template: ``mslk/attention/fmha/flydsl.py`` (FwOp/BwOp pattern).
"""

from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple

import torch

from .attn_bias import LowerTriangularFromBottomRightMask, LowerTriangularMask
from .common import AttentionBwOpBase, AttentionFwOpBase, Context, Gradients, Inputs


def _is_aiter_fmha_available() -> bool:
    try:
        from mslk.aiter.common import is_aiter_available

        return is_aiter_available()
    except Exception:
        return False


def _check_aiter(reasons: List[str]) -> bool:
    """Append AITER-unavailability reasons; return True if usable."""
    try:
        from mslk.aiter.common import is_aiter_available, is_aiter_enabled
    except ImportError:
        reasons.append("AITER infrastructure not installed")
        return False
    if not is_aiter_available():
        reasons.append("AITER is not available")
        return False
    if not is_aiter_enabled("fmha"):
        reasons.append("AITER fmha is disabled")
        return False
    return True


def _check_mask(reasons: List[str], d: Inputs) -> None:
    """Reject masks AITER can't handle correctly."""
    if isinstance(d.attn_bias, LowerTriangularMask):
        Lq = d.query.shape[1]
        Lkv = d.key.shape[1]
        if Lq != Lkv:
            reasons.append(
                "LowerTriangularMask with Lq != Lkv: AITER causal is "
                "bottom-right aligned, not top-left"
            )


def _check_contiguous(reasons: List[str], d: Inputs) -> None:
    if d.query.stride(-1) != 1:
        reasons.append("query last dim must be contiguous (stride==1)")
    if d.key.stride(-1) != 1:
        reasons.append("key last dim must be contiguous (stride==1)")
    if d.value.stride(-1) != 1:
        reasons.append("value last dim must be contiguous (stride==1)")


class FwOp(AttentionFwOpBase):
    """AITER flash-attention forward (gfx950, dense only)."""

    OPERATOR = staticmethod(_is_aiter_fmha_available)
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.bfloat16, torch.float16}
    SUPPORTED_MAX_K = 256
    SUPPORTED_MIN_K = 1
    SUPPORTED_GFX_ARCHS: Set[str] = {"gfx950"}

    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
    )

    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = False
    SUPPORTS_PARTIAL = False
    VARLEN_LSE_PACKED = True
    NAME = "aiterF"

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

    _TEST_K: List[int] = [64, 128, 256]

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super().not_supported_reasons(d)
        _check_aiter(reasons)
        K = d.query.shape[-1]
        if not (1 <= K <= 256):
            reasons.append(f"query head_dim={K} out of range [1, 256]")
        _check_mask(reasons, d)
        _check_contiguous(reasons, d)
        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if inp.query.ndim != 4:
            raise NotImplementedError("Only 4D BMHK tensors are supported")

        from mslk.attention.aiter.fmha import aiter_fmha_forward

        causal = isinstance(
            inp.attn_bias, (LowerTriangularMask, LowerTriangularFromBottomRightMask)
        )
        scale = float(inp.scale) if inp.scale is not None else None

        out, lse = aiter_fmha_forward(
            inp.query, inp.key, inp.value, scale=scale, causal=causal
        )

        ctx: Optional[Context] = None
        if needs_gradient:
            ctx = Context(lse=lse, out=out, op_bw=None)
        return out, ctx


class BwOp(AttentionBwOpBase):
    """AITER flash-attention backward (gfx950, dense only)."""

    OPERATOR = staticmethod(_is_aiter_fmha_available)
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.bfloat16, torch.float16}
    SUPPORTED_MAX_K = 256
    SUPPORTED_MIN_K = 1
    SUPPORTED_GFX_ARCHS: Set[str] = {"gfx950"}

    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
    )

    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = False
    SUPPORTS_ATTN_BIAS_GRAD = False
    VARLEN_LSE_PACKED = True
    NAME = "aiterB"

    _TEST_K: List[int] = [64, 128, 256]

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super().not_supported_reasons(d)
        _check_aiter(reasons)
        K = d.query.shape[-1]
        if not (1 <= K <= 256):
            reasons.append(f"query head_dim={K} out of range [1, 256]")
        _check_mask(reasons, d)
        _check_contiguous(reasons, d)
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        from mslk.attention.aiter.fmha import aiter_fmha_backward

        causal = isinstance(
            inp.attn_bias, (LowerTriangularMask, LowerTriangularFromBottomRightMask)
        )
        scale = float(inp.scale) if inp.scale is not None else None

        dq, dk, dv = aiter_fmha_backward(
            grad,
            inp.query,
            inp.key,
            inp.value,
            ctx.out,
            ctx.lse,
            scale=scale,
            causal=causal,
        )

        return Gradients(dq=dq, dk=dk, dv=dv)
