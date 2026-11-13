# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Thin wrappers around AITER's flash-attention forward and backward.

AITER's ``flash_attn_func`` accepts Q/K/V in BLHD layout and returns
``(out, lse)`` when ``return_lse=True``.  The backward calls AITER's
``_flash_attn_backward`` directly with saved ``out`` and ``softmax_lse``
from the forward context, avoiding a redundant forward pass.

Nothing here is imported by ``import mslk``; the fmha op module
(``mslk.attention.fmha.rocm_aiter``) imports these lazily inside ``apply``.
"""

import math
from typing import Optional, Tuple

import torch


def aiter_fmha_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass through AITER flash attention.

    Args:
        query: (B, Lq, H, D) bf16/f16
        key:   (B, Lkv, Hkv, D) bf16/f16
        value: (B, Lkv, Hkv, D) bf16/f16
        scale: Optional softmax scale (default 1/sqrt(D))
        causal: Whether to apply causal masking

    Returns:
        out: (B, Lq, H, D) same dtype as query
        lse: (B, H, Lq) float32 log-sum-exp
    """
    from mslk.aiter.common import import_aiter_module

    mha = import_aiter_module("aiter.ops.mha")

    kwargs = {"causal": causal, "return_lse": True}
    if scale is not None:
        kwargs["softmax_scale"] = scale

    result = mha.flash_attn_func(query, key, value, **kwargs)
    out, lse = result[0], result[1]
    return out, lse


def aiter_fmha_backward(
    grad: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass through AITER flash attention.

    Calls AITER's ``_flash_attn_backward`` directly with the saved ``out``
    and ``softmax_lse`` from the forward context, avoiding a redundant
    forward pass.
    """
    from mslk.aiter.common import import_aiter_module

    mha = import_aiter_module("aiter.ops.mha")

    D = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    dq = torch.empty_like(query)
    dk = torch.empty_like(key)
    dv = torch.empty_like(value)

    # AITER's v3 backward needs rng_state even when dropout_p=0.
    # Create a dummy rng_state matching what FlashAttnFunc.forward saves.
    rng_state = torch.cuda.get_rng_state()

    mha._flash_attn_backward(
        dout=grad.contiguous(),
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        out=out.contiguous(),
        softmax_lse=lse.contiguous(),
        dq=dq,
        dk=dk,
        dv=dv,
        dbias=None,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        bias=None,
        alibi_slopes=None,
        deterministic=False,
        rng_state=rng_state,
    )

    return dq, dk, dv
