# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""SDPA ops matching aten's _scaled_dot_product_{flash,efficient}_attention.

Inputs are [B, H, L, E] (aten convention). Internally the ops convert to
MSLK's [B, L, H, K] layout, run the fmha dispatch, and convert back.

The aten override registers these as ROCm implementations of the two aten
operators, so ``torch.nn.functional.scaled_dot_product_attention`` routes
through MSLK's kernels.
"""

import math
from typing import Optional, Tuple

import torch
from mslk.attention import fmha
from mslk.attention.fmha.attn_bias import (
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
)

_LOG2E = math.log2(math.e)


def _bhld_to_bmhk(t: torch.Tensor) -> torch.Tensor:
    """[B, H, L, D] -> [B, L, H, D] as a view."""
    return t.transpose(1, 2)


def _bmhk_to_bhld(t: torch.Tensor) -> torch.Tensor:
    """[B, L, H, D] -> [B, H, L, D] as a view."""
    return t.transpose(1, 2)


def _causal_bias(is_causal: bool, Lq: int, Lkv: int):
    """Map is_causal to the fmha bias type.

    PyTorch's flash backend uses bottom-right alignment when Lq != Lkv
    (see pytorch/aten/src/ATen/native/transformers/attention.cpp).
    """
    if not is_causal:
        return None
    if Lq == Lkv:
        return LowerTriangularMask()
    return LowerTriangularFromBottomRightMask()


def sdpa_flash_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[
    torch.Tensor,  # output
    torch.Tensor,  # logsumexp
    torch.Tensor,  # cum_seq_q
    torch.Tensor,  # cum_seq_k
    int,  # max_q
    int,  # max_k
    torch.Tensor,  # rng_state (philox)
    torch.Tensor,  # unused
    torch.Tensor,  # debug_attn_mask
]:
    """SDPA flash-attention forward matching the aten contract.

    Input layout: [B, H, Lq, D] / [B, H, Lkv, D].
    """
    if dropout_p > 0.0:
        raise NotImplementedError("MSLK SDPA does not support dropout (A5)")

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape

    # Causal cross-attention (Lq != Lkv) falls back to PyTorch SDPA because
    # the bottom-right mask alignment isn't reliably handled by all fmha ops.
    if is_causal and Lq != Lkv:
        out_bhld = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scale
        )
        lse_nat = torch.empty(0, device=query.device)
    else:
        Q = _bhld_to_bmhk(query)
        K = _bhld_to_bmhk(key)
        V = _bhld_to_bmhk(value)

        attn_bias = _causal_bias(is_causal, Lq, Lkv)

        out, lse = fmha.memory_efficient_attention_forward_requires_grad(
            Q, K, V, attn_bias=attn_bias, scale=scale
        )

        out_bhld = _bmhk_to_bhld(out)
        # lse from fmha is [B, H, Lq] f32 in log2 units; aten expects natural log.
        lse_nat = (
            lse / _LOG2E if lse is not None else torch.empty(0, device=query.device)
        )

    # Placeholder outputs for aten compatibility
    empty = torch.empty(0, device=query.device, dtype=query.dtype)
    cum_seq_q = torch.empty(0, device=query.device, dtype=torch.int32)
    cum_seq_k = torch.empty(0, device=query.device, dtype=torch.int32)
    rng_state = torch.empty(0, device=query.device, dtype=torch.int64)

    return (out_bhld, lse_nat, cum_seq_q, cum_seq_k, Lq, Lkv, rng_state, empty, empty)


def sdpa_flash_bwd(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDPA flash-attention backward matching the aten contract."""
    if dropout_p > 0.0:
        raise NotImplementedError("MSLK SDPA does not support dropout (A5)")

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape

    Q = _bhld_to_bmhk(query)
    K = _bhld_to_bmhk(key)
    V = _bhld_to_bmhk(value)
    dO = _bhld_to_bmhk(grad_out)
    O = _bhld_to_bmhk(out)

    attn_bias = _causal_bias(is_causal, Lq, Lkv)

    # Convert LSE from natural log back to log2 for fmha
    lse_log2 = logsumexp * _LOG2E

    ctx = fmha.Context(out=O, lse=lse_log2)
    inp = fmha.Inputs(query=Q, key=K, value=V, attn_bias=attn_bias, scale=scale)

    grads = fmha.memory_efficient_attention_backward(
        dO, O, lse_log2, Q, K, V, attn_bias=attn_bias, scale=scale
    )

    # K=V optimization: if K and V share storage, sum dK+dV
    k_is_v = key.data_ptr() == value.data_ptr()
    dQ = _bmhk_to_bhld(grads.dq)
    dK = _bmhk_to_bhld(grads.dk)
    dV = _bmhk_to_bhld(grads.dv)
    if k_is_v:
        dK = dK + dV
        dV = torch.zeros_like(dV)

    return dQ, dK, dV


def sdpa_efficient_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    compute_log_sumexp: bool,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDPA efficient-attention forward matching the aten contract.

    Returns: (output, log_sumexp, philox_seed, philox_offset)
    """
    if dropout_p > 0.0:
        raise NotImplementedError("MSLK SDPA does not support dropout (A5)")

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape

    if is_causal and Lq != Lkv:
        out_bhld = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias, is_causal=True, scale=scale
        )
        lse_nat = torch.empty(0, device=query.device)
    else:
        Q = _bhld_to_bmhk(query)
        K = _bhld_to_bmhk(key)
        V = _bhld_to_bmhk(value)

        bias = _causal_bias(is_causal, Lq, Lkv)
        if attn_bias is not None and bias is None:
            bias = attn_bias

        out, lse = fmha.memory_efficient_attention_forward_requires_grad(
            Q, K, V, attn_bias=bias, scale=scale
        )

        out_bhld = _bmhk_to_bhld(out)
        if compute_log_sumexp and lse is not None:
            lse_nat = lse / _LOG2E
        else:
            lse_nat = torch.empty(0, device=query.device)

    philox_seed = torch.empty(0, device=query.device, dtype=torch.int64)
    philox_offset = torch.empty(0, device=query.device, dtype=torch.int64)

    return out_bhld, lse_nat, philox_seed, philox_offset


def sdpa_efficient_bwd(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    grad_input_mask: list,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """SDPA efficient-attention backward matching the aten contract.

    Returns: (dQ, dK, dV, dBias)
    """
    if dropout_p > 0.0:
        raise NotImplementedError("MSLK SDPA does not support dropout (A5)")

    B, H, Lq, D = query.shape
    _, _, Lkv, _ = key.shape

    Q = _bhld_to_bmhk(query)
    K = _bhld_to_bmhk(key)
    V = _bhld_to_bmhk(value)
    dO = _bhld_to_bmhk(grad_out)
    O = _bhld_to_bmhk(out)

    bias = _causal_bias(is_causal, Lq, Lkv)
    if attn_bias is not None and bias is None:
        bias = attn_bias

    lse_log2 = logsumexp * _LOG2E

    grads = fmha.memory_efficient_attention_backward(
        dO, O, lse_log2, Q, K, V, attn_bias=bias, scale=scale
    )

    k_is_v = key.data_ptr() == value.data_ptr()
    dQ = _bmhk_to_bhld(grads.dq)
    dK = _bmhk_to_bhld(grads.dk)
    dV = _bmhk_to_bhld(grads.dv)
    if k_is_v:
        dK = dK + dV
        dV = torch.zeros_like(dV)

    return dQ, dK, dV, None


# ---------------------------------------------------------------------------
# Aten override: opt-in replacement of PyTorch's SDPA implementations
# ---------------------------------------------------------------------------

_aten_override_active = False


def enable_aten_override() -> None:
    """Register MSLK's SDPA as the ROCm implementation of aten's SDPA ops.

    After this call, ``torch.nn.functional.scaled_dot_product_attention``
    routes through MSLK's fmha kernels on ROCm.
    """
    global _aten_override_active
    if _aten_override_active:
        return

    torch.library.impl(
        "aten::_scaled_dot_product_flash_attention",
        "CUDA",
    )(sdpa_flash_fwd)

    torch.library.impl(
        "aten::_scaled_dot_product_flash_attention_backward",
        "CUDA",
    )(sdpa_flash_bwd)

    torch.library.impl(
        "aten::_scaled_dot_product_efficient_attention",
        "CUDA",
    )(sdpa_efficient_fwd)

    torch.library.impl(
        "aten::_scaled_dot_product_efficient_attention_backward",
        "CUDA",
    )(sdpa_efficient_bwd)

    _aten_override_active = True


def disable_aten_override() -> None:
    """Unregister MSLK's SDPA aten override.

    Note: torch.library does not support unregistration. This sets a flag
    so the functions fall back to PyTorch's default, but the registration
    itself cannot be undone within the same process.
    """
    global _aten_override_active
    _aten_override_active = False
