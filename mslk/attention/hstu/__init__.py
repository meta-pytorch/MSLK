# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HSTU jagged attention: ragged self-attention, cross attention, and grouped
ragged self-attention.

Entry points
------------
Forward and backward functions with the signatures of the public Triton
references:

- :func:`ragged_hstu_attention_fwd`, :func:`ragged_hstu_attention_bwd`
- :func:`cross_hstu_attention_fwd`, :func:`cross_hstu_attention_bwd`
- :func:`grouped_ragged_hstu_attention_fwd`,
  :func:`grouped_ragged_hstu_attention_bwd`

Differentiable wrappers (autograd runs the matching backward):

- :func:`ragged_hstu_attention`
- :func:`cross_hstu_attention`
- :func:`grouped_ragged_hstu_attention`

Tensors are packed (jagged): ``[total_tokens, heads, head_dim]`` with
``seq_offsets`` giving each sequence's token range.

Shared K/V. With ``shared_kv=True`` V is K: the forward requires ``v is k``
(cross also accepts ``v=None``) and the backward writes the combined dK+dV
gradient to ``dk``. The ragged and grouped backwards then require
``dv=None``; :func:`cross_hstu_attention_bwd` returns ``dv`` as an alias of
``dk``. The grouped wrapper detects sharing from ``group_v is group_k``.

Every op is a ``torch.library`` op in the ``mslk`` namespace with a fake
implementation, so it works under torch.compile. On ROCm each call is routed
per shape between MSLK's kernel and other backends (see
:mod:`mslk.attention.dispatch`). Warm up before CUDA-graph capture: the first
call of a shape loads the routing table and may import and JIT-compile a
backend, and a captured graph keeps the backend chosen at capture time.
"""

from typing import Optional, Tuple

import mslk.attention.hstu._ops  # noqa: F401  registers the ops
import torch

__all__ = [
    "ragged_hstu_attention",
    "ragged_hstu_attention_fwd",
    "ragged_hstu_attention_bwd",
    "cross_hstu_attention",
    "cross_hstu_attention_fwd",
    "cross_hstu_attention_bwd",
    "grouped_ragged_hstu_attention",
    "grouped_ragged_hstu_attention_fwd",
    "grouped_ragged_hstu_attention_bwd",
]


# ── Ragged self-attention ────────────────────────────────────────────────


def ragged_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    num_softmax_heads: int = 0,
    shared_kv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ragged HSTU self-attention forward.

    ``N`` is the maximum sequence length; ``q``, ``k``: ``[T, H, Dqk]``,
    ``v``: ``[T, H, Dv]``. Returns ``(out, M)``: ``out`` ``[T, H, Dv]`` and
    ``M`` ``[T, H]`` float32, the forward state the backward takes.
    """
    return torch.ops.mslk.hstu_ragged_fwd(
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        invalid_attn_mask_type,
        num_targets,
        attn_scale,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        num_softmax_heads,
        shared_kv,
    )


def ragged_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    invalid_attn_mask_type: str,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    full_attn_size: int,
    num_softmax_heads: int = 0,
    M: Optional[torch.Tensor] = None,
    Out: Optional[torch.Tensor] = None,
    shared_kv: bool = False,
) -> None:
    """Ragged HSTU self-attention backward; writes ``dq``, ``dk`` and ``dv``
    in place. With ``shared_kv`` pass ``dv=None``: dK+dV is written to ``dk``.
    """
    torch.ops.mslk.hstu_ragged_bwd(
        dout,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        seq_offsets,
        num_targets,
        attn_scale,
        N,
        alpha,
        max_attn_len,
        invalid_attn_mask_type,
        contextual_seq_len,
        sort_by_length_indices,
        full_attn_size,
        num_softmax_heads,
        M,
        Out,
        shared_kv,
    )


def _ragged_setup_context(ctx, inputs, output) -> None:
    (
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        invalid_attn_mask_type,
        num_targets,
        attn_scale,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        num_softmax_heads,
        shared_kv,
    ) = inputs
    out, M = output
    ctx.mark_non_differentiable(M)
    # With shared_kv, v is k: save it once and restore the alias in backward,
    # since unpacked saved tensors need not be the same object.
    saved_v = None if shared_kv else v
    ctx.save_for_backward(q, k, saved_v, seq_offsets, num_targets, attn_scale, out, M)
    ctx.N = N
    ctx.alpha = alpha
    ctx.invalid_attn_mask_type = invalid_attn_mask_type
    ctx.max_attn_len = max_attn_len
    ctx.contextual_seq_len = contextual_seq_len
    ctx.full_attn_size = full_attn_size
    ctx.num_softmax_heads = num_softmax_heads
    ctx.shared_kv = shared_kv


def _ragged_backward(ctx, grad_out, grad_M):
    q, k, v, seq_offsets, num_targets, attn_scale, out, M = ctx.saved_tensors
    if ctx.shared_kv:
        v = k
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = None if ctx.shared_kv else torch.empty_like(v)
    ragged_hstu_attention_bwd(
        grad_out.contiguous(),
        q,
        k,
        v,
        dq,
        dk,
        dv,
        seq_offsets,
        num_targets,
        attn_scale,
        ctx.N,
        ctx.alpha,
        ctx.max_attn_len,
        ctx.invalid_attn_mask_type,
        ctx.contextual_seq_len,
        None,
        ctx.full_attn_size,
        ctx.num_softmax_heads,
        M,
        out,
        ctx.shared_kv,
    )
    # With shared_kv, v is k and dk already holds dK+dV.
    return (None, None, dq, dk, dv) + (None,) * 9


torch.library.register_autograd(
    "mslk::hstu_ragged_fwd", _ragged_backward, setup_context=_ragged_setup_context
)


def ragged_hstu_attention(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    full_attn_size: int = 0,
    num_softmax_heads: int = 0,
    shared_kv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable ragged HSTU self-attention: :func:`ragged_hstu_attention_fwd`
    with defaults for the optional arguments. ``M`` is not differentiable."""
    return ragged_hstu_attention_fwd(
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        invalid_attn_mask_type,
        num_targets,
        attn_scale,
        max_attn_len,
        contextual_seq_len,
        full_attn_size,
        num_softmax_heads,
        shared_kv,
    )


# ── Cross attention ──────────────────────────────────────────────────────


def cross_hstu_attention_fwd(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: torch.Tensor,
    G: int,
    shared_kv: bool = False,
    num_softmax_heads: int = 0,
    enable_tma: bool = True,
    truncate_method: str = "none",
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HSTU cross attention forward.

    ``q``: ``[Tq, H, Dqk]`` packed by ``seq_offsets_q``; ``k``, ``v``:
    ``[Tkv, H, D]`` packed by ``seq_offsets``; ``max_seq_len`` is the maximum
    KV length. ``enable_tma`` is accepted and ignored. Returns ``(out, M)``:
    ``out`` ``[Tq, H, Dv]`` and ``M`` ``[Tq, H]`` float32.
    """
    return torch.ops.mslk.hstu_cross_fwd(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        seq_offsets_q,
        max_q_len,
        attn_scale,
        G,
        shared_kv,
        num_softmax_heads,
        enable_tma,
        truncate_method,
        num_targets,
        causal,
    )


def cross_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    causal: bool,
    shared_kv: bool,
    G: int,
    num_softmax_heads: int = 0,
    M: Optional[torch.Tensor] = None,
    Delta: Optional[torch.Tensor] = None,
    stride_mm: int = 0,
    truncate_method: str = "none",
    enable_tma: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HSTU cross attention backward; returns ``(dq, dk, dv)``.

    With ``shared_kv``, ``dk`` holds dK+dV and ``dv`` is the same tensor as
    ``dk`` (as in the reference): add only one of them to K's gradient.
    """
    grads = torch.ops.mslk.hstu_cross_bwd(
        dout,
        q,
        k,
        v,
        seq_offsets,
        attn_scale,
        max_seq_len,
        alpha,
        max_q_len,
        seq_offsets_q,
        num_targets,
        causal,
        shared_kv,
        G,
        num_softmax_heads,
        M,
        Delta,
        stride_mm,
        truncate_method,
        enable_tma,
    )
    dq, dk = grads[0], grads[1]
    dv = dk if shared_kv else grads[2]
    return dq, dk, dv


def _cross_setup_context(ctx, inputs, output) -> None:
    (
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        seq_offsets_q,
        max_q_len,
        attn_scale,
        G,
        shared_kv,
        num_softmax_heads,
        enable_tma,
        truncate_method,
        num_targets,
        causal,
    ) = inputs
    out, M = output
    ctx.mark_non_differentiable(M)
    # With shared_kv, V is k (or None): the backward passes v=None.
    saved_v = None if shared_kv else v
    ctx.save_for_backward(
        q, k, saved_v, seq_offsets, seq_offsets_q, attn_scale, num_targets, M
    )
    ctx.max_seq_len = max_seq_len
    ctx.alpha = alpha
    ctx.max_q_len = max_q_len
    ctx.G = G
    ctx.shared_kv = shared_kv
    ctx.num_softmax_heads = num_softmax_heads
    ctx.enable_tma = enable_tma
    ctx.truncate_method = truncate_method
    ctx.causal = causal


def _cross_backward(ctx, grad_out, grad_M):
    q, k, v, seq_offsets, seq_offsets_q, attn_scale, num_targets, M = ctx.saved_tensors
    dq, dk, dv = cross_hstu_attention_bwd(
        grad_out.contiguous(),
        q,
        k,
        v,
        seq_offsets,
        attn_scale,
        ctx.max_seq_len,
        ctx.alpha,
        ctx.max_q_len,
        seq_offsets_q,
        num_targets,
        ctx.causal,
        ctx.shared_kv,
        ctx.G,
        ctx.num_softmax_heads,
        M,
        None,  # Delta: computed by the backend
        M.stride(0),
        ctx.truncate_method,
        ctx.enable_tma,
    )
    # With shared_kv, dk holds dK+dV and dv aliases it: V (k itself or None)
    # gets no separate gradient, or K's would be counted twice.
    if ctx.shared_kv:
        dv = None
    return (None, None, dq, dk, dv) + (None,) * 11


torch.library.register_autograd(
    "mslk::hstu_cross_fwd", _cross_backward, setup_context=_cross_setup_context
)


def cross_hstu_attention(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: torch.Tensor,
    G: int,
    shared_kv: bool = False,
    num_softmax_heads: int = 0,
    enable_tma: bool = True,
    truncate_method: str = "none",
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable HSTU cross attention: :func:`cross_hstu_attention_fwd`
    with autograd. ``M`` is not differentiable."""
    return cross_hstu_attention_fwd(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        seq_offsets_q,
        max_q_len,
        attn_scale,
        G,
        shared_kv,
        num_softmax_heads,
        enable_tma,
        truncate_method,
        num_targets,
        causal,
    )


# ── Grouped ragged self-attention ────────────────────────────────────────


def grouped_ragged_hstu_attention_fwd(
    Z: int,
    group_q: torch.Tensor,
    group_k: torch.Tensor,
    group_v: torch.Tensor,
    seq_offsets: torch.Tensor,
    group_offsets: torch.Tensor,
    group_num_targets: Optional[torch.Tensor],
    max_seq_len_tensor: torch.Tensor,
    max_attn_len_tensor: torch.Tensor,
    full_attn_size_tensor: torch.Tensor,
    contextual_seq_len_tensor: torch.Tensor,
    alpha: float,
    invalid_attn_mask_type: str,
    attn_scale_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Grouped ragged HSTU self-attention forward (SiLU).

    ``G = len(group_offsets) - 1`` groups of ``Z`` sequences each share the
    packed ``group_q``, ``group_k`` (``[T, H, DimQ]``) and ``group_v``
    (``[T, H, DimV]``); ``seq_offsets`` has ``G * Z + 1`` entries,
    ``group_num_targets`` ``G * Z``, and the per-group settings are ``[G]``
    integer tensors. Only the length of ``group_offsets`` is used.

    ``attn_scale_tensor`` selects the scale mode: ``None`` uses ``1 /`` the
    group's maximum sequence length; if ``attn_scale_tensor[0].ndim == 0``
    (a ``[G]`` tensor) one scale is read per group; otherwise (``[T, ...]``)
    scales are indexed by packed query token.

    Returns a new tensor with the shape, dtype and device of ``group_v``.
    """
    return torch.ops.mslk.hstu_grouped_fwd(
        Z,
        group_q,
        group_k,
        group_v,
        seq_offsets,
        group_offsets,
        group_num_targets,
        max_seq_len_tensor,
        max_attn_len_tensor,
        full_attn_size_tensor,
        contextual_seq_len_tensor,
        alpha,
        invalid_attn_mask_type,
        attn_scale_tensor,
    )


def grouped_ragged_hstu_attention_bwd(
    dout: torch.Tensor,
    Z: int,
    group_q: torch.Tensor,
    group_k: torch.Tensor,
    group_v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    group_offsets: torch.Tensor,
    group_num_targets: Optional[torch.Tensor],
    max_seq_len_tensor: torch.Tensor,
    max_attn_len_tensor: torch.Tensor,
    full_attn_size_tensor: torch.Tensor,
    contextual_seq_len_tensor: torch.Tensor,
    alpha: float,
    invalid_attn_mask_type: str,
    attn_scale_tensor: Optional[torch.Tensor] = None,
    Out: Optional[torch.Tensor] = None,
    shared_kv: bool = False,
) -> None:
    """Grouped ragged HSTU self-attention backward; writes ``dq``, ``dk`` and
    ``dv`` in place. With ``shared_kv`` (``group_v is group_k``) pass
    ``dv=None``: dK+dV is written to ``dk``."""
    torch.ops.mslk.hstu_grouped_bwd(
        dout,
        Z,
        group_q,
        group_k,
        group_v,
        dq,
        dk,
        dv,
        seq_offsets,
        group_offsets,
        group_num_targets,
        max_seq_len_tensor,
        max_attn_len_tensor,
        full_attn_size_tensor,
        contextual_seq_len_tensor,
        alpha,
        invalid_attn_mask_type,
        attn_scale_tensor,
        Out,
        shared_kv,
    )


def _grouped_setup_context(ctx, inputs, output) -> None:
    (
        Z,
        q,
        k,
        v,
        seq_offsets,
        group_offsets,
        num_targets,
        max_seq_len_tensor,
        max_attn_len_tensor,
        full_attn_size_tensor,
        contextual_seq_len_tensor,
        alpha,
        invalid_attn_mask_type,
        attn_scale_tensor,
    ) = inputs
    shared_kv = v is k
    ctx.save_for_backward(
        q,
        k,
        None if shared_kv else v,
        seq_offsets,
        group_offsets,
        num_targets,
        max_seq_len_tensor,
        max_attn_len_tensor,
        full_attn_size_tensor,
        contextual_seq_len_tensor,
        attn_scale_tensor,
        output,
    )
    ctx.Z = Z
    ctx.alpha = alpha
    ctx.invalid_attn_mask_type = invalid_attn_mask_type
    ctx.shared_kv = shared_kv


def _grouped_backward(ctx, grad_out):
    (
        q,
        k,
        v,
        seq_offsets,
        group_offsets,
        num_targets,
        max_seq_len_tensor,
        max_attn_len_tensor,
        full_attn_size_tensor,
        contextual_seq_len_tensor,
        attn_scale_tensor,
        out,
    ) = ctx.saved_tensors
    # v was saved once, as k; restore the alias the backward op checks.
    if ctx.shared_kv:
        v = k
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = None if ctx.shared_kv else torch.empty_like(v)
    grouped_ragged_hstu_attention_bwd(
        grad_out.contiguous(),
        ctx.Z,
        q,
        k,
        v,
        dq,
        dk,
        dv,
        seq_offsets,
        group_offsets,
        num_targets,
        max_seq_len_tensor,
        max_attn_len_tensor,
        full_attn_size_tensor,
        contextual_seq_len_tensor,
        ctx.alpha,
        ctx.invalid_attn_mask_type,
        attn_scale_tensor,
        out,
        ctx.shared_kv,
    )
    # With shared_kv, v is k and dk already holds dK+dV.
    return (None, dq, dk, dv) + (None,) * 10


torch.library.register_autograd(
    "mslk::hstu_grouped_fwd", _grouped_backward, setup_context=_grouped_setup_context
)


def grouped_ragged_hstu_attention(
    Z: int,
    group_q: torch.Tensor,
    group_k: torch.Tensor,
    group_v: torch.Tensor,
    seq_offsets: torch.Tensor,
    group_offsets: torch.Tensor,
    group_num_targets: Optional[torch.Tensor],
    max_seq_len_tensor: torch.Tensor,
    max_attn_len_tensor: torch.Tensor,
    full_attn_size_tensor: torch.Tensor,
    contextual_seq_len_tensor: torch.Tensor,
    alpha: float,
    invalid_attn_mask_type: str,
    attn_scale_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable grouped ragged HSTU self-attention:
    :func:`grouped_ragged_hstu_attention_fwd` with autograd. Pass the same
    tensor as ``group_k`` and ``group_v`` for shared K/V."""
    return grouped_ragged_hstu_attention_fwd(
        Z,
        group_q,
        group_k,
        group_v,
        seq_offsets,
        group_offsets,
        group_num_targets,
        max_seq_len_tensor,
        max_attn_len_tensor,
        full_attn_size_tensor,
        contextual_seq_len_tensor,
        alpha,
        invalid_attn_mask_type,
        attn_scale_tensor,
    )
