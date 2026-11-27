# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Pure-PyTorch definition of jagged HSTU attention.

This module is the semantic oracle for the HSTU attention ops: every public
test compares against it, and it is the correctness fallback of the MSLK path.
It favors readability over speed. Each sequence is attended to on its own (no
padding), in fp32 (fp64 inputs stay fp64), and autograd provides the backward.

Jagged layout: B sequences are packed along the token axis as
``[total_tokens, H, D]``; ``seq_offsets`` (``[B + 1]``) holds the cumulative
starts, so sequence ``b`` is rows ``seq_offsets[b]:seq_offsets[b + 1]``.

Ragged masking and activation follow the public TLX gfx950 kernel
(facebookexperimental/triton, ``third_party/tlx/tutorials/hstu_self_attn/
tlx_gfx950_ragged_hstu_attention.py``: the ``invalid_mask`` construction and
``ATTN_SCALE_TYPE``), including its argument restrictions. ``upper_triangular``
is the mirror image of ``lower_triangular``. Cross attention follows
``tlx_gfx950_cross_attn_fwd`` and ``pytorch_hstu_cross_mha``
(``third_party/tlx/ops/kernels/hstu_cross_attn``) with ``targets_in_kv=False``.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

MASK_TYPES = ("lower_triangular", "upper_triangular", "none")
TRUNCATE_METHODS = ("none", "keep_first", "keep_last")

Scale = Union[float, torch.Tensor]


def ragged_hstu_valid_mask(
    seq_len: int,
    invalid_attn_mask_type: str = "lower_triangular",
    num_targets: Optional[int] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    full_attn_size: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """``[seq_len, seq_len]`` bool mask of one sequence; True where query m sees key n.

    Positions are remapped before the rules apply:

    * contextual prefix: every position shifts down by ``contextual_seq_len - 1``
      and clamps at 0 (the prefix collapses to position 0); ``max_ids`` shrinks
      by the same amount.
    * targets (``num_targets`` not None): ``max_ids -= num_targets`` and positions
      clamp to ``max_ids``, so the targets share one position and never see
      each other.

    A pair is valid on the (raw) diagonal, or when, on the remapped positions:

    * ``lower_triangular``: ``n < m``; with a window only
      ``m - max_attn_len <= n``. With a window and ``full_attn_size``, rows
      ``m >= max_ids - full_attn_size`` see every earlier key.
    * ``upper_triangular``: ``n > m``.
    * ``none``: ``n < max_ids``; with a window only ``|n - m| <= max_attn_len``.
    * contextual: row 0 sees every key below ``max_ids``, and every row sees
      column 0.
    """
    raw = torch.arange(seq_len, device=device)
    diagonal = raw[:, None] == raw[None, :]
    pos = raw
    max_ids = seq_len
    if contextual_seq_len > 0:
        pos = (pos - contextual_seq_len + 1).clamp(min=0)
        max_ids -= contextual_seq_len - 1
    if num_targets is not None:
        max_ids -= num_targets
        pos = pos.clamp(max=max_ids)
    m, n = pos[:, None], pos[None, :]

    if invalid_attn_mask_type == "lower_triangular":
        visible = n < m
        if max_attn_len > 0:
            visible = visible & (n >= m - max_attn_len)
            if full_attn_size > 0:
                visible = visible | ((m >= max_ids - full_attn_size) & (n < m))
    elif invalid_attn_mask_type == "upper_triangular":
        visible = n > m
    else:
        visible = n < max_ids
        if max_attn_len > 0:
            visible = visible & ((n - m).abs() <= max_attn_len)

    if contextual_seq_len > 0:
        visible = visible | ((m == 0) & (n < max_ids))
        visible = visible | (n == 0)
    return diagonal | visible


def cross_hstu_valid_mask(
    len_q: int,
    len_kv: int,
    num_targets: int = 0,
    causal: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """``[len_q, len_kv]`` bool mask of one cross-attention sequence.

    Non-causal: every query sees every (kept) key. Causal is bottom-right
    aligned against the non-target queries: with ``uih_len_q = len_q -
    num_targets``, query i sees key j when ``i + len_kv - uih_len_q >= j``, so
    the last history query sees the whole KV and target queries see all of it.
    """
    if not causal:
        return torch.ones(len_q, len_kv, dtype=torch.bool, device=device)
    uih_len_q = len_q - num_targets
    i = torch.arange(len_q, device=device)[:, None]
    j = torch.arange(len_kv, device=device)[None, :]
    return i + len_kv - uih_len_q >= j


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid: torch.Tensor,
    alpha: float,
    scale: Scale,
    num_softmax_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One sequence. q ``[n_q, H, D]``, k ``[n_kv, H, D]``, v ``[n_kv, H, V]``.

    Returns the output ``[n_q, H, V]`` and the base-2 LSE
    ``[n_q, num_softmax_heads]``. Softmax rows with no valid key output 0 and
    have an LSE of -inf.
    """
    scores = alpha * torch.einsum("qhd,khd->hqk", q, k)
    softmax_scores = scores[:num_softmax_heads]
    silu_scores = scores[num_softmax_heads:]

    # Empty rows take finite logits so neither forward nor backward makes NaNs.
    row_has_key = valid.any(dim=-1, keepdim=True)
    fill = torch.where(row_has_key, float("-inf"), 0.0).to(scores.dtype)
    logits = torch.where(valid, softmax_scores, fill)
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(logits - lse[..., None]) * valid
    lse2 = torch.where(row_has_key.squeeze(-1), lse / math.log(2.0), float("-inf"))

    if isinstance(scale, torch.Tensor) and scale.ndim == 1:
        scale = scale.view(1, -1, 1)
    silu = torch.where(valid, F.silu(silu_scores) * scale, 0.0)

    weights = torch.cat([probs, silu])
    out = torch.einsum("hqk,khv->qhv", weights, v)
    return out, lse2.transpose(0, 1)


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.promote_types(x.dtype, torch.float32)


def _check_offsets(name: str, seq_offsets: torch.Tensor, total: int) -> List[int]:
    if seq_offsets.ndim != 1 or seq_offsets.numel() < 2:
        raise ValueError(
            f"{name} must be 1-D with at least 2 entries, "
            f"got shape {tuple(seq_offsets.shape)}"
        )
    if seq_offsets.is_floating_point() or seq_offsets.is_complex():
        raise ValueError(f"{name} must be an integer tensor, got {seq_offsets.dtype}")
    offsets = seq_offsets.tolist()
    if offsets[0] != 0 or offsets[-1] != total:
        raise ValueError(
            f"{name} must start at 0 and end at {total}, "
            f"got {offsets[0]} and {offsets[-1]}"
        )
    if any(e < s for s, e in zip(offsets[:-1], offsets[1:])):
        raise ValueError(f"{name} must be non-decreasing")
    return offsets


def _check_targets(
    num_targets: Optional[torch.Tensor], lengths: List[int], name: str = "num_targets"
) -> List[Optional[int]]:
    if num_targets is None:
        return [None] * len(lengths)
    if num_targets.ndim != 1 or num_targets.numel() != len(lengths):
        raise ValueError(
            f"{name} must have shape [{len(lengths)}], got {tuple(num_targets.shape)}"
        )
    targets = num_targets.tolist()
    for b, (t, n) in enumerate(zip(targets, lengths)):
        if not 0 <= t <= n:
            raise ValueError(f"{name}[{b}]={t} must be in [0, {n}]")
    return targets


def _check_dynamic_scale(attn_scale: torch.Tensor, total: int, what: str) -> None:
    if attn_scale.ndim != 1 or attn_scale.numel() != total:
        raise ValueError(
            f"attn_scale must be a 0-dim tensor or have one value per {what} token "
            f"(shape [{total}]), got {tuple(attn_scale.shape)}"
        )


def _check_mask_options(
    invalid_attn_mask_type: str,
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
) -> None:
    if invalid_attn_mask_type not in MASK_TYPES:
        raise ValueError(
            f"invalid_attn_mask_type must be one of {MASK_TYPES}, "
            f"got {invalid_attn_mask_type!r}"
        )
    for name, value in (
        ("max_attn_len", max_attn_len),
        ("contextual_seq_len", contextual_seq_len),
        ("full_attn_size", full_attn_size),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if invalid_attn_mask_type != "lower_triangular" and (
        contextual_seq_len > 0 or full_attn_size > 0
    ):
        raise ValueError(
            "contextual_seq_len and full_attn_size require "
            f"invalid_attn_mask_type='lower_triangular', got {invalid_attn_mask_type!r}"
        )
    if invalid_attn_mask_type == "upper_triangular" and max_attn_len > 0:
        raise ValueError(
            "max_attn_len is not supported with "
            "invalid_attn_mask_type='upper_triangular'"
        )


def _check_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shared_kv: bool,
    same_tokens: bool,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k and v must be [tokens, heads, dim] tensors")
    if q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k head dims differ: {q.shape[2]} != {k.shape[2]}")
    if k.shape[:2] != v.shape[:2]:
        raise ValueError(
            "k and v must match in tokens and heads, "
            f"got {tuple(k.shape)} and {tuple(v.shape)}"
        )
    if same_tokens and q.shape[:2] != k.shape[:2]:
        raise ValueError(
            "q and k must match in tokens and heads, "
            f"got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if shared_kv and v is not k:
        raise ValueError("shared_kv=True requires v to be k")


def _ragged_forward(
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: List[int],
    targets: List[Optional[int]],
    invalid_attn_mask_type: str,
    seq_options: List[Tuple[int, int, int, Scale]],
    num_softmax_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence loop; ``seq_options[b]`` is sequence b's
    ``(max_attn_len, contextual_seq_len, full_attn_size, scale)``."""
    dtype = _compute_dtype(q)
    qc, kc, vc = q.to(dtype), k.to(dtype), v.to(dtype)
    outs, lses = [], []
    for b, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        max_attn_len, contextual_seq_len, full_attn_size, scale = seq_options[b]
        valid = ragged_hstu_valid_mask(
            end - start,
            invalid_attn_mask_type,
            targets[b],
            max_attn_len,
            contextual_seq_len,
            full_attn_size,
            device=q.device,
        )
        out, lse = _attend(
            qc[start:end],
            kc[start:end],
            vc[start:end],
            valid,
            alpha,
            scale,
            num_softmax_heads,
        )
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs).to(q.dtype), torch.cat(lses).to(torch.float32)


def ragged_hstu_attention_ref(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str = "lower_triangular",
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    full_attn_size: int = 0,
    num_softmax_heads: int = 0,
    shared_kv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ragged HSTU self-attention.

    Args:
        N: max sequence length; every sequence must fit. The default SiLU scale
            is ``1 / N``.
        alpha: score multiplier, ``s = alpha * q @ k^T``.
        q, k: ``[total_tokens, H, D]``. v: ``[total_tokens, H, V]``.
        seq_offsets: ``[B + 1]`` cumulative sequence starts.
        invalid_attn_mask_type: ``lower_triangular``, ``upper_triangular`` or
            ``none``; see :func:`ragged_hstu_valid_mask`.
        num_targets: ``[B]`` number of trailing target tokens per sequence.
        attn_scale: SiLU scale. ``None`` is ``1 / N``; a 0-dim tensor is one
            scalar; a ``[total_tokens]`` tensor is one scale per query token.
        max_attn_len: window length; 0 disables the window.
        contextual_seq_len: length of the contextual prefix;
            ``lower_triangular`` only.
        full_attn_size: rows before the targets that ignore the window;
            ``lower_triangular`` only.
        num_softmax_heads: heads ``[0, num_softmax_heads)`` use softmax, the
            rest SiLU.
        shared_kv: V is K; requires ``v is k``. Its gradient is the combined
            K-score + V-value gradient, so take gradients with respect to
            ``(q, k)``.

    Returns:
        ``(out, M)``: ``out`` is ``[total_tokens, H, V]`` in ``q.dtype``; ``M`` is
        the fp32 base-2 LSE ``log2(sum(exp(s)))`` over the valid keys of each
        softmax head, ``[total_tokens, num_softmax_heads]``.
    """
    _check_qkv(q, k, v, shared_kv, same_tokens=True)
    total, heads = q.shape[0], q.shape[1]
    offsets = _check_offsets("seq_offsets", seq_offsets, total)
    lengths = [e - s for s, e in zip(offsets[:-1], offsets[1:])]
    if max(lengths) > N:
        raise ValueError(
            f"every sequence must have length <= N={N}, got {max(lengths)}"
        )
    targets = _check_targets(num_targets, lengths)
    _check_mask_options(
        invalid_attn_mask_type, max_attn_len, contextual_seq_len, full_attn_size
    )
    if not 0 <= num_softmax_heads <= heads:
        raise ValueError(
            f"num_softmax_heads must be in [0, {heads}], got {num_softmax_heads}"
        )
    if attn_scale is not None and attn_scale.ndim != 0:
        _check_dynamic_scale(attn_scale, total, "query")

    seq_options = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        if attn_scale is None:
            scale: Scale = 1.0 / N
        elif attn_scale.ndim == 0:
            scale = attn_scale.to(_compute_dtype(q))
        else:
            scale = attn_scale[start:end].to(_compute_dtype(q))
        seq_options.append((max_attn_len, contextual_seq_len, full_attn_size, scale))
    return _ragged_forward(
        alpha,
        q,
        k,
        k if shared_kv else v,
        offsets,
        targets,
        invalid_attn_mask_type,
        seq_options,
        num_softmax_heads,
    )


def cross_hstu_attention_ref(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: Optional[torch.Tensor],
    G: int,
    shared_kv: bool = False,
    num_softmax_heads: int = 0,
    truncate_method: str = "none",
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Jagged HSTU cross attention: queries and keys/values have separate offsets.

    Args:
        max_seq_len: KV length cap used by ``truncate_method``; the default SiLU
            scale is ``1 / max_seq_len``.
        alpha: score multiplier.
        q: ``[total_q, H, D]``. k: ``[total_kv, H // G, D]``.
            v: ``[total_kv, H // G, V]``.
        seq_offsets: ``[B + 1]`` KV offsets.
        seq_offsets_q: ``[B + 1]`` query offsets.
        max_q_len: every query sequence must have length <= ``max_q_len``.
        attn_scale: SiLU scale; ``None`` is ``1 / max_seq_len``, a 0-dim tensor is
            one scalar, a ``[total_q]`` tensor is one scale per query token.
        G: query heads per KV head (GQA); query head h reads KV head ``h // G``.
        shared_kv: V is K; requires ``v is k``.
        num_softmax_heads: heads ``[0, num_softmax_heads)`` use softmax, the
            rest SiLU.
        truncate_method: ``none``, or ``keep_first`` / ``keep_last`` to keep the
            first or last ``max_seq_len`` KV tokens of each sequence. Dropped
            tokens get zero gradient.
        num_targets: ``[B]`` trailing target queries per sequence; only used
            when ``causal``.
        causal: bottom-right aligned causal mask; see
            :func:`cross_hstu_valid_mask`.

    Returns:
        ``(out, M)`` as in :func:`ragged_hstu_attention_ref`, over query tokens.
    """
    _check_qkv(q, k, v, shared_kv, same_tokens=False)
    total_q, heads = q.shape[0], q.shape[1]
    if G < 1 or heads % G != 0 or k.shape[1] != heads // G:
        raise ValueError(
            f"k must have H // G heads, got H={heads}, G={G}, k heads={k.shape[1]}"
        )
    offsets_kv = _check_offsets("seq_offsets", seq_offsets, k.shape[0])
    offsets_q = _check_offsets("seq_offsets_q", seq_offsets_q, total_q)
    if len(offsets_kv) != len(offsets_q):
        raise ValueError(
            "seq_offsets and seq_offsets_q must describe the same batch, "
            f"got {len(offsets_kv) - 1} and {len(offsets_q) - 1} sequences"
        )
    lengths_q = [e - s for s, e in zip(offsets_q[:-1], offsets_q[1:])]
    if max(lengths_q) > max_q_len:
        raise ValueError(
            f"every query sequence must have length <= max_q_len={max_q_len}, "
            f"got {max(lengths_q)}"
        )
    if truncate_method not in TRUNCATE_METHODS:
        raise ValueError(
            f"truncate_method must be one of {TRUNCATE_METHODS}, "
            f"got {truncate_method!r}"
        )
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
    targets = _check_targets(num_targets, lengths_q)
    if not 0 <= num_softmax_heads <= heads:
        raise ValueError(
            f"num_softmax_heads must be in [0, {heads}], got {num_softmax_heads}"
        )
    if attn_scale is not None and attn_scale.ndim != 0:
        _check_dynamic_scale(attn_scale, total_q, "query")

    dtype = _compute_dtype(q)
    qc, kc = q.to(dtype), k.to(dtype)
    vc = kc if shared_kv else v.to(dtype)
    outs, lses = [], []
    for b in range(len(offsets_q) - 1):
        q_start, q_end = offsets_q[b], offsets_q[b + 1]
        kv_start, kv_end = offsets_kv[b], offsets_kv[b + 1]
        if truncate_method != "none":
            kept = min(kv_end - kv_start, max_seq_len)
            if truncate_method == "keep_last":
                kv_start = kv_end - kept
            else:
                kv_end = kv_start + kept
        if attn_scale is None:
            scale: Scale = 1.0 / max_seq_len
        elif attn_scale.ndim == 0:
            scale = attn_scale.to(dtype)
        else:
            scale = attn_scale[q_start:q_end].to(dtype)
        valid = cross_hstu_valid_mask(
            q_end - q_start,
            kv_end - kv_start,
            (targets[b] or 0) if causal else 0,
            causal,
            device=q.device,
        )
        out, lse = _attend(
            qc[q_start:q_end],
            kc[kv_start:kv_end].repeat_interleave(G, dim=1),
            vc[kv_start:kv_end].repeat_interleave(G, dim=1),
            valid,
            alpha,
            scale,
            num_softmax_heads,
        )
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs).to(q.dtype), torch.cat(lses).to(torch.float32)


def grouped_ragged_hstu_attention_ref(
    Z: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    group_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_seq_len_tensor: torch.Tensor,
    max_attn_len_tensor: torch.Tensor,
    full_attn_size_tensor: torch.Tensor,
    contextual_seq_len_tensor: torch.Tensor,
    alpha: float,
    invalid_attn_mask_type: str,
    attn_scale_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Grouped ragged HSTU self-attention; every head uses SiLU.

    ``G = max_seq_len_tensor.numel()`` groups of ``Z`` sequences share one packed
    tensor. Group g applies the ragged semantics of
    :func:`ragged_hstu_attention_ref` to sequences ``g*Z ... g*Z + Z - 1`` with its
    own options; groups attend independently and share Z, H and head dims.

    Args:
        Z: sequences per group.
        q, k, v, num_targets, alpha, invalid_attn_mask_type: as in
            :func:`ragged_hstu_attention_ref`.
        seq_offsets: ``[G * Z + 1]`` cumulative token offsets of every sequence.
        group_offsets: ``[G + 1]`` token offset of each group's first sequence;
            must equal ``seq_offsets[::Z]``.
        max_seq_len_tensor: ``[G]`` per-group max length; every sequence of
            group g must fit.
        max_attn_len_tensor, full_attn_size_tensor, contextual_seq_len_tensor:
            ``[G]`` per-group values.
        attn_scale_tensor: ``None`` is ``1 / max_seq_len_tensor[g]``; ``[G]`` is
            one scale per group; ``[total_tokens]`` is one scale per query token.
            ``[G]`` wins when ``G == total_tokens``.

    Returns:
        ``[total_tokens, H, V]`` in ``q.dtype``.
    """
    _check_qkv(q, k, v, shared_kv=False, same_tokens=True)
    total = q.shape[0]
    if max_seq_len_tensor.ndim != 1 or max_seq_len_tensor.numel() < 1:
        raise ValueError(
            "max_seq_len_tensor must be 1-D with one entry per group, "
            f"got shape {tuple(max_seq_len_tensor.shape)}"
        )
    num_groups = max_seq_len_tensor.numel()
    offsets = _check_offsets("seq_offsets", seq_offsets, total)
    batch = len(offsets) - 1
    if Z < 1 or batch != num_groups * Z:
        raise ValueError(
            f"seq_offsets must describe G * Z = {num_groups} * {Z} sequences, "
            f"got {batch}"
        )
    if _check_offsets("group_offsets", group_offsets, total) != offsets[::Z]:
        raise ValueError("group_offsets must equal seq_offsets[::Z]")

    def per_group(name: str, values: torch.Tensor) -> List[int]:
        if values.ndim != 1 or values.numel() != num_groups:
            raise ValueError(
                f"{name} must have shape [{num_groups}] (one per group), "
                f"got {tuple(values.shape)}"
            )
        if values.is_floating_point() or values.is_complex():
            raise ValueError(f"{name} must be an integer tensor, got {values.dtype}")
        return values.tolist()

    max_seq_lens = per_group("max_seq_len_tensor", max_seq_len_tensor)
    windows = per_group("max_attn_len_tensor", max_attn_len_tensor)
    full_sizes = per_group("full_attn_size_tensor", full_attn_size_tensor)
    contexts = per_group("contextual_seq_len_tensor", contextual_seq_len_tensor)
    for g in range(num_groups):
        _check_mask_options(
            invalid_attn_mask_type, windows[g], contexts[g], full_sizes[g]
        )

    lengths = [e - s for s, e in zip(offsets[:-1], offsets[1:])]
    for b, n in enumerate(lengths):
        g = b // Z
        if n > max_seq_lens[g]:
            raise ValueError(
                f"sequence {b} has length {n} > "
                f"max_seq_len_tensor[{g}]={max_seq_lens[g]}"
            )
    targets = _check_targets(num_targets, lengths)
    if attn_scale_tensor is not None and (
        attn_scale_tensor.ndim != 1
        or attn_scale_tensor.numel() not in (num_groups, total)
    ):
        raise ValueError(
            f"attn_scale_tensor must have shape [{num_groups}] (per group) or "
            f"[{total}] (per token), got {tuple(attn_scale_tensor.shape)}"
        )
    per_group_scale = (
        attn_scale_tensor is not None and attn_scale_tensor.numel() == num_groups
    )

    seq_options = []
    for b, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        g = b // Z
        if attn_scale_tensor is None:
            scale: Scale = 1.0 / max_seq_lens[g]
        elif per_group_scale:
            scale = attn_scale_tensor[g].to(_compute_dtype(q))
        else:
            scale = attn_scale_tensor[start:end].to(_compute_dtype(q))
        seq_options.append((windows[g], contexts[g], full_sizes[g], scale))
    out, _ = _ragged_forward(
        alpha,
        q,
        k,
        v,
        offsets,
        targets,
        invalid_attn_mask_type,
        seq_options,
        0,
    )
    return out
