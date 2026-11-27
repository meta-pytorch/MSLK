# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Argument validation shared by every backend of the HSTU attention ops.

Runs in both the device implementation and the fake implementation, so bad
arguments are rejected eagerly, on meta tensors and while torch.compile traces.
Only metadata is inspected (shapes, dtypes, devices, strides, tensor identity);
nothing reads device memory, so validation is CUDA-graph safe.

Validators take the op's arguments by keyword, under the op's argument names.
"""

from typing import Optional, Sequence

import torch

# The references compute in bf16/fp16 and also take fp32 (their dot products
# honour the TF32 setting). A backend that cannot take a dtype declines it.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_INDEX_DTYPES = (torch.int32, torch.int64)
_MASK_TYPES = ("lower_triangular", "upper_triangular")


def _fail(message: str) -> None:
    raise ValueError(message)


def _check_dtype(t: torch.Tensor, name: str) -> None:
    if t.dtype not in _SUPPORTED_DTYPES:
        _fail(f"{name} must be bfloat16, float16 or float32, got {t.dtype}")


def _check_contiguous_last_dim(t: torch.Tensor, name: str) -> None:
    if t.dim() > 0 and t.shape[-1] > 1 and t.stride(-1) != 1:
        _fail(f"{name} must be contiguous in the last dimension")


def _check_same_device(t: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    if t.device != ref.device:
        _fail(f"{name} is on {t.device}, expected {ref.device} (the device of q)")


def _check_index(
    t: torch.Tensor,
    name: str,
    ref: torch.Tensor,
    length: Optional[int] = None,
    min_length: int = 0,
) -> None:
    """A 1-D integer metadata tensor on q's device."""
    if t.dtype not in _INDEX_DTYPES:
        _fail(f"{name} must be int32 or int64, got {t.dtype}")
    if t.dim() != 1:
        _fail(f"{name} must be 1-D, got {t.dim()}-D")
    if length is not None and t.shape[0] != length:
        _fail(f"{name} must have length {length}, got {t.shape[0]}")
    if t.shape[0] < min_length:
        _fail(f"{name} must have at least {min_length} entries, got {t.shape[0]}")
    _check_same_device(t, ref, name)


def _check_float(t: torch.Tensor, name: str, ref: torch.Tensor) -> None:
    if not t.dtype.is_floating_point:
        _fail(f"{name} must be a floating-point tensor, got {t.dtype}")
    _check_same_device(t, ref, name)


def _check_non_negative(**values: int) -> None:
    for name, value in values.items():
        if value < 0:
            _fail(f"{name} must be >= 0, got {value}")


def _check_mask_type(invalid_attn_mask_type: str) -> None:
    if invalid_attn_mask_type not in _MASK_TYPES:
        _fail(
            f"invalid_attn_mask_type must be one of {_MASK_TYPES}, "
            f"got {invalid_attn_mask_type!r}"
        )


def _check_softmax_heads(num_softmax_heads: int, H: int) -> None:
    if num_softmax_heads not in (0, H):
        _fail(
            f"num_softmax_heads must be 0 (SiLU) or H={H} (softmax), "
            f"got {num_softmax_heads}"
        )


def _check_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    *,
    self_attention: bool,
) -> None:
    """q, k: [T, H, Dqk]; v: [T_kv, H, Dv]. Same dtype and device."""
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t is None:
            continue
        if t.dim() != 3:
            _fail(f"{name} must be 3-D [tokens, heads, head_dim], got {t.dim()}-D")
        _check_dtype(t, name)
        _check_contiguous_last_dim(t, name)
        _check_same_device(t, q, name)
        if t.dtype != q.dtype:
            _fail(f"{name} has dtype {t.dtype}, but q has {q.dtype}")
    if k.shape[1] != q.shape[1] or k.shape[2] != q.shape[2]:
        _fail(
            f"k must have the heads and head dim of q {tuple(q.shape[1:])}, "
            f"got {tuple(k.shape[1:])}"
        )
    if self_attention and k.shape[0] != q.shape[0]:
        _fail(f"k must have the {q.shape[0]} tokens of q, got {k.shape[0]}")
    if v is not None and (v.shape[0] != k.shape[0] or v.shape[1] != k.shape[1]):
        _fail(
            f"v must have the tokens and heads of k {tuple(k.shape[:2])}, "
            f"got {tuple(v.shape[:2])}"
        )


def _check_shared_kv_fwd(k: torch.Tensor, v: Optional[torch.Tensor]) -> None:
    if v is not k:
        _fail("shared_kv=True requires v to be the same tensor object as k")


def _check_grads(
    *,
    shared_kv: bool,
    q: torch.Tensor,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
) -> None:
    """Gradient buffers written in place. With shared_kv the combined dK+dV
    goes to dk and dv must be None."""
    if shared_kv:
        _check_shared_kv_fwd(k, v)
        if dv is not None:
            _fail(
                "shared_kv=True: dv must be None; the dK+dV gradient is written to dk"
            )
    elif dv is None:
        _fail("shared_kv=False requires a dv buffer")
    if dq is dk or (dv is not None and (dv is dk or dv is dq)):
        _fail("dq, dk and dv must be distinct tensors")
    for name, g, ref in (("dq", dq, q), ("dk", dk, k), ("dv", dv, v)):
        if g is None:
            continue
        if g.shape != ref.shape or g.dtype != ref.dtype:
            _fail(
                f"{name} must match the shape and dtype of its input "
                f"{tuple(ref.shape)} {ref.dtype}, got {tuple(g.shape)} {g.dtype}"
            )
        _check_contiguous_last_dim(g, name)
        _check_same_device(g, q, name)


def _check_dout(
    dout: torch.Tensor, q: torch.Tensor, shape: Sequence[int], name: str = "dout"
) -> None:
    if tuple(dout.shape) != tuple(shape):
        _fail(f"{name} must have shape {tuple(shape)}, got {tuple(dout.shape)}")
    if dout.dtype != q.dtype:
        _fail(f"{name} has dtype {dout.dtype}, but q has {q.dtype}")
    _check_contiguous_last_dim(dout, name)
    _check_same_device(dout, q, name)


def _check_num_targets(
    num_targets: Optional[torch.Tensor], num_seqs: int, q: torch.Tensor
) -> None:
    if num_targets is not None:
        _check_index(num_targets, "num_targets", q, length=num_seqs)


# ── Ragged self-attention ────────────────────────────────────────────────


def validate_ragged_fwd(
    *,
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
    num_softmax_heads: int,
    shared_kv: bool,
) -> None:
    _check_qkv(q, k, v, self_attention=True)
    if shared_kv:
        _check_shared_kv_fwd(k, v)
    _check_index(seq_offsets, "seq_offsets", q, min_length=1)
    _check_num_targets(num_targets, seq_offsets.shape[0] - 1, q)
    if attn_scale is not None:
        _check_float(attn_scale, "attn_scale", q)
    _check_mask_type(invalid_attn_mask_type)
    _check_softmax_heads(num_softmax_heads, q.shape[1])
    _check_non_negative(
        N=N,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        full_attn_size=full_attn_size,
    )


def validate_ragged_bwd(
    *,
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
    num_softmax_heads: int,
    M: Optional[torch.Tensor],
    Out: Optional[torch.Tensor],
    shared_kv: bool,
) -> None:
    validate_ragged_fwd(
        N=N,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        invalid_attn_mask_type=invalid_attn_mask_type,
        num_targets=num_targets,
        attn_scale=attn_scale,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        full_attn_size=full_attn_size,
        num_softmax_heads=num_softmax_heads,
        shared_kv=shared_kv,
    )
    _check_grads(shared_kv=shared_kv, q=q, k=k, v=v, dq=dq, dk=dk, dv=dv)
    _check_dout(dout, q, v.shape)
    if Out is not None:
        _check_dout(Out, q, v.shape, "Out")
    if sort_by_length_indices is not None:
        _check_index(
            sort_by_length_indices,
            "sort_by_length_indices",
            q,
            length=seq_offsets.shape[0] - 1,
        )
    if M is not None:
        _check_same_device(M, q, "M")


# ── Cross attention ──────────────────────────────────────────────────────


def _cross_v(
    k: torch.Tensor, v: Optional[torch.Tensor], shared_kv: bool
) -> torch.Tensor:
    if shared_kv:
        if v is not None and v is not k:
            _fail("shared_kv=True requires v to be None or the same tensor as k")
        return k
    if v is None:
        _fail("shared_kv=False requires v")
    return v


def validate_cross_fwd(
    *,
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
    shared_kv: bool,
    num_softmax_heads: int,
    enable_tma: bool,
    truncate_method: str,
    num_targets: Optional[torch.Tensor],
    causal: bool,
) -> None:
    _check_qkv(q, k, v, self_attention=False)
    _cross_v(k, v, shared_kv)
    _check_index(seq_offsets, "seq_offsets", q, min_length=1)
    _check_index(seq_offsets_q, "seq_offsets_q", q, length=seq_offsets.shape[0])
    _check_num_targets(num_targets, seq_offsets.shape[0] - 1, q)
    _check_float(attn_scale, "attn_scale", q)
    _check_softmax_heads(num_softmax_heads, q.shape[1])
    _check_non_negative(max_seq_len=max_seq_len, max_q_len=max_q_len)
    if G < 1:
        _fail(f"G must be >= 1, got {G}")


def validate_cross_bwd(
    *,
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
    num_softmax_heads: int,
    M: Optional[torch.Tensor],
    Delta: Optional[torch.Tensor],
    stride_mm: int,
    truncate_method: str,
    enable_tma: bool,
) -> None:
    _check_qkv(q, k, v, self_attention=False)
    value = _cross_v(k, v, shared_kv)
    _check_index(seq_offsets, "seq_offsets", q, min_length=1)
    if seq_offsets_q is not None:
        _check_index(seq_offsets_q, "seq_offsets_q", q, length=seq_offsets.shape[0])
    _check_num_targets(num_targets, seq_offsets.shape[0] - 1, q)
    _check_float(attn_scale, "attn_scale", q)
    _check_softmax_heads(num_softmax_heads, q.shape[1])
    _check_non_negative(max_seq_len=max_seq_len, stride_mm=stride_mm)
    if max_q_len is not None:
        _check_non_negative(max_q_len=max_q_len)
    if G < 1:
        _fail(f"G must be >= 1, got {G}")
    _check_dout(dout, q, (q.shape[0], q.shape[1], value.shape[2]))
    for name, t in (("M", M), ("Delta", Delta)):
        if t is not None:
            _check_same_device(t, q, name)


# ── Grouped ragged self-attention ────────────────────────────────────────


def validate_grouped_fwd(
    *,
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
    attn_scale_tensor: Optional[torch.Tensor],
) -> None:
    _check_qkv(q, k, v, self_attention=True)
    if Z < 1:
        _fail(f"Z must be >= 1, got {Z}")
    _check_index(group_offsets, "group_offsets", q, min_length=2)
    G = group_offsets.shape[0] - 1
    _check_index(seq_offsets, "seq_offsets", q, length=G * Z + 1)
    _check_num_targets(num_targets, G * Z, q)
    for name, t in (
        ("max_seq_len_tensor", max_seq_len_tensor),
        ("max_attn_len_tensor", max_attn_len_tensor),
        ("full_attn_size_tensor", full_attn_size_tensor),
        ("contextual_seq_len_tensor", contextual_seq_len_tensor),
    ):
        _check_index(t, name, q, length=G)
    if attn_scale_tensor is not None:
        _check_float(attn_scale_tensor, "attn_scale_tensor", q)
        # The mode follows the launcher: attn_scale_tensor[0].ndim == 0 is
        # one scale per group, anything else is one per packed query token.
        if attn_scale_tensor.dim() == 0:
            _fail("attn_scale_tensor must have at least one dimension")
        if attn_scale_tensor.dim() == 1:
            if attn_scale_tensor.shape[0] != G:
                _fail(
                    f"scalar attn_scale_tensor must have one entry per group "
                    f"({G}), got {attn_scale_tensor.shape[0]}"
                )
        elif attn_scale_tensor.shape[0] != q.shape[0]:
            _fail(
                f"dynamic attn_scale_tensor must have one row per query token "
                f"({q.shape[0]}), got {attn_scale_tensor.shape[0]}"
            )
    _check_mask_type(invalid_attn_mask_type)


def validate_grouped_bwd(
    *,
    dout: torch.Tensor,
    Z: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    seq_offsets: torch.Tensor,
    group_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_seq_len_tensor: torch.Tensor,
    max_attn_len_tensor: torch.Tensor,
    full_attn_size_tensor: torch.Tensor,
    contextual_seq_len_tensor: torch.Tensor,
    alpha: float,
    invalid_attn_mask_type: str,
    attn_scale_tensor: Optional[torch.Tensor],
    Out: Optional[torch.Tensor],
    shared_kv: bool,
) -> None:
    validate_grouped_fwd(
        Z=Z,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        group_offsets=group_offsets,
        num_targets=num_targets,
        max_seq_len_tensor=max_seq_len_tensor,
        max_attn_len_tensor=max_attn_len_tensor,
        full_attn_size_tensor=full_attn_size_tensor,
        contextual_seq_len_tensor=contextual_seq_len_tensor,
        alpha=alpha,
        invalid_attn_mask_type=invalid_attn_mask_type,
        attn_scale_tensor=attn_scale_tensor,
    )
    _check_grads(shared_kv=shared_kv, q=q, k=k, v=v, dq=dq, dk=dk, dv=dv)
    _check_dout(dout, q, v.shape)
    if Out is not None:
        _check_dout(Out, q, v.shape, "Out")
