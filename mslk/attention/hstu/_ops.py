# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""torch.library schemas, fake impls and device impls for HSTU attention ops.

Six ops: forward and backward of ragged, cross and grouped HSTU attention.

- On ROCm the device impl calls the op's routing in
  ``mslk/attention/dispatch/hstu_*.py``, which picks MSLK's kernel or another
  backend per shape. Elsewhere it calls MSLK's backend directly.
- Both the device impl and the fake impl validate the arguments, so errors
  surface in eager mode, on meta tensors and under torch.compile alike.
- Importing this module imports no backend (FlyDSL, AITER, Triton); the
  dispatch module is imported on the first call.

Routing is decided at run time, on the first call of a shape: that call reads
the device properties and the routing table and may import and JIT-compile a
backend. Warm up before CUDA-graph capture; a captured graph keeps the backend
chosen at capture time.
"""

import importlib
from typing import Any, List, Optional, Tuple

import torch
from mslk.attention.hstu._validation import (
    validate_cross_bwd,
    validate_cross_fwd,
    validate_grouped_bwd,
    validate_grouped_fwd,
    validate_ragged_bwd,
    validate_ragged_fwd,
)

_IS_ROCM: bool = torch.version.hip is not None

_lib = torch.library.Library("mslk", "FRAGMENT")


def _route(op_module: str, direction: str, q: torch.Tensor, **kwargs: Any) -> Any:
    """Serve a call through ``mslk.attention.dispatch.<op_module>``."""
    module = importlib.import_module(f"mslk.attention.dispatch.{op_module}")
    routing = getattr(module, f"{direction}_ROUTING")
    if _IS_ROCM:
        return routing(q, **kwargs)
    from mslk.aiter.routing import MSLK

    return routing.backends[MSLK](q, **kwargs)


def _new_out(like: torch.Tensor, shape: Any) -> torch.Tensor:
    return torch.empty(tuple(shape), dtype=like.dtype, device=like.device)


def _new_lse(q: torch.Tensor) -> torch.Tensor:
    # M: [T, H] float32 for every backend (see mslk.attention.dispatch).
    return torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)


# Every impl below starts with ``args = dict(locals())``: at that point the
# locals are exactly the op's arguments, keyed by their schema names.

# ═══════════════════════════════════════════════════════════════════════════
# Ragged HSTU self-attention
# ═══════════════════════════════════════════════════════════════════════════

_lib.define(
    "hstu_ragged_fwd("
    "int N, float alpha, "
    "Tensor q, Tensor k, Tensor v, "
    "Tensor seq_offsets, "
    "str invalid_attn_mask_type, "
    "Tensor? num_targets, Tensor? attn_scale, "
    "int max_attn_len, int contextual_seq_len, int full_attn_size, "
    "int num_softmax_heads=0, bool shared_kv=False"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "hstu_ragged_bwd("
    "Tensor dout, "
    "Tensor q, Tensor k, Tensor v, "
    "Tensor(a!) dq, Tensor(b!) dk, Tensor(c!)? dv, "
    "Tensor seq_offsets, "
    "Tensor? num_targets, Tensor? attn_scale, "
    "int N, float alpha, "
    "int max_attn_len, str invalid_attn_mask_type, "
    "int contextual_seq_len, "
    "Tensor? sort_by_length_indices, "
    "int full_attn_size, "
    "int num_softmax_heads=0, "
    "Tensor? M=None, Tensor? Out=None, "
    "bool shared_kv=False"
    ") -> ()"
)


def _hstu_ragged_fwd(
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
    args = dict(locals())
    validate_ragged_fwd(**args)
    return _route("hstu_ragged", "FWD", **args)


def _hstu_ragged_fwd_fake(
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
    validate_ragged_fwd(**dict(locals()))
    return _new_out(v, v.shape), _new_lse(q)


def _hstu_ragged_bwd(
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
    args = dict(locals())
    validate_ragged_bwd(**args)
    _route("hstu_ragged", "BWD", **args)


def _hstu_ragged_bwd_fake(
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
    validate_ragged_bwd(**dict(locals()))


_lib.impl("hstu_ragged_fwd", _hstu_ragged_fwd, "CUDA")
_lib.impl("hstu_ragged_bwd", _hstu_ragged_bwd, "CUDA")
torch.library.register_fake("mslk::hstu_ragged_fwd")(_hstu_ragged_fwd_fake)
torch.library.register_fake("mslk::hstu_ragged_bwd")(_hstu_ragged_bwd_fake)


# ═══════════════════════════════════════════════════════════════════════════
# Cross HSTU attention
# ═══════════════════════════════════════════════════════════════════════════

_lib.define(
    "hstu_cross_fwd("
    "int max_seq_len, float alpha, "
    "Tensor q, Tensor k, Tensor? v, "
    "Tensor seq_offsets, Tensor seq_offsets_q, "
    "int max_q_len, Tensor attn_scale, int G, "
    "bool shared_kv=False, int num_softmax_heads=0, "
    'bool enable_tma=True, str truncate_method="none", '
    "Tensor? num_targets=None, bool causal=False"
    ") -> (Tensor, Tensor)"
)

# Returns [dq, dk] when shared_kv (dk holds dK+dV), else [dq, dk, dv]: an op
# may not return two aliasing outputs, so the reference's ``dv = dk`` alias is
# made by the Python wrapper (cross_hstu_attention_bwd), not by the op.
_lib.define(
    "hstu_cross_bwd("
    "Tensor dout, "
    "Tensor q, Tensor k, Tensor? v, "
    "Tensor seq_offsets, Tensor attn_scale, "
    "int max_seq_len, float alpha, "
    "int? max_q_len, Tensor? seq_offsets_q, "
    "Tensor? num_targets, "
    "bool causal, bool shared_kv, int G, "
    "int num_softmax_heads=0, "
    "Tensor? M=None, Tensor? Delta=None, "
    "int stride_mm=0, "
    'str truncate_method="none", bool enable_tma=True'
    ") -> Tensor[]"
)


def _cross_value(
    k: torch.Tensor, v: Optional[torch.Tensor], shared_kv: bool
) -> torch.Tensor:
    return k if (shared_kv or v is None) else v


def _hstu_cross_fwd(
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
    args = dict(locals())
    validate_cross_fwd(**args)
    return _route("hstu_cross", "FWD", **args)


def _hstu_cross_fwd_fake(
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
    validate_cross_fwd(**dict(locals()))
    value = _cross_value(k, v, shared_kv)
    out = _new_out(value, (q.shape[0], q.shape[1], value.shape[2]))
    return out, _new_lse(q)


def _hstu_cross_bwd(
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
) -> List[torch.Tensor]:
    args = dict(locals())
    validate_cross_bwd(**args)
    return list(_route("hstu_cross", "BWD", **args))


def _hstu_cross_bwd_fake(
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
) -> List[torch.Tensor]:
    validate_cross_bwd(**dict(locals()))
    grads = [_new_out(q, q.shape), _new_out(k, k.shape)]
    if not shared_kv:
        assert v is not None
        grads.append(_new_out(v, v.shape))
    return grads


_lib.impl("hstu_cross_fwd", _hstu_cross_fwd, "CUDA")
_lib.impl("hstu_cross_bwd", _hstu_cross_bwd, "CUDA")
torch.library.register_fake("mslk::hstu_cross_fwd")(_hstu_cross_fwd_fake)
torch.library.register_fake("mslk::hstu_cross_bwd")(_hstu_cross_bwd_fake)


# ═══════════════════════════════════════════════════════════════════════════
# Grouped ragged HSTU attention
# ═══════════════════════════════════════════════════════════════════════════

_lib.define(
    "hstu_grouped_fwd("
    "int Z, "
    "Tensor q, Tensor k, Tensor v, "
    "Tensor seq_offsets, Tensor group_offsets, "
    "Tensor? num_targets, "
    "Tensor max_seq_len_tensor, Tensor max_attn_len_tensor, "
    "Tensor full_attn_size_tensor, Tensor contextual_seq_len_tensor, "
    "float alpha, str invalid_attn_mask_type, "
    "Tensor? attn_scale_tensor=None"
    ") -> Tensor"
)

_lib.define(
    "hstu_grouped_bwd("
    "Tensor dout, int Z, "
    "Tensor q, Tensor k, Tensor v, "
    "Tensor(a!) dq, Tensor(b!) dk, Tensor(c!)? dv, "
    "Tensor seq_offsets, Tensor group_offsets, "
    "Tensor? num_targets, "
    "Tensor max_seq_len_tensor, Tensor max_attn_len_tensor, "
    "Tensor full_attn_size_tensor, Tensor contextual_seq_len_tensor, "
    "float alpha, str invalid_attn_mask_type, "
    "Tensor? attn_scale_tensor=None, "
    "Tensor? Out=None, "
    "bool shared_kv=False"
    ") -> ()"
)


def _hstu_grouped_fwd(
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
    args = dict(locals())
    validate_grouped_fwd(**args)
    return _route("hstu_grouped", "FWD", **args)


def _hstu_grouped_fwd_fake(
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
    validate_grouped_fwd(**dict(locals()))
    return _new_out(v, v.shape)


def _hstu_grouped_bwd(
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
    attn_scale_tensor: Optional[torch.Tensor] = None,
    Out: Optional[torch.Tensor] = None,
    shared_kv: bool = False,
) -> None:
    args = dict(locals())
    validate_grouped_bwd(**args)
    _route("hstu_grouped", "BWD", **args)


def _hstu_grouped_bwd_fake(
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
    attn_scale_tensor: Optional[torch.Tensor] = None,
    Out: Optional[torch.Tensor] = None,
    shared_kv: bool = False,
) -> None:
    validate_grouped_bwd(**dict(locals()))


_lib.impl("hstu_grouped_fwd", _hstu_grouped_fwd, "CUDA")
_lib.impl("hstu_grouped_bwd", _hstu_grouped_bwd, "CUDA")
torch.library.register_fake("mslk::hstu_grouped_fwd")(_hstu_grouped_fwd_fake)
torch.library.register_fake("mslk::hstu_grouped_bwd")(_hstu_grouped_bwd_fake)
