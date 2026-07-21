# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL flash-attention backend for MSLK (ROCm / gfx950).

Optional ROCm-only backend (install with ``pip install mslk[flydsl]``). The
gfx950 DUALWAVE_SWP forward kernel (with a generic gfx942 fallback) is exposed
through :func:`flydsl_flash_attn_func`, which optionally returns the per-row
log-sum-exp (LSE) needed by a backward pass.

The kernel authoring modules live in the vendored ``_kernels`` subpackage
(not shipped in the ``flydsl`` wheel); this module is a thin, availability-gated
wrapper so importing ``mslk.attention.flydsl`` never fails on a build without
FlyDSL or on an unsupported GPU arch.
"""

from typing import Any

import torch
from mslk.utils.flydsl import is_flydsl_available, require_flydsl

__all__ = ["flydsl_flash_attn_func", "is_flydsl_available"]


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs: Any,
) -> Any:
    """FlyDSL flash attention forward (gfx950 DUALWAVE_SWP / gfx942 generic).

    Thin wrapper over the vendored FlyDSL interface. Raises a ``RuntimeError``
    with an install hint when FlyDSL is unavailable, then dispatches to the
    kernel (compiled once per static config and cached).

    Args:
        q: Query tensor. Dense: ``[B, Sq, H, D]`` (BSHD).
            Varlen: ``[total_q, H, D]`` (packed; ``cu_seqlens_q`` required).
        k: Key tensor. Dense: ``[B, Skv, Hkv, D]``. Varlen: ``[total_kv, Hkv, D]``.
        v: Value tensor, same layout as ``k``.
        **kwargs: Forwarded to the FlyDSL interface. Common options:
            ``causal`` (bool), ``num_kv_heads`` (int, GQA/MQA),
            ``num_kv_splits`` (int, split-K; gfx950, D in {64,128}, bf16/f16,
            seq_len >= 384), the varlen controls ``cu_seqlens_q`` /
            ``cu_seqlens_kv`` / ``max_seqlen_q`` / ``max_seqlen_kv`` /
            ``cross_seqlen``, ``out`` (pre-allocated output), and ``return_lse``.

            ``return_lse=True`` returns ``(out, lse)`` where ``lse`` is an fp32
            tensor of shape ``[B, H, Sq]`` (varlen: ``[B, H, max_seqlen_q]``,
            padded) holding the per-row **natural-log, scale-folded**
            log-sum-exp ``LSE_i = ln(sum_j exp(sm_scale * (q_i . k_j)))`` over
            visible keys; fully-masked rows store ``-inf``. Not supported for
            fp8 inputs. This is the convention a FlyDSL flash-attention backward
            pass consumes.

    Returns:
        ``out`` with the same shape as ``q`` (dtype bf16 for fp8 inputs, else
        ``q.dtype``), or ``(out, lse)`` when ``return_lse=True``.
    """
    require_flydsl()
    from ._kernels.flash_attn_interface import flydsl_flash_attn_func as _impl

    return _impl(q, k, v, **kwargs)
