# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Portions of this file are adapted from AITER
# (https://github.com/ROCm/aiter,
# aiter/ops/flydsl/kernels/hstu/hstu_attention_common.py):
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# pyre-unsafe

"""Shared pieces of the FlyDSL HSTU attention kernels.

Host-side: MFMA/DMA geometry per architecture and the compile-time feature set
of a ragged launch. Device-side (call from inside a ``@flyc.kernel`` body):
the position remap, the validity mask and the KV column range of a query tile,
all following the T1 reference (``mslk/attention/hstu/_torch_ref.py``,
``ragged_hstu_valid_mask``), plus the SiLU gate and MFMA helpers.
"""

from __future__ import annotations

import math as host_math
from typing import NamedTuple

import flydsl.expr as fx  # pyre-ignore[21]
from flydsl.expr.typing import Vector as Vec  # pyre-ignore[21]

LOG2E: float = host_math.log2(host_math.e)

WARP_SIZE = 64
MFMA_M = 16
MFMA_N = 16
# Score fragments feed the sequence-axis GEMM directly, and their accumulator
# layout is the operand layout of a 16-deep MFMA, so that GEMM stays at K=16.
MFMA_K = 16
MFMA_LANE_K = 4
MFMA_ELEMS_PER_LANE = (MFMA_M * MFMA_N) // WARP_SIZE

MASK_TYPES = ("lower_triangular", "upper_triangular", "none")


class RaggedFeatures(NamedTuple):
    """Which ragged features a launch compiles in.

    The values of the window, contextual prefix and full-attention size are
    runtime arguments; only their presence specializes the kernel.
    """

    mask_type: str
    has_targets: bool
    has_window: bool
    has_contextual: bool
    has_full: bool

    def validate(self) -> None:
        if self.mask_type not in MASK_TYPES:
            raise ValueError(f"mask_type must be one of {MASK_TYPES}")
        if self.mask_type != "lower_triangular" and (
            self.has_contextual or self.has_full
        ):
            raise ValueError("contextual and full attention need lower_triangular")
        if self.mask_type == "upper_triangular" and self.has_window:
            raise ValueError("a window is not supported with upper_triangular")
        if self.has_full and not self.has_window:
            raise ValueError("full_attn_size only applies with a window")


def arch_mfma_params(arch: str) -> tuple[int, int]:
    """Dimension-axis ``(mfma_k, operand_elems_per_lane)``: CDNA3 uses the
    16x16x16 MFMA, CDNA4 the 16x16x32 one. Both produce four fp32 per lane."""
    mfma_k = 16 if arch.startswith("gfx942") else 32
    return mfma_k, (MFMA_M * mfma_k) // WARP_SIZE


def mfma_params_for_dim(dim: int, arch: str) -> tuple[int, int]:
    """The architecture's preferred MFMA depth when ``dim`` allows it."""
    preferred = arch_mfma_params(arch)
    if dim % preferred[0] == 0:
        return preferred
    return MFMA_K, MFMA_LANE_K


def arch_dma_params(arch: str) -> tuple[int, int, int]:
    """``(dma_elems, k_swz_rows, k_swz_shift)`` for 16-bit elements.

    K columns are XOR-swizzled off LDS banks:
    ``col ^ ((row & (rows - 1)) << shift)``. gfx942 has 32 banks and dword
    loads, gfx950 64 banks and dwordx4 loads. Both stay within a 64-column row.
    """
    if arch.startswith("gfx942"):
        return 2, 16, 2
    return 8, 8, 3


def exp2_f32(x):
    """Fast approximate base-2 exponential (lowers to ``v_exp_f32``)."""
    return fx.exp2(x, fastmath="afn")


def bind_mfma_accs(elem_dtype, *shapes):
    """One ``acc(a_pack, b_pack, c) -> c'`` per ``(mfma_k, lane_k)``, sharing
    atoms between equal shapes."""
    c_rmem = fx.make_rmem_tensor(MFMA_ELEMS_PER_LANE, fx.Float32)
    cache = {}
    accs = []
    for mfma_k, lane_k in shapes:
        if (mfma_k, lane_k) not in cache:
            atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_N, mfma_k, elem_dtype))
            a_rmem = fx.make_rmem_tensor(lane_k, elem_dtype)
            b_rmem = fx.make_rmem_tensor(lane_k, elem_dtype)

            def acc(a_pack, b_pack, c, atom=atom, a_rmem=a_rmem, b_rmem=b_rmem):
                a_rmem.store(Vec(a_pack))
                b_rmem.store(Vec(b_pack))
                c_rmem.store(Vec(c))
                fx.mma_atom_call(atom, c_rmem, a_rmem, b_rmem, c_rmem)
                return c_rmem.load().ir_value()

            cache[(mfma_k, lane_k)] = acc
        accs.append(cache[(mfma_k, lane_k)])
    return accs


def pack_mfma_frag(vals, elem_dtype, truncate: bool):
    """Pack 4 fp32 values into a 16-bit MFMA operand fragment.

    ``truncate`` (bf16 only) keeps the high 16 bits of each fp32, rounding
    toward zero; otherwise the values convert with round-to-nearest.
    """
    if truncate and elem_dtype is fx.BFloat16:
        c16 = fx.Int32(16)
        cmask = fx.Int32(0xFFFF0000)

        def bf16_pair(lo, hi):
            return (hi.bitcast(fx.Int32) & cmask) | lo.bitcast(fx.Int32).shrui(c16)

        pairs = [bf16_pair(vals[0], vals[1]), bf16_pair(vals[2], vals[3])]
        return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
    return Vec.from_elements([v.to(elem_dtype) for v in vals], elem_dtype).ir_value()


def grouped_loader(t, dim: int, g: int):
    """Loader of a g-wide vector from ``t[row, head, :]``.

    The row is indexed through the tensor's own layout, whose strides are
    i64, so ``row * row_stride`` stays 64-bit on packed tensors past 2^31
    elements; the in-row access is a small i32 offset.
    """
    in_row = fx.make_layout((dim // g, g), (g, 1))

    def load(row_i64, head, colgrp):
        sub = t[row_i64, head, None]
        return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

    return load


def grouped_storer(t, dim: int, g: int):
    """Store counterpart of :func:`grouped_loader`."""
    in_row = fx.make_layout((dim // g, g), (g, 1))

    def store(row_i64, head, colgrp, vec):
        sub = t[row_i64, head, None]
        fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].store(vec)

    return store


def _min(a, b):
    return (a < b).select(a, b)


def _max(a, b):
    return (a > b).select(a, b)


def silu_gate(s_list, c_alpha_neg_log2e):
    """``(-log2e) * silu(alpha * s)`` per score, stage-batched for ILP.

    ``exp2(alpha * s * -log2e) == e^(-alpha * s)`` builds the sigmoid without a
    slow exp, and ``g = alpha * s * -log2e`` doubles as the outer factor, so
    the caller multiplies the result by ``-1 / log2e``. A zeroed score gives 0.
    """
    g = [s * c_alpha_neg_log2e for s in s_list]
    den = [fx.Float32(1.0) + exp2_f32(gi) for gi in g]
    sig = [fx.Float32(fx.rocdl.rcp(fx.Float32.ir_type, d.ir_value())) for d in den]
    return [g[i] * sig[i] for i in range(len(s_list))]


class RaggedMask:
    """T1 ragged mask of one sequence, built inside a kernel.

    ``seq_len`` and ``num_target`` are the sequence's values; ``window``,
    ``contextual`` and ``full`` are the runtime option values (ignored when
    their feature is not compiled in).
    """

    def __init__(
        self, feats: RaggedFeatures, seq_len, num_target, window, contextual, full
    ):
        self.f = feats
        self.seq_len = seq_len
        self.window = window
        self.contextual = contextual
        self.full = full
        max_id = seq_len
        if feats.has_contextual:
            max_id = max_id - contextual + fx.Int32(1)
        if feats.has_targets:
            max_id = max_id - num_target
        self.max_id = max_id

    def to_id(self, x):
        """Raw position -> remapped position: the contextual prefix collapses
        to 0, then targets clamp to ``max_id``."""
        if self.f.has_contextual:
            x = _max(x - self.contextual + fx.Int32(1), fx.Int32(0))
        if self.f.has_targets:
            x = _min(x, self.max_id)
        return x

    def keep(self, row, col, row_id, col_id):
        """Whether query ``row`` sees key ``col`` (raw positions and their
        ids). In-sequence bounds are the caller's job."""
        f = self.f
        m, n = row_id, col_id
        if f.mask_type == "lower_triangular":
            visible = n < m
            if f.has_window:
                visible = visible & (n >= m - self.window)
                if f.has_full:
                    visible = visible | ((m >= self.max_id - self.full) & (n < m))
        elif f.mask_type == "upper_triangular":
            visible = n > m
        else:
            visible = n < self.max_id
            if f.has_window:
                d = n - m
                visible = visible & (d <= self.window) & (d >= -self.window)
        if f.has_contextual:
            visible = visible | ((m == fx.Int32(0)) & (n < self.max_id))
            visible = visible | (n == fx.Int32(0))
        return (row == col) | visible

    def _first_raw(self, x):
        """Smallest raw position whose id is >= ``x`` (``x`` below max_id)."""
        if self.f.has_contextual:
            return (x > fx.Int32(0)).select(
                x + self.contextual - fx.Int32(1), fx.Int32(0)
            )
        return _max(x, fx.Int32(0))

    def kv_range(self, q_first, q_last):
        """``(low, high)``: raw key columns any row of ``[q_first, q_last]``
        can see, outside the contextual columns. With a window and a
        contextual prefix, columns ``[0, contextual)`` (all id 0, which every
        row sees) must be visited in addition. ``q_last < seq_len``."""
        f = self.f
        seq_len = self.seq_len
        diag_high = q_last + fx.Int32(1)
        if f.mask_type == "upper_triangular":
            return q_first, seq_len
        if f.mask_type == "lower_triangular":
            low = fx.Int32(0)
            high = diag_high
            if f.has_window:
                low = self._first_raw(self.to_id(q_first) - self.window)
                if f.has_full:
                    full_rows = self.to_id(q_last) >= self.max_id - self.full
                    low = full_rows.select(fx.Int32(0), low)
            if f.has_contextual:
                # Rows with id 0 see every key below max_id.
                prefix = q_first < self.contextual
                low = prefix.select(fx.Int32(0), low)
                high = prefix.select(seq_len, high)
            return low, high
        # "none": keys with id < max_id, plus the diagonal.
        visible_high = self.max_id
        low = fx.Int32(0)
        if f.has_window:
            low = self._first_raw(self.to_id(q_first) - self.window)
            visible_high = _min(
                visible_high, self.to_id(q_last) + self.window + fx.Int32(1)
            )
        return low, _min(_max(visible_high, diag_high), seq_len)
