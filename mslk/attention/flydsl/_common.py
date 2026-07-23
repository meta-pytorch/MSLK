# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from contextlib import contextmanager

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf as _scf


@contextmanager
def _if_then(if_op, scf=None):
    """Context manager for SCF IfOp then-region across old/new Python APIs.

    Ensures the then block always ends with a YieldOp.
    The optional *scf* parameter is accepted for backward compatibility
    but ignored — the module-level import is used.
    """
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], _scf.YieldOp):
                _scf.YieldOp([])


def dtype_to_elem_type(dtype_str: str):
    """Map a dtype string to its FlyDSL numeric type.

    Supported: 'f32', 'f16', 'bf16', 'fp8' (OCP e4m3fn, not the fnuz variant).
    """
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "fp8":
        return fx.Float8E4M3FN
    raise ValueError(
        f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', 'bf16', or 'fp8')"
    )
