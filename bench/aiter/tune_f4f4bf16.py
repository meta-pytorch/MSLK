# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tune which backend serves ``mslk::f4f4bf16`` on ROCm, per shape.

Compares MSLK's Triton kernel with aiter's ``gemm_afp4wfp4`` on every shape in
``bench/aiter/untuned/f4f4bf16.csv`` and updates
``mslk/gemm/dispatch/tables/f4f4bf16.csv``. The tuning itself (incremental runs,
multi-GPU, timing, choosing) is :mod:`mslk.aiter.tuning`; this file holds only
what is specific to the op. Needs the pinned aiter and a gfx950 GPU.

Correctness: the reference is the product of the dequantized inputs, which
the kernels match up to summation order and BF16 rounding, so the tolerance is
tight (rtol = atol = 1e-2, aiter's default). Inputs are unit-scale, so outputs
are far larger than atol at every K.

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_f4f4bf16.py --mp 4
"""

from pathlib import Path
from typing import Any, Dict

import mslk.gemm  # noqa: F401  (registers mslk::f4f4bf16)
import torch
from mslk.aiter import tuning
from mslk.aiter.tuning import OpTuner, Row
from mslk.gemm.dispatch.f4f4bf16 import BACKENDS, OP
from mslk.quantize.triton.fp4_quantize import triton_quantize_mx4_unpack
from mslk.quantize.triton.fp4_utils import dequantize_mx4_unswizzled

REPO = Path(__file__).resolve().parents[2]


class F4F4BF16Tuner(OpTuner):
    op = OP
    key_columns = ("N", "K")
    bucket_column = "M_max"
    backends = BACKENDS
    atol = rtol = 1e-2
    outputs = ("out",)

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        M, N, K = int(shape["M_max"]), int(shape["N"]), int(shape["K"])
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        aq, a_scale = triton_quantize_mx4_unpack(A)
        bq, b_scale = triton_quantize_mx4_unpack(B)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        return {"aq": aq, "bq": bq, "a_scale": a_scale, "b_scale": b_scale, "out": out}

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return torch.ops.mslk.f4f4bf16(
            inputs["aq"],
            inputs["bq"],
            inputs["a_scale"],
            inputs["b_scale"],
            output=inputs["out"],
        )

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        a = dequantize_mx4_unswizzled(inputs["aq"], inputs["a_scale"])
        b = dequantize_mx4_unswizzled(inputs["bq"], inputs["b_scale"])
        return a @ b.T

    def flops(self, shape: Row) -> float:
        return 2.0 * int(shape["M_max"]) * int(shape["N"]) * int(shape["K"])

    def bytes_moved(self, shape: Row) -> float:
        M, N, K = int(shape["M_max"]), int(shape["N"]), int(shape["K"])
        # FP4 operands, E8M0 scales (one per 32 elements), BF16 output.
        return (M + N) * K / 2 + (M + N) * K / 32 + 2 * M * N


if __name__ == "__main__":
    tuning.main(
        F4F4BF16Tuner(),
        untune_file=str(REPO / "bench" / "aiter" / "untuned" / "f4f4bf16.csv"),
        tune_file=str(REPO / "mslk" / "gemm" / "dispatch" / "tables" / "f4f4bf16.csv"),
    )
