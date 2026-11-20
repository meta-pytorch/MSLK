# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tune which backend serves ``mslk::bf16bf16bf16_grouped_stacked`` on ROCm, per shape.

Compares MSLK's FlyDSL grouped kernel with aiter's Triton ``gmm`` on every shape
in ``bench/aiter/untuned/bf16bf16bf16_grouped_stacked.csv`` and updates
``mslk/gemm/dispatch/tables/bf16bf16bf16_grouped_stacked.csv``. The tuning
itself (incremental runs, multi-GPU, timing, choosing) is
:mod:`mslk.aiter.tuning`; this file holds only what is specific to the op. Needs
the pinned aiter and a gfx950 GPU.

MSLK's FlyDSL kernel picks its tile by autotuning only with
``MSLK_AUTOTUNE_ENABLE``, which this script sets unless it is already set, so
aiter is compared with the tuned kernel. Group sizes are drawn from the seed the
tuning driver sets per shape, so every run times the same split.

Correctness: the reference is the FP32 product of the BF16 inputs, group by
group, which the kernels match up to summation order and BF16 rounding, so the
tolerance is tight (rtol = atol = 1e-2, aiter's default). Inputs are
unit-scale, so outputs (about sqrt(K)) are far larger than atol at every K.

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_bf16bf16bf16_grouped_stacked.py --mp 4
"""

import os
from pathlib import Path
from typing import Any, Dict

import mslk.gemm  # noqa: F401  (registers mslk::bf16bf16bf16_grouped_stacked)
import torch
from mslk.aiter import tuning
from mslk.aiter.tuning import OpTuner, Row
from mslk.gemm.dispatch.bf16bf16bf16_grouped_stacked import BACKENDS, OP

REPO = Path(__file__).resolve().parents[2]


def uneven_group_sizes(M: int, G: int) -> torch.Tensor:
    """Rows per group, summing to ``M``, skewed as MoE routing makes them: a
    few groups take most rows, others few or none."""
    weights = torch.rand(G, dtype=torch.float64).pow(3) + 1e-3
    picks = torch.multinomial(weights, M, replacement=True)
    return torch.bincount(picks, minlength=G)


def grouped_reference(
    X: torch.Tensor, W_nk: torch.Tensor, M_sizes: torch.Tensor
) -> torch.Tensor:
    """FP32 ``X[g] @ W_nk[g].T`` for each group ``g`` of the rows of ``X``."""
    out = torch.empty(X.shape[0], W_nk.shape[1], device=X.device, dtype=torch.float32)
    start = 0
    for g, size in enumerate(M_sizes.tolist()):
        out[start : start + size] = X[start : start + size].float() @ W_nk[g].float().T
        start += size
    return out


class BF16GroupedStackedTuner(OpTuner):
    op = OP
    key_columns = ("G", "N", "K")
    bucket_column = "M_max"
    backends = BACKENDS
    atol = rtol = 1e-2
    outputs = ("out",)

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        M, G = int(shape["M_max"]), int(shape["G"])
        N, K = int(shape["N"]), int(shape["K"])
        M_sizes = uneven_group_sizes(M, G).to("cuda")
        X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        return {"X": X, "W": W, "M_sizes": M_sizes, "out": out}

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return torch.ops.mslk.bf16bf16bf16_grouped_stacked(
            inputs["X"], inputs["W"], inputs["M_sizes"], inputs["out"]
        )

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        return grouped_reference(inputs["X"], inputs["W"], inputs["M_sizes"])

    def flops(self, shape: Row) -> float:
        return 2.0 * int(shape["M_max"]) * int(shape["N"]) * int(shape["K"])

    def bytes_moved(self, shape: Row) -> float:
        M, G = int(shape["M_max"]), int(shape["G"])
        N, K = int(shape["N"]), int(shape["K"])
        return 2.0 * (M * K + G * N * K + M * N)


if __name__ == "__main__":
    os.environ.setdefault("MSLK_AUTOTUNE_ENABLE", "1")
    tuning.main(
        BF16GroupedStackedTuner(),
        untune_file=str(REPO / "bench/aiter/untuned/bf16bf16bf16_grouped_stacked.csv"),
        tune_file=str(
            REPO / "mslk/gemm/dispatch/tables/bf16bf16bf16_grouped_stacked.csv"
        ),
    )
