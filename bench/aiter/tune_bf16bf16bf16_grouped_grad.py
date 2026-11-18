# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tune which backend serves ``mslk::bf16bf16bf16_grouped_grad`` on ROCm, per shape.

Compares MSLK's Triton dgrad kernel with aiter's Triton ``gmm`` on every shape
in ``bench/aiter/untuned/bf16bf16bf16_grouped_grad.csv`` and updates
``mslk/gemm/dispatch/tables/bf16bf16bf16_grouped_grad.csv``. The tuning itself
(incremental runs, multi-GPU, timing, choosing) is :mod:`mslk.aiter.tuning`;
this file holds only what is specific to the op. Needs the pinned aiter and a
gfx950 GPU.

The op is the input gradient of a grouped GEMM; ``W`` is passed as the
``[G, N, K]`` view of a ``[G, K, N]`` weight. Group sizes, reference and
tolerance are as in ``tune_bf16bf16bf16_grouped_stacked.py``.

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_bf16bf16bf16_grouped_grad.py --mp 4
"""

from pathlib import Path
from typing import Any, Dict

import mslk.gemm  # noqa: F401  (registers mslk::bf16bf16bf16_grouped_grad)
import torch
from mslk.aiter import tuning
from mslk.aiter.tuning import OpTuner, Row
from mslk.gemm.dispatch.bf16bf16bf16_grouped_grad import BACKENDS, OP

REPO = Path(__file__).resolve().parents[2]


def uneven_group_sizes(M: int, G: int) -> torch.Tensor:
    """Rows per group, summing to ``M``, skewed as MoE routing makes them: a
    few groups take most rows, others few or none."""
    weights = torch.rand(G, dtype=torch.float64).pow(3) + 1e-3
    picks = torch.multinomial(weights, M, replacement=True)
    return torch.bincount(picks, minlength=G)


class BF16GroupedGradTuner(OpTuner):
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
        W = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M * N, device="cuda", dtype=torch.bfloat16)
        return {"X": X, "W": W, "M_sizes": M_sizes, "out": out}

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return torch.ops.mslk.bf16bf16bf16_grouped_grad(
            inputs["X"],
            inputs["W"].permute(0, 2, 1),
            inputs["M_sizes"],
            inputs["out"],
        )

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        X, W = inputs["X"], inputs["W"]
        out = torch.empty(X.shape[0], W.shape[2], device=X.device, dtype=torch.float32)
        start = 0
        for g, size in enumerate(inputs["M_sizes"].tolist()):
            out[start : start + size] = X[start : start + size].float() @ W[g].float()
            start += size
        return out

    def flops(self, shape: Row) -> float:
        return 2.0 * int(shape["M_max"]) * int(shape["N"]) * int(shape["K"])

    def bytes_moved(self, shape: Row) -> float:
        M, G = int(shape["M_max"]), int(shape["G"])
        N, K = int(shape["N"]), int(shape["K"])
        return 2.0 * (M * K + G * N * K + M * N)


if __name__ == "__main__":
    tuning.main(
        BF16GroupedGradTuner(),
        untune_file=str(REPO / "bench/aiter/untuned/bf16bf16bf16_grouped_grad.csv"),
        tune_file=str(REPO / "mslk/gemm/dispatch/tables/bf16bf16bf16_grouped_grad.csv"),
    )
