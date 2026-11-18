# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each routed GEMM op serves its calls with the backend its tuning table names.

End to end, on the GPU: for each op, the cheapest row of its table per backend
is called through the registered op, and the test checks which backend served
it. Adding a routed op means adding one entry to ``CASES``.
"""

import math
import unittest
from typing import Any, Callable, Dict, List, Tuple
from unittest import mock

import mslk.gemm  # noqa: F401
import torch
from mslk.aiter.common import is_aiter_enabled
from mslk.aiter.routing import device_key, OpRouting
from mslk.gemm.dispatch import (
    bf16bf16bf16_grouped_grad,
    bf16bf16bf16_grouped_stacked,
    f4f4bf16,
)
from mslk.quantize.triton.fp4_quantize import triton_quantize_mx4_unpack
from mslk.testing.device import skipUnlessRocm
from parameterized import parameterized

Args = Tuple[Tuple[Any, ...], Dict[str, Any]]


def _f4f4bf16_args(row: Dict[str, str]) -> Args:
    M, N, K = int(row["M_max"]), int(row["N"]), int(row["K"])
    aq, a_scale = triton_quantize_mx4_unpack(
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    )
    bq, b_scale = triton_quantize_mx4_unpack(
        torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    )
    return (aq, bq, a_scale, b_scale), {}


def _group_sizes(M: int, G: int) -> torch.Tensor:
    """M rows spread as evenly as possible over G groups."""
    sizes = torch.full((G,), M // G, dtype=torch.int64)
    sizes[: M % G] += 1
    return sizes.cuda()


def _bf16bf16bf16_grouped_stacked_args(row: Dict[str, str]) -> Args:
    M, G = int(row["M_max"]), int(row["G"])
    N, K = int(row["N"]), int(row["K"])
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16)
    return (X, W, _group_sizes(M, G)), {}


def _bf16bf16bf16_grouped_grad_args(row: Dict[str, str]) -> Args:
    M, G = int(row["M_max"]), int(row["G"])
    N, K = int(row["N"]), int(row["K"])
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1)
    return (X, W, _group_sizes(M, G)), {}


# (name, routing, the registered op, arguments for a table row)
CASES: List[Tuple[str, OpRouting, Callable[..., Any], Callable[..., Args]]] = [
    (
        "f4f4bf16",
        f4f4bf16.ROUTING,
        lambda *a, **kw: torch.ops.mslk.f4f4bf16(*a, **kw),
        _f4f4bf16_args,
    ),
    (
        "bf16bf16bf16_grouped_stacked",
        bf16bf16bf16_grouped_stacked.ROUTING,
        lambda *a, **kw: torch.ops.mslk.bf16bf16bf16_grouped_stacked(*a, **kw),
        _bf16bf16bf16_grouped_stacked_args,
    ),
    (
        "bf16bf16bf16_grouped_grad",
        bf16bf16bf16_grouped_grad.ROUTING,
        lambda *a, **kw: torch.ops.mslk.bf16bf16bf16_grouped_grad(*a, **kw),
        _bf16bf16bf16_grouped_grad_args,
    ),
]


@skipUnlessRocm()
class DispatchTest(unittest.TestCase):
    @parameterized.expand(CASES, skip_on_empty=True)
    def test_follows_its_tuning_table(
        self,
        _name: str,
        routing: OpRouting,
        op: Callable[..., Any],
        make_args: Callable[..., Args],
    ) -> None:
        # Off when aiter is not installed, or switched off for the op (e.g.
        # MSLK_DISABLE_AITER=1); MSLK's kernel then serves every call.
        if not is_aiter_enabled(routing.op):
            self.skipTest("requires the pinned aiter, enabled for the op")
        gfx, cu_num = device_key(torch.cuda.current_device())
        table = routing.table
        columns = table.key_columns + (table.bucket_column,)
        rows = [r for r in table.rows if (r["gfx"], r["cu_num"]) == (gfx, str(cu_num))]
        # The cheapest shape for each backend is enough to show the routing.
        rows.sort(key=lambda r: math.prod(int(r[c]) for c in columns if r[c].isdigit()))
        cheapest = {}
        for row in rows:
            cheapest.setdefault(row["backend"], row)
        if not cheapest:
            self.skipTest(f"no tuning-table rows for {gfx} with {cu_num} CUs")
        for expected, row in cheapest.items():
            args, kwargs = make_args(row)
            spies = {b: mock.Mock(wraps=fn) for b, fn in routing.backends.items()}
            with mock.patch.dict(routing.backends, spies):
                op(*args, **kwargs)
            for backend, spy in spies.items():
                self.assertEqual(spy.call_count, int(backend == expected), row)


if __name__ == "__main__":
    unittest.main()
