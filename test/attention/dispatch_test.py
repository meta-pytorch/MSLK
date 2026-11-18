# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each routed attention op serves its calls with the backend its tuning table
names.

End to end, on the GPU: for each op, the cheapest row of its table per backend
is called through the op, and the test checks which backend served it. Adding
a routed op means adding one entry to ``CASES``.
"""

import math
import unittest
from typing import Any, Callable, Dict, List, Tuple
from unittest import mock

import torch
from mslk.aiter.common import is_aiter_enabled
from mslk.aiter.routing import device_key, OpRouting
from mslk.attention.dispatch import paged_attention_decode_fp8
from mslk.attention.flydsl.decode.fp8_paged_cache import random_fp8_paged_cache
from mslk.attention.flydsl.decode.pa_decode_fp8_dispatch import (
    is_fp8_paged_decode_available,
    paged_attention_decode_fp8 as paged_attention_decode_fp8_op,
)
from mslk.testing.device import skipUnlessRocm
from parameterized import parameterized

Args = Tuple[Tuple[Any, ...], Dict[str, Any]]


def _paged_attention_decode_fp8_args(row: Dict[str, str]) -> Args:
    if not is_fp8_paged_decode_available():
        raise unittest.SkipTest("requires a gfx950 GPU with FlyDSL")
    heads, kv_heads = int(row["num_heads"]), int(row["num_kv_heads"])
    head_size, block_size = int(row["head_size"]), int(row["block_size"])
    batch, context = int(row["batch_max"]), int(row["context_max"])
    key_cache, value_cache, key_scale, value_scale, block_tables, context_lengths = (
        random_fp8_paged_cache(kv_heads, head_size, block_size, [context] * batch)
    )
    query = torch.randn(batch, heads, head_size, dtype=torch.bfloat16, device="cuda")
    return (
        torch.empty_like(query),
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_size**-0.5,
        key_scale,
        value_scale,
    ), {}


# (name, routing, the op, arguments for a table row)
CASES: List[Tuple[str, OpRouting, Callable[..., Any], Callable[..., Args]]] = [
    (
        "paged_attention_decode_fp8",
        paged_attention_decode_fp8.ROUTING,
        paged_attention_decode_fp8_op,
        _paged_attention_decode_fp8_args,
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
