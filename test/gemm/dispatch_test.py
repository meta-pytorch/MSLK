# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
from mslk.aiter.common import is_aiter_available
from mslk.aiter.routing import device_key, OpRouting
from mslk.testing.device import skipUnlessRocm
from parameterized import parameterized

Args = Tuple[Tuple[Any, ...], Dict[str, Any]]


# (name, routing, the registered op, arguments for a table row)
CASES: List[Tuple[str, OpRouting, Callable[..., Any], Callable[..., Args]]] = []


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
        if not is_aiter_available():
            self.skipTest("requires the pinned aiter")
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
