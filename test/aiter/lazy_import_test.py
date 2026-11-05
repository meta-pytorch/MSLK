# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Importing MSLK, including its aiter backends, must not import aiter.

This test owns its own target so that nothing else in the process can import
aiter first and mask a regression.
"""

import importlib
import sys
import unittest


class LazyAiterImportTest(unittest.TestCase):
    def test_importing_mslk_does_not_import_aiter(self) -> None:
        self.assertNotIn("aiter", sys.modules, "aiter imported before the test ran")

        for module in (
            "mslk.gemm",
            "mslk.aiter.common",
            "mslk.aiter.routing",
            "mslk.gemm.aiter.f4f4bf16",
            "mslk.gemm.dispatch.f4f4bf16",
        ):
            importlib.import_module(module)

        self.assertNotIn("aiter", sys.modules)


if __name__ == "__main__":
    unittest.main()
