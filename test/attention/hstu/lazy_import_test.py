# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Importing the HSTU API must import neither aiter nor FlyDSL.

This test owns its own target so that nothing else in the process can import
them first and mask a regression.
"""

import importlib
import sys
import unittest

_BACKEND_PACKAGES = ("aiter", "flydsl")


def _imported_backends() -> list:
    return [
        name
        for name in sys.modules
        if any(name == p or name.startswith(p + ".") for p in _BACKEND_PACKAGES)
    ]


class LazyHstuImportTest(unittest.TestCase):
    def test_importing_hstu_imports_no_backend(self) -> None:
        self.assertEqual(_imported_backends(), [], "imported before the test ran")

        for module in (
            "mslk.attention.hstu",
            "mslk.attention.dispatch.hstu_ragged",
            "mslk.attention.dispatch.hstu_cross",
            "mslk.attention.dispatch.hstu_grouped",
        ):
            importlib.import_module(module)

        self.assertEqual(_imported_backends(), [])


if __name__ == "__main__":
    unittest.main()
