# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Importing ``mslk.gemm`` must not import FlyDSL.

FlyDSL bundles its own MLIR/LLVM. Loading it into a process that already holds
Triton's makes the two interpose on each other, and the process dies on SIGSEGV
during ``flydsl._mlir`` initialization -- which no ``except`` can catch, so the
graceful-degradation paths in ``mslk.flydsl.common`` do not help. Any consumer
that reaches ``mslk.gemm`` transitively (torchao does) would take the whole
process down at import.

This test owns its own target so that nothing else in the process can import
FlyDSL first and mask a regression.
"""

import sys
import unittest


class LazyGemmImportTest(unittest.TestCase):
    def test_importing_gemm_does_not_import_flydsl(self) -> None:
        self.assertNotIn("flydsl", sys.modules, "FlyDSL imported before the test ran")

        import mslk.gemm  # noqa: F401

        self.assertNotIn("flydsl", sys.modules)
