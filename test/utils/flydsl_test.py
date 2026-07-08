# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest import mock

from mslk.utils import flydsl


class FlyDSLSupportTest(unittest.TestCase):
    def setUp(self) -> None:
        flydsl.is_flydsl_available.cache_clear()

    def tearDown(self) -> None:
        flydsl.is_flydsl_available.cache_clear()

    def test_unavailable_when_not_installed(self) -> None:
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(flydsl.is_flydsl_available())

    def test_require_raises_when_unavailable(self) -> None:
        with mock.patch.object(flydsl, "is_flydsl_available", return_value=False):
            with self.assertRaises(RuntimeError):
                flydsl.require_flydsl()

    def test_require_passes_when_available(self) -> None:
        with mock.patch.object(flydsl, "is_flydsl_available", return_value=True):
            flydsl.require_flydsl()

    def test_run_compiled_compiles_then_caches(self) -> None:
        compiled = mock.Mock()
        compile_fn = mock.Mock(return_value=compiled)

        def launcher() -> None:
            pass

        flydsl_compiler = mock.Mock(compile=compile_fn)
        with mock.patch.dict(
            "sys.modules",
            {
                "flydsl": mock.Mock(compiler=flydsl_compiler),
                "flydsl.compiler": flydsl_compiler,
            },
        ):
            flydsl.run_compiled(launcher, 1, 2)
            # First call compiles with the launcher and args; does not invoke result.
            compile_fn.assert_called_once_with(launcher, 1, 2)
            compiled.assert_not_called()

            flydsl.run_compiled(launcher, 3, 4)
            # Second call reuses the cached CompiledFunction.
            compile_fn.assert_called_once()
            compiled.assert_called_once_with(3, 4)


if __name__ == "__main__":
    unittest.main()
