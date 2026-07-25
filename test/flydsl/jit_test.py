# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest import mock

from mslk.flydsl import jit


class FlyDSLJitTest(unittest.TestCase):
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
            jit.run_compiled(launcher, 1, 2)
            # First call compiles with the launcher and args; does not invoke result.
            compile_fn.assert_called_once_with(launcher, 1, 2)
            compiled.assert_not_called()

            jit.run_compiled(launcher, 3, 4)
            # Second call reuses the cached CompiledFunction.
            compile_fn.assert_called_once()
            compiled.assert_called_once_with(3, 4)


if __name__ == "__main__":
    unittest.main()
