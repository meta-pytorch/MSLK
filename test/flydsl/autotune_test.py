# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import unittest
from unittest import mock

from mslk.flydsl.autotune import _with_hints


class FlyDSLAutotuneHintsTest(unittest.TestCase):
    def test_compiler_hint_reaches_launcher(self) -> None:
        for hints, expected in [
            ({"waves_per_eu": 2}, 2),
            ({"waves_per_eu": 0}, 0),
            ({}, 3),
        ]:
            with self.subTest(hints=hints):
                context = mock.Mock()
                context.get_compile_hints.return_value = hints

                def launch(out: object, *, waves_per_eu: int = 0) -> tuple[object, int]:
                    return out, waves_per_eu

                output = object()
                with mock.patch.dict(
                    "sys.modules",
                    {
                        "flydsl.compiler.kernel_function": mock.Mock(
                            CompilationContext=context
                        )
                    },
                ):
                    result = _with_hints(launch)(output, waves_per_eu=3)
                self.assertEqual(result, (output, expected))

    def test_preserves_tuning_cache_identity_and_signature(self) -> None:
        def named_launch(out: object, *, waves_per_eu: int = 0) -> object:
            return out

        wrapped = _with_hints(named_launch)
        self.assertEqual(wrapped.__name__, named_launch.__name__)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(named_launch))
