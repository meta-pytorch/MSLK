# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
from mslk.flydsl import jit


class FlyDSLJitTest(unittest.TestCase):
    def test_run_compiled_compiles_then_caches(self) -> None:
        compiled = mock.Mock()
        compile_fn = mock.Mock(return_value=compiled)

        def launcher() -> None:
            pass

        flydsl_compiler = mock.Mock(compile=compile_fn)
        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "flydsl": mock.Mock(compiler=flydsl_compiler),
                    "flydsl.compiler": flydsl_compiler,
                },
            ),
            mock.patch("torch.cuda.current_device", return_value=0),
        ):
            jit.run_compiled(launcher, 1, 2, device="cuda:0")
            compile_fn.assert_called_once_with(launcher, 1, 2)
            compiled.assert_not_called()

            jit.run_compiled(launcher, 3, 4, device="cuda:0")
            compile_fn.assert_called_once()
            compiled.assert_called_once_with(3, 4)

    def test_run_compiled_has_independent_caches_per_device(self) -> None:
        compiled0 = mock.Mock()
        compiled1 = mock.Mock()
        compile_fn = mock.Mock(side_effect=[compiled0, compiled1])

        def launcher() -> None:
            pass

        def copy() -> None:
            pass

        vars(launcher)["_original_func"] = launcher
        vars(launcher)["compile_hints"] = {"waves_per_eu": 2}
        constructor = mock.Mock(return_value=copy)
        flydsl_compiler = mock.Mock(compile=compile_fn)
        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "flydsl": mock.Mock(compiler=flydsl_compiler),
                    "flydsl.compiler": flydsl_compiler,
                    "flydsl.compiler.jit_function": mock.Mock(JitFunction=constructor),
                },
            ),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.device") as device_context,
        ):
            jit.run_compiled(launcher, 1, device="cuda:0")
            jit.run_compiled(launcher, 2, device="cuda:1")
            jit.run_compiled(launcher, 3, device="cuda:0")
            jit.run_compiled(launcher, 4, device="cuda:1")

        self.assertEqual(
            compile_fn.call_args_list, [mock.call(launcher, 1), mock.call(copy, 2)]
        )
        constructor.assert_called_once_with(launcher, compile_hints={"waves_per_eu": 2})
        compiled0.assert_called_once_with(3)
        compiled1.assert_called_once_with(4)
        self.assertEqual(device_context.call_args_list, [mock.call(1), mock.call(1)])

    def test_run_compiled_infers_device_when_not_given(self) -> None:
        compiled = mock.Mock()
        flydsl_compiler = mock.Mock(compile=mock.Mock(return_value=compiled))

        def launcher() -> None:
            pass

        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "flydsl": mock.Mock(compiler=flydsl_compiler),
                    "flydsl.compiler": flydsl_compiler,
                },
            ),
            mock.patch.object(
                jit, "_launch_device", return_value=torch.device("cuda:1")
            ) as infer_device,
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.device") as device_context,
        ):
            jit.run_compiled(launcher, "operand")

        infer_device.assert_called_once_with(("operand",))
        device_context.assert_called_once_with(1)
        flydsl_compiler.compile.assert_called_once_with(launcher, "operand")
