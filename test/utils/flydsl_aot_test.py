# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from unittest import mock

from mslk.utils import flydsl_aot


def _fake_kernel_module() -> mock.Mock:
    mod = mock.Mock()
    mod.AOT_ARCHS = ["gfx942", "gfx950"]
    mod.AOT_CONFIGS = [{"n": 128, "k": 256}, {"n": 512, "k": 256}]
    return mod


class _InlineExecutor:
    """Runs submitted work synchronously so mocks/closures apply (the real
    ProcessPoolExecutor runs in child processes where patches do not)."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass

    def submit(self, fn, *args, **kwargs):
        future = __import__("concurrent.futures").futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            future.set_exception(e)
        return future


class FlyDSLAOTTest(unittest.TestCase):
    def test_collect_jobs_is_cartesian_product(self) -> None:
        mod = _fake_kernel_module()
        with (
            mock.patch.object(flydsl_aot, "_AOT_KERNEL_MODULES", ["fake.kernel"]),
            mock.patch("importlib.import_module", return_value=mod),
        ):
            jobs = flydsl_aot.collect_aot_jobs()
        # 2 archs x 2 configs = 4 jobs.
        self.assertEqual(len(jobs), 4)
        self.assertEqual({arch for _, _, arch in jobs}, {"gfx942", "gfx950"})

    def test_compile_only_env_set_during_compile(self) -> None:
        seen = {}

        def _record(config, arch):
            seen["COMPILE_ONLY"] = os.environ.get("COMPILE_ONLY")
            seen["cache_dir"] = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")

        mod = _fake_kernel_module()
        mod.AOT_ARCHS = ["gfx950"]
        mod.AOT_CONFIGS = [{"n": 128, "k": 256}]
        mod.compile_aot_config = _record

        with (
            mock.patch.object(flydsl_aot, "_AOT_KERNEL_MODULES", ["fake.kernel"]),
            mock.patch("importlib.import_module", return_value=mod),
            mock.patch.object(flydsl_aot, "ProcessPoolExecutor", _InlineExecutor),
        ):
            flydsl_aot.compile_aot("/tmp/mslk_aot_unittest")

        self.assertEqual(seen["COMPILE_ONLY"], "1")
        self.assertEqual(seen["cache_dir"], "/tmp/mslk_aot_unittest")

    def test_compile_only_env_restored_after(self) -> None:
        before = os.environ.get("COMPILE_ONLY")
        with mock.patch.object(flydsl_aot, "_AOT_KERNEL_MODULES", []):
            flydsl_aot.compile_aot("/tmp/mslk_aot_unittest")
        self.assertEqual(os.environ.get("COMPILE_ONLY"), before)


if __name__ == "__main__":
    unittest.main()
