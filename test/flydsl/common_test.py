# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest import mock

from mslk.flydsl import common


class FlyDSLCommonTest(unittest.TestCase):
    def setUp(self) -> None:
        common.is_flydsl_available.cache_clear()

    def tearDown(self) -> None:
        common.is_flydsl_available.cache_clear()

    def test_unavailable_when_not_installed(self) -> None:
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(common.is_flydsl_available())

    def test_require_raises_when_unavailable(self) -> None:
        with mock.patch.object(common, "is_flydsl_available", return_value=False):
            with self.assertRaises(RuntimeError):
                common.require_flydsl()

    def test_require_passes_when_available(self) -> None:
        with mock.patch.object(common, "is_flydsl_available", return_value=True):
            common.require_flydsl()


if __name__ == "__main__":
    unittest.main()
