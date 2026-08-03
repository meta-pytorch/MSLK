# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest import mock

from mslk.flydsl import common
from parameterized import parameterized


class FlyDSLCommonTest(unittest.TestCase):
    def setUp(self) -> None:
        common.is_flydsl_available.cache_clear()
        common.flydsl_version.cache_clear()

    def tearDown(self) -> None:
        common.is_flydsl_available.cache_clear()
        common.flydsl_version.cache_clear()

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

    @parameterized.expand(
        [
            ("exact_match", "0.2.4", True),
            ("older_minor", "0.1.5", False),
            ("newer_patch", "0.2.10", True),
            ("newer_minor", "0.3.0", True),
            ("newer_major", "1.0", True),
            ("prerelease_of_minimum", "0.2.4rc1", True),
            ("dev_build", "0.0.1.dev95158637", False),
            ("truncated_below_minimum", "0.2", False),
        ]
    )
    def test_version_at_least(self, _name: str, version: str, expected: bool) -> None:
        with mock.patch.object(common, "is_flydsl_available", return_value=True):
            with mock.patch.object(common, "flydsl_version", return_value=version):
                self.assertEqual(common.is_flydsl_version_at_least("0.2.4"), expected)

    def test_version_at_least_false_when_unavailable(self) -> None:
        with mock.patch.object(common, "is_flydsl_available", return_value=False):
            self.assertFalse(common.is_flydsl_version_at_least("0.2.4"))

    def test_version_at_least_false_when_version_unknown(self) -> None:
        with mock.patch.object(common, "is_flydsl_available", return_value=True):
            with mock.patch.object(common, "flydsl_version", return_value=None):
                self.assertFalse(common.is_flydsl_version_at_least("0.2.4"))

    def test_version_none_when_not_importable(self) -> None:
        with mock.patch.dict("sys.modules", {"flydsl": None}):
            self.assertIsNone(common.flydsl_version())

    def test_defaults_to_min_supported_version(self) -> None:
        with mock.patch.object(common, "is_flydsl_available", return_value=True):
            with mock.patch.object(
                common, "flydsl_version", return_value=common.MIN_FLYDSL_VERSION
            ):
                self.assertTrue(common.is_flydsl_version_at_least())


if __name__ == "__main__":
    unittest.main()
