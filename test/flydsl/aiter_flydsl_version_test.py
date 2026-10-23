# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL must be the version the installed aiter requires.

MSLK pins FlyDSL exactly (``ci/flydsl_version.txt``) for its own kernels and its
bundled AOT cache, and aiter pins it exactly too, so with aiter installed the
two pins must be the same. Moving the aiter pin (``mslk/aiter/aiter_version.txt``)
therefore means moving the FlyDSL pin to the version that aiter build requires.
pip does not enforce this when the two are installed one after the other: the
later install replaces FlyDSL and only warns.

Everything is read from installed package metadata, so this checks what was
installed rather than the files in the repository. It skips without aiter. With
aiter installed, MSLK must be found as a package too; it may be named other
than ``mslk`` (CI builds e.g. ``mslk-rocm-nightly``).
"""

import unittest
from importlib import metadata
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.version import Version


def _mslk_distributions() -> list[str]:
    """Names of the installed packages providing ``mslk``.

    Looked up by the import name, since the package name varies with how MSLK
    was built (``mslk``, ``mslk-rocm-nightly``, ...). Editable installs are not
    always mapped to the import name, so packages named ``mslk*`` count too.
    """
    names = set(metadata.packages_distributions().get("mslk", []))
    for dist in metadata.distributions():
        name = dist.metadata["Name"] or ""
        if name.lower().replace("_", "-").split("-")[0] == "mslk":
            names.add(name)
    return sorted(names)


def _pinned_flydsl(dist: str) -> Optional["Version"]:
    """The FlyDSL version ``dist`` pins, or ``None`` if it requires no FlyDSL.

    Both MSLK and aiter pin one exact version (``===`` or ``==``); anything
    looser fails the test rather than being compared loosely.
    """
    # Imported here so that the module still loads, and the test skips, in
    # environments without packaging (it is installed wherever aiter is).
    from packaging.requirements import Requirement
    from packaging.version import Version

    for line in metadata.requires(dist) or []:
        req = Requirement(line)
        if req.name != "flydsl" or (
            req.marker is not None and not req.marker.evaluate()
        ):
            continue
        specs = list(req.specifier)
        if len(specs) != 1 or specs[0].operator not in ("==", "==="):
            raise AssertionError(f"{dist} does not pin one FlyDSL version: {line}")
        return Version(specs[0].version)
    return None


class AiterFlyDSLVersionTest(unittest.TestCase):
    aiter_flydsl: "Version"
    mslk_dists: list[str]

    def setUp(self) -> None:
        try:
            metadata.version("amd-aiter")
        except metadata.PackageNotFoundError:
            self.skipTest("aiter is not installed")
        self.mslk_dists = _mslk_distributions()
        if not self.mslk_dists:
            self.fail("aiter is installed, but no installed package provides mslk")
        pinned = _pinned_flydsl("amd-aiter")
        if pinned is None:
            self.fail("aiter declares no FlyDSL requirement")
        self.aiter_flydsl = pinned

    def test_mslk_pins_the_flydsl_aiter_requires(self) -> None:
        for dist in self.mslk_dists:
            self.assertEqual(
                _pinned_flydsl(dist),
                self.aiter_flydsl,
                f"{dist}: ci/flydsl_version.txt must match the FlyDSL version "
                "required by the aiter build pinned in mslk/aiter/aiter_version.txt",
            )

    def test_installed_flydsl_is_the_one_aiter_requires(self) -> None:
        from packaging.version import Version

        # Compared as versions, so a local build such as 0.4.0+20261016.5ce3fbe
        # does not pass for the 0.4.0 release.
        installed = Version(metadata.version("flydsl"))
        self.assertEqual(installed, self.aiter_flydsl)


if __name__ == "__main__":
    unittest.main()
