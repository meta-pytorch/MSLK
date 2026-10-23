# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""What CI's ``install_aiter_pip`` (ci/scripts/utils_aiter.bash) does on each outcome.

aiter is optional and comes from a server outside PyPI, so CI must continue
without it when it cannot be installed or imported, while a FlyDSL pin that
differs from aiter's must fail. Those paths only happen when something breaks,
so they are tested here instead: the function runs in bash with ``conda``, the
retry wrapper and the import check replaced by stubs that log their calls and
succeed or fail as each case needs. Needs only bash.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
PINNED_FLYDSL: str = "".join((REPO / "ci" / "flydsl_version.txt").read_text().split())

_SCRIPT = """
. "$REPO/ci/scripts/utils_aiter.bash"
env_name_or_prefix () { echo "-n $1"; }
exec_with_retries () { shift; "$@"; }
test_python_import_package () { echo "import $2" >> "$CALLS"; [ "$FAKE_IMPORT_OK" = 1 ]; }
conda () {
  case "$*" in
    *"pip install"*) echo "install" >> "$CALLS"; [ "$FAKE_INSTALL_OK" = 1 ] ;;
    *"pip uninstall"*) echo "uninstall" >> "$CALLS" ;;
    *"version('amd-aiter'), version('flydsl')"*) echo "aiter $FAKE_FLYDSL" ;;
    *"version('flydsl')"*) echo "$FAKE_FLYDSL" ;;
  esac
}
install_aiter_pip build_env
"""


def _run(
    install_ok: bool = True,
    import_ok: bool = True,
    flydsl: str = PINNED_FLYDSL,
    variant: str = "rocm",
) -> Tuple[int, str, List[str]]:
    """(exit code, output, logged calls) of one run."""
    with tempfile.TemporaryDirectory() as tmp:
        calls = os.path.join(tmp, "calls")
        Path(calls).touch()
        env: Dict[str, str] = {
            **os.environ,
            "REPO": str(REPO),
            "CALLS": calls,
            "BUILD_VARIANT": variant,
            "FAKE_INSTALL_OK": "1" if install_ok else "0",
            "FAKE_IMPORT_OK": "1" if import_ok else "0",
            "FAKE_FLYDSL": flydsl,
        }
        result = subprocess.run(
            ["bash", "-c", _SCRIPT], env=env, capture_output=True, text=True
        )
        return (
            result.returncode,
            result.stdout,
            Path(calls).read_text().split("\n")[:-1],
        )


class InstallAiterCITest(unittest.TestCase):
    def test_success(self) -> None:
        code, out, calls = _run()
        self.assertEqual(code, 0, out)
        self.assertEqual(calls, ["install", "import aiter"])
        self.assertNotIn("::warning::", out)

    def test_install_failure_continues_without_aiter(self) -> None:
        code, out, calls = _run(install_ok=False)
        self.assertEqual(code, 0, out)
        self.assertIn("::warning::Could not install aiter", out)
        self.assertEqual(calls, ["install", "uninstall"])

    def test_import_failure_uninstalls_and_continues(self) -> None:
        code, out, calls = _run(import_ok=False)
        self.assertEqual(code, 0, out)
        self.assertIn("::warning::", out)
        self.assertIn("cannot be imported", out)
        self.assertEqual(calls, ["install", "import aiter", "uninstall"])

    def test_flydsl_mismatch_fails_before_importing(self) -> None:
        code, out, calls = _run(flydsl="0.0.1", import_ok=False)
        self.assertEqual(code, 1, out)
        self.assertIn("requires a different FlyDSL", out)
        self.assertNotIn("::warning::", out)
        self.assertEqual(calls, ["install"])

    def test_skipped_off_rocm(self) -> None:
        code, out, calls = _run(variant="cuda")
        self.assertEqual(code, 0, out)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
