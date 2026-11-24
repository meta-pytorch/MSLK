# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL support for MSLK.

Importing any part of this package configures the FlyDSL runtime, so that a
kernel is served from the bundled AOT cache and honours the JIT switch
whichever launch helper it goes through. The call belongs here rather than
beside one of those helpers, so that the guarantee does not depend on which of
them a caller happens to use.

fbcode copy. The other files in this directory are byte-identical to the genai
mirror at genai/msl/ops/kernels/flydsl/extern/mslk_flydsl/ (hgdirsync, see
fbsource/.hgdirsync) and import this package by the short top-level name
`mslk_flydsl`, and mslk.attention.flydsl by `mslk_flydsl_attn`. In genai those
names resolve via a sys.path shim; here the alias finder below redirects them
to the canonical packages without re-importing, so each module initializes
once. This __init__.py is excluded from the mirror.
"""

import importlib
import importlib.abc
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Optional, Sequence


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Redirect imports of `alias_root[.*]` to `target_root[.*]`, registering the
    SAME module object under both names (true aliasing, single initialization)."""

    def __init__(self, alias_root: str, target_root: str) -> None:
        self._alias = alias_root
        self._target = target_root

    def _to_target(self, fullname: str) -> Optional[str]:
        if fullname == self._alias:
            return self._target
        prefix = self._alias + "."
        if fullname.startswith(prefix):
            return self._target + "." + fullname[len(prefix) :]
        return None

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        if self._to_target(fullname) is None:
            return None  # not ours -> defer to the normal finders
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        target = self._to_target(spec.name)
        assert target is not None
        # import_module is cached, so the canonical target initializes once.
        module = importlib.import_module(target)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module: ModuleType) -> None:
        pass  # already executed as the canonical target module


def install_alias(alias_root: str, target_root: str) -> None:
    if not any(
        isinstance(f, _AliasFinder) and f._alias == alias_root for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _AliasFinder(alias_root, target_root))


install_alias("mslk_flydsl", "mslk.flydsl")
install_alias("mslk_flydsl_attn", "mslk.attention.flydsl")

from mslk.flydsl.jit import configure_runtime_cache  # noqa: E402

configure_runtime_cache()
