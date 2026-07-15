# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Ahead-of-time (AOT) pre-compilation for FlyDSL kernels.

Builds FlyDSL's on-disk cache for selected kernels in selected
configurations so the front-end compile is skipped at runtime. Each
config is compiled under ``COMPILE_ONLY=1`` into
``FLYDSL_RUNTIME_CACHE_DIR``; at runtime FlyDSL loads from that cache and
falls back to JIT on a miss. The cache stores lowered IR rather than a
standalone binary, so FlyDSL remains a runtime dependency.

AOT-eligible kernel modules declare what to pre-compile via three
attributes:
  - ``AOT_CONFIGS``: list of config dicts (kernel-specific params).
  - ``AOT_ARCHS``: list of gfx arch strings to build for.
  - ``compile_aot_config(config, arch)``: compile one config for one arch
    (under FakeTensorMode; no GPU tensors needed).
"""

import importlib
import multiprocessing as mp
import os
from concurrent.futures import as_completed, ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

from mslk.utils.flydsl import _BUNDLED_AOT_CACHE

# Module paths of AOT-eligible FlyDSL kernel modules. Each must expose
# AOT_CONFIGS, AOT_ARCHS, and compile_aot_config(config, arch).
_AOT_KERNEL_MODULES: List[str] = []

_DEFAULT_MAX_WORKERS: int = 64


@contextmanager
def _compile_only_env(cache_dir: str) -> Iterator[None]:
    prev_compile = os.environ.get("COMPILE_ONLY")
    prev_cache = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    os.environ["COMPILE_ONLY"] = "1"
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir
    try:
        yield
    finally:
        for name, prev in (
            ("COMPILE_ONLY", prev_compile),
            ("FLYDSL_RUNTIME_CACHE_DIR", prev_cache),
        ):
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev


def _affinity_aware_cpu_count() -> int:
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        return max(os.cpu_count() or 1, 1)


def _resolve_max_workers(num_jobs: int) -> int:
    env = os.environ.get("MSLK_FLYDSL_AOT_WORKERS")
    if env is not None:
        workers = max(int(env), 1)
    else:
        workers = min(_affinity_aware_cpu_count(), _DEFAULT_MAX_WORKERS)
    return max(min(workers, num_jobs), 1)


def collect_aot_jobs() -> List[Tuple[str, Dict[str, Any], str]]:
    """Collect (module_path, config, arch) jobs from registered kernels."""
    jobs: List[Tuple[str, Dict[str, Any], str]] = []
    for module_path in _AOT_KERNEL_MODULES:
        mod = importlib.import_module(module_path)
        for arch in mod.AOT_ARCHS:
            for config in mod.AOT_CONFIGS:
                jobs.append((module_path, config, arch))
    return jobs


def _compile_one(module_path: str, config: Dict[str, Any], arch: str) -> bool:
    """Worker: compile one config for one arch. Runs in a child process."""
    mod = importlib.import_module(module_path)
    mod.compile_aot_config(config, arch)
    return True


def compile_aot(cache_dir: str) -> None:
    """Pre-compile all registered FlyDSL kernel configs into ``cache_dir``.

    Raises ``RuntimeError`` if any job fails.
    """
    os.makedirs(cache_dir, exist_ok=True)
    jobs = collect_aot_jobs()
    if not jobs:
        print("[mslk] FlyDSL AOT: no kernels to compile, skipping")
        return

    max_workers = _resolve_max_workers(len(jobs))
    print(
        f"[mslk] FlyDSL AOT: {len(jobs)} jobs, {max_workers} worker processes "
        f"(cache: {cache_dir})"
    )

    errors: List[str] = []
    # Use spawn: workers compile FlyDSL kernels which touch CUDA, and forking
    # a parent that has already initialized CUDA breaks re-initialization.
    with _compile_only_env(cache_dir):
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(_compile_one, module_path, config, arch): (
                    module_path,
                    config,
                    arch,
                )
                for module_path, config, arch in jobs
            }
            for done, future in enumerate(as_completed(futures), 1):
                module_path, config, arch = futures[future]
                label = f"{module_path} {config} arch={arch}"
                try:
                    future.result()
                    print(f"  [OK] {done}/{len(jobs)} {label}")
                except Exception as e:
                    errors.append(f"{label}: {e}")
                    print(f"  [FAIL] {done}/{len(jobs)} {label}: {e}")

    if errors:
        raise RuntimeError(
            f"FlyDSL AOT failed for {len(errors)}/{len(jobs)} jobs: "
            + "; ".join(errors[:10])
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AOT pre-compile FlyDSL kernels")
    parser.add_argument(
        "--cache-dir",
        default=_BUNDLED_AOT_CACHE,
        help="Output cache directory (default: the bundled package cache).",
    )
    args = parser.parse_args()
    compile_aot(args.cache_dir)
