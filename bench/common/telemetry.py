# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Telemetry utilities for MSLK benchmark suite.

This module provides a generic interface for logging benchmark results.
FB-specific implementations (Scuba) are loaded optionally when available.

In OSS builds, telemetry functions are no-ops that print a warning message.
In FB-internal builds, this module delegates to the FB-specific implementation.

Example usage:
    from mslk.bench.common.telemetry import (
        export_benchmark_to_scuba,
        is_scuba_available,
    )

    if is_scuba_available():
        export_benchmark_to_scuba(samples=benchmark_results)
"""

from typing import Any, Callable, Optional

# Detect if we're running in FB environment by attempting to import FB-specific module
_SCUBA_AVAILABLE = False
_export_benchmark_to_scuba_impl: Optional[Callable[..., None]] = None

try:
    from mslk.bench.common.fb.scuba_utils import export_benchmark_to_scuba as _fb_export

    _export_benchmark_to_scuba_impl = _fb_export
    _SCUBA_AVAILABLE = True
except ImportError:
    pass


def is_scuba_available() -> bool:
    """Check if Scuba logging is available.

    Returns:
        True if running in FB-internal build with Scuba support, False otherwise.
    """
    return _SCUBA_AVAILABLE


def export_benchmark_to_scuba(*args: Any, **kwargs: Any) -> None:
    """Export benchmark results to Scuba.

    In OSS builds, this is a no-op with a warning message.
    In FB-internal builds, this logs to the mslk_gemm_bench Scuba table.

    Args:
        *args: Positional arguments passed to the underlying implementation.
        **kwargs: Keyword arguments passed to the underlying implementation.
    """
    if _export_benchmark_to_scuba_impl is not None:
        _export_benchmark_to_scuba_impl(*args, **kwargs)
    else:
        print("Scuba export is not available in OSS builds. Skipping.")
