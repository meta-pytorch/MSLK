# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Per-shape choice between MSLK's own kernel and aiter, for one op at a time.

Each op that can be served by aiter has a tuning table, written by the op's
tuning script (``bench/aiter/``, driven by :mod:`mslk.aiter.tuning`) and shipped
with the package next to the op's dispatch module (``mslk/gemm/dispatch/``),
which reads it through :class:`OpRouting`. A table is a CSV with one row per
device, op-specific key and bucket::

    gfx,cu_num,N,K,M_max,backend[,informational columns...]
    gfx950,256,8192,8192,64,aiter
    gfx950,256,8192,8192,16384,mslk

A call is served by the row for its device and key with the smallest bucket
bound (``M_max`` here) at or above the call's value, e.g. M=100 by the row with
M_max=128. Calls the table does not cover use MSLK's kernel, as do all calls
when aiter is not enabled for the op (see :mod:`mslk.aiter.common`).
"""

import bisect
import csv
import functools
import importlib.resources
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from mslk.aiter.common import is_aiter_enabled

MSLK: str = "mslk"

_DeviceKey = Tuple[str, int]


class ShapeTable:
    """An op's tuning table: which backend serves each shape on each device."""

    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        key_columns: Sequence[str],
        bucket_column: str,
    ) -> None:
        self.rows: List[Dict[str, str]] = list(rows)
        self.key_columns: Tuple[str, ...] = tuple(key_columns)
        self.bucket_column: str = bucket_column
        # Keys are compared as strings, so key columns can hold any value
        # (sizes, dtypes, ...); only the bucket column is numeric.
        buckets: Dict[Tuple[str, ...], List[Tuple[int, str]]] = {}
        for row in rows:
            key = (row["gfx"], row["cu_num"]) + tuple(row[c] for c in self.key_columns)
            buckets.setdefault(key, []).append(
                (int(row[bucket_column]), row["backend"])
            )
        self._bounds: Dict[Tuple[str, ...], List[int]] = {}
        self._backends: Dict[Tuple[str, ...], List[str]] = {}
        for key, entries in buckets.items():
            entries.sort()
            self._bounds[key] = [bound for bound, _ in entries]
            self._backends[key] = [backend for _, backend in entries]

    @classmethod
    def from_package(
        cls,
        package: str,
        resource: str,
        key_columns: Sequence[str],
        bucket_column: str,
    ) -> "ShapeTable":
        text = importlib.resources.files(package).joinpath(resource).read_text()
        return cls(list(csv.DictReader(text.splitlines())), key_columns, bucket_column)

    def lookup(
        self, device: _DeviceKey, key: Sequence[object], value: int
    ) -> Optional[str]:
        """The backend for ``value`` (e.g. M) under ``key`` (e.g. (N, K)), or
        ``None`` when the table does not cover it."""
        full_key = tuple(str(k) for k in tuple(device) + tuple(key))
        bounds = self._bounds.get(full_key)
        # Empty calls (e.g. M=0) are never tuned; MSLK's kernel handles them.
        if bounds is None or value < 1:
            return None
        i = bisect.bisect_left(bounds, value)
        if i == len(bounds):
            return None
        return self._backends[full_key][i]

    def __len__(self) -> int:
        return sum(len(b) for b in self._bounds.values())


@functools.lru_cache(maxsize=None)
def device_key(device_index: int) -> _DeviceKey:
    """(gfx arch, CU count) of a GPU, the device part of every table key."""
    props = torch.cuda.get_device_properties(device_index)
    return props.gcnArchName.split(":")[0], props.multi_processor_count


_forced: Dict[str, str] = {}


@contextmanager
def force_backend(op: str, backend: str) -> Iterator[None]:
    """Make ``op`` use ``backend`` for every shape, ignoring its table and
    switches. For tests and tuning scripts, which need to run each backend on
    the same shapes.

    Not thread-safe: the setting is process-wide, so it applies to calls from
    every thread while active."""
    previous = _forced.get(op)
    _forced[op] = backend
    try:
        yield
    finally:
        if previous is None:
            _forced.pop(op, None)
        else:
            _forced[op] = previous


def choose_backend(
    op: str,
    table: Callable[[], ShapeTable],
    device_index: int,
    key: Sequence[object],
    value: int,
) -> str:
    """The backend that should serve this call of ``op``.

    ``table`` is a zero-argument loader, so that an op whose aiter path is
    switched off never reads its table.
    """
    forced = _forced.get(op)
    if forced is not None:
        return forced
    if not is_aiter_enabled(op):
        return MSLK
    return table().lookup(device_key(device_index), key, value) or MSLK


class OpRouting:
    """Everything that routes one op: its backends, its tuning table, and how a
    call maps onto the table. Calling it serves the call.

    An op's dispatch module (e.g. ``mslk/gemm/dispatch/f4f4bf16.py``) only
    declares these; choosing and calling are done here, the same way for every
    op:

    - ``backends`` maps each backend's name to a callable taking the op's
      arguments; MSLK's kernel (:data:`MSLK`) is the default and fallback.
    - ``key`` maps the op's arguments to the table key and bucket value, e.g.
      ``((N, K), M)``. The first argument must be a tensor on the call's device.
    - ``accepts`` optionally maps a backend's name to a predicate on the op's
      arguments; a call a backend cannot take goes to MSLK's kernel instead,
      even when the table or :func:`force_backend` chose that backend.
    - The table is a CSV in a package, read on first use.
    """

    def __init__(
        self,
        op: str,
        package: str,
        resource: str,
        key_columns: Sequence[str],
        bucket_column: str,
        backends: Dict[str, Callable[..., Any]],
        key: Callable[..., Tuple[Sequence[object], int]],
        accepts: Optional[Dict[str, Callable[..., bool]]] = None,
    ) -> None:
        assert MSLK in backends, "MSLK's kernel must be one of the backends"
        self.op = op
        self.backends = backends
        self._key = key
        self._accepts: Dict[str, Callable[..., bool]] = accepts or {}
        self._table_args = (package, resource, tuple(key_columns), bucket_column)

    @functools.cached_property
    def table(self) -> ShapeTable:
        return ShapeTable.from_package(*self._table_args)

    def backend_for(self, *args: Any, **kwargs: Any) -> str:
        """The backend that serves a call with these arguments."""
        key, value = self._key(*args, **kwargs)
        name = choose_backend(
            self.op, lambda: self.table, args[0].get_device(), key, value
        )
        accepts = self._accepts.get(name)
        if accepts is not None and not accepts(*args, **kwargs):
            return MSLK
        return name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.backends[self.backend_for(*args, **kwargs)](*args, **kwargs)
