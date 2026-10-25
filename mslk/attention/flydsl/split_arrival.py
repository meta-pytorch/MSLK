# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""In-kernel split-K merge by the last split to finish.

Each split stores its partial with agent-coherent (SC1) stores, then counts
itself in on a per-tile counter at the head of a persistent workspace. The
split that completes the count merges every partial and resets the counter to
zero for the next call, so the counters never need clearing on the host.
"""

from typing import Any

import flydsl.compiler as flyc  # pyre-ignore[21]
import flydsl.expr as fx  # pyre-ignore[21]
import torch
from flydsl._mlir.dialects import llvm  # pyre-ignore[21]
from flydsl.expr import gpu, rocdl  # pyre-ignore[21]
from flydsl.expr.typing import T  # pyre-ignore[21]
from flydsl.expr.utils.arith import _to_raw as as_mlir_value  # pyre-ignore[21]
from mslk.flydsl import buffer_ops

# Arrival counters (int32) at the head of the workspace; bounds the tile count.
ARRIVAL_COUNTERS = 1 << 16
# gfx94x/gfx950 SC1 cache-policy bit: an agent-scope coherent access. Stores
# write through the per-XCD L2 and loads do not hit a stale L2 line, so the
# partials cross XCDs without agent-scope fences (which would write back /
# invalidate the whole L2 per workgroup).
CPOL_SC1 = 16

# (device, stream) -> [workspace, used by a captured graph]. The counters are
# zero between calls: every launch resets the ones it used.
_WORKSPACES: dict = {}
# Workspaces a captured graph still addresses, kept alive for its replays.
_GRAPH_HELD: list = []
# device -> stream that zeroes workspaces created during graph capture.
_ZEROING_STREAMS: dict = {}


def _zeroed(numel: int, device: torch.device) -> torch.Tensor:
    """fp32 zeros. Under graph capture the whole buffer is zeroed right away on
    a side stream, so other graphs that share it do not depend on this one
    having run; this graph also clears the counters itself, which orders its
    own first replay after the zeroing. (Waiting on the side stream instead
    would add a node every replay pays for, and an event cannot be queried
    during capture; a later graph is captured and instantiated long after the
    side stream's memset was queued.)"""
    buf = torch.empty(numel, dtype=torch.float32, device=device)
    if not torch.cuda.is_current_stream_capturing():
        buf.zero_()
        return buf
    side = _ZEROING_STREAMS.get(device)
    if side is None:
        side = _ZEROING_STREAMS[device] = torch.cuda.Stream(device)
    with torch.cuda.stream(side):
        buf.zero_()
    buf[:ARRIVAL_COUNTERS].zero_()
    return buf


def arrival_workspace(elems: int, device: torch.device, stream: Any) -> torch.Tensor:
    """fp32 workspace of ARRIVAL_COUNTERS zeroed counters followed by `elems`.

    Buffers are per stream so concurrent streams never share counters, and
    grow geometrically so a slowly growing batch does not reallocate per call.
    A buffer a captured graph used is kept alive when outgrown. Graphs captured
    on one stream share its workspace, so they must not replay concurrently.
    """
    key = (device, stream.cuda_stream)
    need = ARRIVAL_COUNTERS + elems
    capturing = torch.cuda.is_current_stream_capturing()
    entry = _WORKSPACES.get(key)
    if entry is not None and entry[0].numel() >= need:
        entry[1] = entry[1] or capturing
        return entry[0]
    if entry is not None and entry[1]:
        _GRAPH_HELD.append(entry[0])
    size = max(need, 2 * entry[0].numel()) if entry is not None else need
    buf = _zeroed(size, device)
    _WORKSPACES[key] = [buf, capturing]
    return buf


def counter_ptr(ws_base_i64: Any, tile: Any) -> Any:  # pyre-ignore[2,3]
    """Global pointer to the arrival counter of `tile` (an Int32 or Index)."""
    addr = ws_base_i64 + fx.Int64(tile * 4)
    return buffer_ops.create_llvm_ptr(addr, address_space=1)


def arrive(
    counter: Any, flag_lds_offset: Any, leader: Any, num_splits: int
) -> Any:  # pyre-ignore[2,3]
    """Count this split in once its partial stores are done; returns a uniform
    Int32 that is nonzero in the split that arrived last.

    `flag_lds_offset` is a free LDS word for broadcasting the result and
    `leader` selects the one thread that touches the counter.
    """
    flag_ptr = buffer_ops.create_llvm_ptr(fx.Int64(flag_lds_offset), address_space=3)
    rocdl.s_waitcnt(0)
    gpu.barrier()

    @flyc.jit
    def _count_in():
        if leader:
            old = fx.llvm.atomic_add(
                counter,
                fx.Int32(1),
                syncscope=rocdl.SyncScope.Agent,
                ordering=fx.AtomicOrdering.Monotonic,
            )
            last = (fx.Int32(old) == fx.Int32(num_splits - 1)).select(
                fx.Int32(1), fx.Int32(0)
            )
            llvm.StoreOp(as_mlir_value(last), flag_ptr)

    _count_in()
    gpu.barrier()
    return fx.Int32(rocdl.readfirstlane(T.i32, llvm.LoadOp(T.i32, flag_ptr).result))


def reset_counter(counter: Any) -> None:  # pyre-ignore[2]
    """Zero the counter for the next call; done by the last split's leader."""
    llvm.StoreOp(as_mlir_value(fx.Int32(0)), counter)
