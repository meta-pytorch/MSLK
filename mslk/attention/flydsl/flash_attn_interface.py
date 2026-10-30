# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level FlyDSL Flash Attention API for gfx950 / gfx942.

Wraps ``flash_attn_generic.build_flash_attn_func_module`` (gfx942-compatible,
dense self/cross-attention) and ``flash_attn_gfx950.build_flash_attn_dualwave_swp_module``
(gfx950 DUALWAVE_SWP, varlen + split-K) behind a single function:

    ``flydsl_flash_attn_func(q, k, v, ...)``

Key features vs calling build_* directly:
- ``@functools.lru_cache`` on the build call so repeated invocations with the
  same (static) config compile only once per process.
- Explicit ``max_seqlen_q`` / ``cross_seqlen`` controls for varlen builds.
- split-K fp32 workspace allocation, zeroing, and the 4 GiB descriptor guard.
- Unified device / stream context (``torch.cuda.device`` + current stream).
- Validates shapes, dtypes, and arch before compiling.
- Accepts ``debug_counts`` tensor to enable the lazy-rescale branch counter
  (gfx950 DUALWAVE_SWP dualwave_swp_debug_lazy_counts=True path).

Known dualwave defects (tracked; routing below escapes affected cases to the
generic light kernel, which moves those users off the faster dualwave path):
  * f16 dualwave softmax overflows at large logits (narrow exponent) -> NaN, so
    f16 is forced to the light path (see ``_paged_light_ok`` / the varlen and
    dense f16 escapes below).
  * dualwave cross-attention NaNs for >=5 KV tiles / hardcodes 1/sqrt(D), so
    cross-length cases are kept off dualwave.
TODO(): Fix the flyDSL dualwave f16 cross-attn NaN
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: F401  (imported for callers' convenience)

# Re-export so callers only need to import from this module.
from .flash_attn_utils import dualwave_splitk_workspace_elems, WAVE_SPLITK_MIN_D
from .split_arrival import ARRIVAL_COUNTERS, arrival_workspace

__all__ = ["flydsl_flash_attn_func", "dualwave_splitk_workspace_elems"]

_DTYPE_MAP = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float8_e4m3fn: "fp8"}

# Short varlen/paged cases use the lightweight generic path.
_VARLEN_LIGHT_MAX_SEQ = 256
# Head dims the generic kernel builds; others compute at the next of these with
# the missing columns masked in-kernel.
_GENERIC_HEAD_DIMS = (64, 128, 256)


def _run_head_dim(d: int) -> int:
    """The head dim a call with stored head dim ``d`` computes at: the smallest of
    _GENERIC_HEAD_DIMS >= d, or 0 when ``d`` is above them all."""
    return next((hd for hd in _GENERIC_HEAD_DIMS if hd >= d), 0)


_DENSE_LIGHT_CU_FALLBACK = 256
_DENSE_DUALWAVE_MIN_SEQ = 256
_DENSE_DUALWAVE_LARGE_BATCH = 8
_RAW_BUFFER_MAX_BYTES = (1 << 32) - 1


def _validate_splitk_workspace_descriptor(
    num_heads: int, seq_len: int, head_dim: int, num_splits: int
) -> None:
    opart_batch_bytes = num_heads * seq_len * (head_dim // 2) * num_splits * 4
    if opart_batch_bytes > _RAW_BUFFER_MAX_BYTES:
        raise ValueError(
            "flydsl_flash_attn_func: split-K O-partial workspace per batch "
            f"({opart_batch_bytes} bytes) exceeds the raw-buffer descriptor limit "
            f"({_RAW_BUFFER_MAX_BYTES} bytes)"
        )


_DENSE_DUALWAVE_MIN_SEQ_LARGE_BATCH = 192
_DENSE_M256_MIN_TOKENS = 4096


def _dtype_str(t: torch.Tensor) -> str:
    s = _DTYPE_MAP.get(t.dtype)
    if s is None:
        raise ValueError(
            f"flydsl_flash_attn_func only supports bf16/f16/fp8, got {t.dtype!r}"
        )
    return s


def _gpu_arch(device: torch.device) -> str:
    try:
        return torch.cuda.get_device_properties(device.index).gcnArchName.split(":")[0]
    except Exception:
        return ""


def _dense_routes_to_dualwave(batch: int, seq_len: int) -> bool:
    if batch >= _DENSE_DUALWAVE_LARGE_BATCH:
        return seq_len >= _DENSE_DUALWAVE_MIN_SEQ_LARGE_BATCH
    return seq_len >= _DENSE_DUALWAVE_MIN_SEQ


def _dense_light_cu(device: torch.device) -> int:
    try:
        return int(torch.cuda.get_device_properties(device.index).multi_processor_count)
    except Exception:
        return _DENSE_LIGHT_CU_FALLBACK


def _dense_generic_tile(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_str: str,
    device: torch.device,
):
    if head_dim in (64, 128) and dtype_str in ("bf16", "f16"):
        main_blocks = batch * num_heads * ((seq_len + 127) // 128)
        if main_blocks < _dense_light_cu(device):
            # Bias too: block_m=128 would halve an already-underfilled grid.
            return 64, 128, "N32"
    if num_heads >= 32 and batch * seq_len >= _DENSE_M256_MIN_TOKENS:
        return 256, 512, "auto"
    return 128, 256, "auto"


# Generic split-K uses BLOCK_M=64, so the "mtile" (Q rows per workgroup) is 64.
_GENERIC_SPLITK_BLOCK_M = 64
# Cross length splits once the KV walk pays for the combine launch.
_CROSS_SPLITK_MIN_KV = 2048
# Dropout splits under the same floor, except at D>=128, where the keep-mask
# gather makes each KV tile dear enough to pay from ~1000 keys.
_DROPOUT_SPLITK_MIN_KV = 1000


def _generic_splitk_list(i: int) -> int:
    # Mirror CK generate_splits_list: 1,2,4,8,16,32,64,96,128,... .
    if i <= 0:
        return 1
    if i <= 5:
        return 1 << (i - 1)
    return (i - 5) * 32


def _num_kv_splits_heuristic(
    num_batches: int,
    num_heads: int,
    seqlen_q: int,
    head_dim: int,
    num_cu: int,
    max_splits: int = 8,
) -> int:
    """Port of CK get_num_kv_splits_heuristic for the generic BLOCK_M=64 kernel.

    Returns the number of KV splits (1 = no split-K). CK varies the mtile by
    head-dim, but the generic kernel always uses BLOCK_M=64, so the occupancy
    estimate uses mtile=64 (or 16 for tiny q, matching CK's smallq branch).
    """

    def ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    if head_dim > 256:
        return 1

    # Enough Q tiles to fill ~all CUs at the default pipeline mtile -> no split.
    if seqlen_q >= _GENERIC_SPLITK_BLOCK_M:
        default_blocks = (
            num_batches * num_heads * ceildiv(seqlen_q, _GENERIC_SPLITK_BLOCK_M)
        )
        if default_blocks >= 0.8 * num_cu:
            return 1

    mtile = _GENERIC_SPLITK_BLOCK_M
    if seqlen_q <= 16:
        mtile = 16
    blocks = num_batches * num_heads * ceildiv(seqlen_q, mtile)
    if blocks >= 0.9 * num_cu:
        return 1

    max_splits = min(max_splits, num_cu)
    max_check = 1
    while _generic_splitk_list(max_check) <= max_splits:
        max_check += 1

    num_splits = 1
    for i in range(2, max_check):
        num_splits = _generic_splitk_list(i)
        if blocks * num_splits >= num_cu:
            break
    return num_splits


# ── build-cache helpers ────────────────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def _build_dense(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    block_m: int,
    flat_work_group_size: int,
    path_tag: str,
    waves_per_eu: int,
    daz: bool,
    return_lse: bool = False,
    window_left: int = -1,
    sm_scale: Optional[float] = None,
    has_bias: bool = False,
    has_dropout: bool = False,
    num_kv_splits: int = 1,
    causal_top_left: bool = False,
    head_dim_qk: Optional[int] = None,
    head_dim_v: Optional[int] = None,
):
    """Build (and cache) one dense generic launcher variant."""
    from .flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        block_m=block_m,
        flat_work_group_size=flat_work_group_size,
        path_tag=path_tag,
        waves_per_eu=waves_per_eu,
        daz=daz,
        return_lse=return_lse,
        window_left=window_left,
        sm_scale=sm_scale,
        has_bias=has_bias,
        has_dropout=has_dropout,
        num_kv_splits=num_kv_splits,
        causal_top_left=causal_top_left,
    )


@functools.lru_cache(maxsize=256)
def _build_dense_dualwave(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
    return_lse: bool = False,
):
    """Build (and cache) the dense gfx950 DUALWAVE_SWP launcher."""
    from .flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_debug_lazy_counts=debug_lazy_counts,
        dualwave_swp_enable_stagger=enable_stagger,
        return_lse=return_lse,
    )


@functools.lru_cache(maxsize=128)
def _build_dense_fp8(
    num_heads: int,
    num_kv_heads: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    enable_stagger: bool,
):
    """Build (and cache) the dense gfx950 fp8 launcher."""
    from .flash_attn_fp8_gfx950 import build_flash_attn_dualwave_swp_fp8_module

    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=num_heads,
        head_dim=128,
        causal=causal,
        dtype_str="fp8",
        num_kv_heads=num_kv_heads,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
    )


@functools.lru_cache(maxsize=256)
def _build_varlen(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
    return_lse: bool = False,
):
    """Build (and cache) a varlen-mode launcher (gfx950 DUALWAVE_SWP, varlen=True)."""
    from .flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        varlen=True,
        cross_seqlen=cross_seqlen,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_debug_lazy_counts=debug_lazy_counts,
        dualwave_swp_enable_stagger=enable_stagger,
        return_lse=return_lse,
    )


@functools.lru_cache(maxsize=256)
def _build_varlen_light(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
    return_lse: bool = False,
    causal_top_left: bool = False,
    window_left: int = -1,
    sm_scale: Optional[float] = None,
    gappy_kv: bool = False,
    num_kv_splits: int = 1,
    head_dim_qk: Optional[int] = None,
    head_dim_v: Optional[int] = None,
):
    """Build a lightweight packed-varlen launcher for short attention."""
    from .flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        causal=causal,
        dtype_str=dtype_str,
        sm_scale=sm_scale,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        varlen=True,
        block_m=64,
        flat_work_group_size=128,
        waves_per_eu=waves_per_eu,
        daz=daz,
        return_lse=return_lse,
        causal_top_left=causal_top_left,
        window_left=window_left,
        gappy_kv=gappy_kv,
        num_kv_splits=num_kv_splits,
    )


@functools.lru_cache(maxsize=256)
def _build_splitk(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    num_kv_splits: int,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    enable_stagger: bool,
    return_lse: bool = False,
):
    """Build (and cache) a split-K launcher (gfx950 DUALWAVE_SWP, num_kv_splits>1)."""
    from .flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        num_kv_splits=num_kv_splits,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
        return_lse=return_lse,
    )


@functools.lru_cache(maxsize=256)
def _build_paged(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    waves_per_eu: int,
    daz: bool,
    lazy_rescale: bool,
    setprio: bool,
    enable_stagger: bool,
    num_kv_splits: int = 1,
    varlen: bool = False,
    kv_cache_layout: str = "linear",
):
    """Build (and cache) a paged-KV launcher (gfx950 DUALWAVE_SWP, paged=True).

    ``num_kv_splits > 1`` builds the paged + split-K variant (KV dimension split
    across grid_z = B*num_kv_splits workgroups + a combine pass), which fills the
    GPU for low-occupancy shapes (small B / few heads).

    ``varlen=True`` builds the packed-Q (cu_seqlens) + paged-KV variant: Q/O are
    ``[total_q, H, D]`` and K/V are the physical page cache, looked up via the
    block table per kv-tile. Mutually exclusive with split-K.

    ``kv_cache_layout`` selects the physical page layout: "linear"
    [NumBlocks,PageSize,Hkv,D] or "vectorized" (aiter 5D).
    """
    from .flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    return build_flash_attn_dualwave_swp_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        paged=True,
        varlen=varlen,
        num_kv_splits=num_kv_splits,
        cross_seqlen=cross_seqlen,
        kv_cache_layout=kv_cache_layout,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=lazy_rescale,
        dualwave_swp_setprio=setprio,
        dualwave_swp_enable_stagger=enable_stagger,
    )


# ── paged-KV native path ────────────────────────────────────────────────────

# gfx950 dualwave paged-KV currently supports exactly one configuration.
_PAGED_PAGE_SIZE = 64
_PAGED_BT_LDS_SIZE = 2048
_PAGED_METADATA_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

# Fold GQA query heads into the M dimension at q_len==1 (see the pack block in
# _flydsl_flash_attn_paged). Set to 0 to fall back to one workgroup per query head.
_PAGED_GQA_STRIDE = os.getenv("FLYDSL_PAGED_GQA_STRIDE", "1") == "1"
_PAGED_GQA_PACK = os.getenv("FLYDSL_PAGED_GQA_PACK", "1") == "1"


# Concurrency at which wave-split-K becomes worthwhile below
# WAVE_SPLITK_MIN_D. The trait builder cannot see batch and num_kv_splits
# does not stand in for it -- the batches that win and the batches that lose
# share a split count -- so the decision is made here and passed through as a
# single bool. A bool rather than the batch itself because the builders are
# lru_cached on their arguments: one more distinct value forks the cache, and
# only the answer matters. Below the cutoff the transform costs more than it
# saves -- steeply so at long context -- so the floor is not a rounding
# choice. 0 disables it entirely, leaving the head-dim bound alone.
_WAVE_SPLITK_MIN_BATCH = 8
_PAGED_LIGHT_BLOCK_M = int(os.getenv("FLYDSL_PAGED_BLOCK_M", "64"))


_UNIFORM_CU_SEQLENS: dict = {}


def _uniform_cu_seqlens(count: int, step: int, device: str) -> torch.Tensor:
    """Prefix sums for uniform sequence lengths, cached by (count, step, device).

    Decode replays the same shape indefinitely, so the cache saves an arange per
    call. A new entry is built on the current stream and waited for once, so
    any stream may read it afterwards. Nothing is stored during CUDA-graph
    capture (the tensor would live in the graph's pool), and entries are never
    evicted, since a captured graph may still point at one.
    """
    key = (count, step, device)
    cached = _UNIFORM_CU_SEQLENS.get(key)
    if cached is not None:
        return cached
    t = torch.arange(0, (count + 1) * step, step, dtype=torch.int32, device=device)
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.current_stream().synchronize()
        _UNIFORM_CU_SEQLENS[key] = t
    return t


def _paged_kv_cumsum(seqlen_k: torch.Tensor) -> torch.Tensor:
    """Exclusive scan of per-request KV lengths on the current stream."""
    return torch.nn.functional.pad(
        seqlen_k.to(torch.int32).cumsum(0, dtype=torch.int32), (1, 0)
    )


def _is_i32_vector(t: torch.Tensor) -> bool:
    return t.dtype == torch.int32 and t.is_contiguous()


def _normalize_paged_lengths(
    lengths: torch.Tensor,
    *,
    launch_kv_bound: int,
    block_table_capacity: int,
    kv_seqstart: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Bound paged metadata using tensor operations on the consuming stream."""
    bounded = lengths.to(torch.int64).clamp(0, launch_kv_bound)
    normalized_start = None
    if kv_seqstart is not None:
        normalized_start = kv_seqstart.to(torch.int64).clamp(0, block_table_capacity)
        bounded = torch.minimum(bounded, block_table_capacity - normalized_start)
    return bounded.to(torch.int32).contiguous(), (
        normalized_start.to(torch.int32).contiguous()
        if normalized_start is not None
        else None
    )


def _normalize_paged_cumulative_lengths(
    cu_seqlens_kv: torch.Tensor,
    *,
    launch_kv_bound: int,
    block_table_capacity: int,
    kv_seqstart: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    cu_seqlens_kv_i64 = cu_seqlens_kv.to(torch.int64)
    lengths = cu_seqlens_kv_i64[1:] - cu_seqlens_kv_i64[:-1]
    bounded, normalized_start = _normalize_paged_lengths(
        lengths,
        launch_kv_bound=launch_kv_bound,
        block_table_capacity=block_table_capacity,
        kv_seqstart=kv_seqstart,
    )
    return _paged_kv_cumsum(bounded), normalized_start


def _validate_paged_metadata(
    name: str,
    tensor: torch.Tensor,
    device: torch.device,
    rank: int,
    first_dim: int,
) -> None:
    if tensor.device != device:
        raise ValueError(
            f"flydsl_flash_attn_func: {name} must be on {device}, got {tensor.device}"
        )
    if tensor.dim() != rank or int(tensor.shape[0]) != first_dim:
        raise ValueError(
            f"flydsl_flash_attn_func: {name} must have rank {rank} and first "
            f"dimension {first_dim}, got shape {tuple(tensor.shape)}"
        )
    if tensor.dtype not in _PAGED_METADATA_DTYPES:
        raise ValueError(
            f"flydsl_flash_attn_func: {name} must have an integral dtype, "
            f"got {tensor.dtype}"
        )


# Paged split-K sizing. `_num_kv_splits_heuristic` (CK's) stops as soon as the
# workgroup count covers the CUs once, because it assumes a workgroup that has
# started is a workgroup making progress. That does not hold for the paged light
# kernel: each workgroup walks its KV range as a serial chain of dependent loads
# at roughly one memory latency per BLOCK_N_OUT tile, so a full GPU can still be
# idle-stalled. Size by chain length instead, then bound the workgroup count so
# the fp32 workspace and combine pass stay cheap.
_PAGED_BLOCK_N_OUT = 64  # generic paged builds with path_tag="N32"
# Target KV tiles per workgroup: short enough to hide the per-tile load
# latency, long enough that the combine pass stays cheap.
_PAGED_TARGET_CHAIN = 8
_PAGED_MAX_SPLITS = 64
# A base grid this small cannot fill the device even at MAX_SPLITS, so it is
# allowed twice the cap.
_PAGED_STARVED_BLOCKS = 8
# Ceiling on total workgroups (blocks * splits). Past this the combine pass and
# scheduling cost outweigh the shorter chain.
_PAGED_MAX_WG = 8192
# The fp32 split-K workspace is B*splits*H*Sq*(D/2+2) elements; cap it so a
# large-batch speculative-decode call cannot quietly allocate gigabytes.
_PAGED_WS_BUDGET_MB = 512


def _paged_affordable_splits(
    num_batches: int, num_heads: int, seqlen_q: int, head_dim: int
) -> int:
    """Largest split count whose fp32 workspace fits _PAGED_WS_BUDGET_MB.

    rows = B*splits*H*Sq and elems = rows*(D//2) + 2*rows (see
    dualwave_splitk_workspace_elems).
    """
    per_split_elems = num_batches * num_heads * max(seqlen_q, 1) * (head_dim // 2 + 2)
    budget_elems = _PAGED_WS_BUDGET_MB * 1024 * 1024 // 4
    return max(1, budget_elems // max(per_split_elems, 1))


def _paged_num_kv_splits(
    num_batches: int, num_heads: int, seqlen_q: int, seqlen_kv: int, head_dim: int
) -> int:
    """KV splits for the generic paged (light) kernel.

    Sized by chain length, not by workgroup count. The KV loop is a serial chain
    of dependent loads costing roughly one memory latency per BLOCK_N_OUT tile,
    so a grid that already covers every CU can still be latency-stalled.
    Pick the smallest split count that brings the chain to _PAGED_TARGET_CHAIN
    tiles, then clamp to the workspace budget.
    """

    def ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    kv_tiles = ceildiv(max(seqlen_kv, 1), _PAGED_BLOCK_N_OUT)
    if kv_tiles <= _PAGED_TARGET_CHAIN:
        return 1  # chain is already short; splitting only adds a combine pass

    blocks = num_batches * num_heads * ceildiv(max(seqlen_q, 1), 64)
    starved = blocks < _PAGED_STARVED_BLOCKS
    cap = _PAGED_MAX_SPLITS * (2 if starved else 1)
    # A starved grid is short of latency hiding, not of work, so it pays to
    # halve the chain and take the extra combine passes. Where the base grid
    # already has enough in flight the same target only costs, so it stays
    # keyed to the starvation test.
    chain = max(1, _PAGED_TARGET_CHAIN // 2 if starved else _PAGED_TARGET_CHAIN)
    want = min(ceildiv(kv_tiles, chain), cap)

    # Don't let the grid blow past the workgroup ceiling: with a large base grid
    # (high concurrency) the extra splits stop buying latency hiding and the
    # combine pass starts to dominate.
    by_grid = max(1, _PAGED_MAX_WG // max(blocks, 1))

    affordable = _paged_affordable_splits(num_batches, num_heads, seqlen_q, head_dim)
    limit = max(1, min(cap, by_grid, affordable, kv_tiles))
    splits = max(1, min(want, limit))
    # NUM_KV_SPLITS is compiled in, and the raw count moves with every ~512
    # tokens of context, so a growing decode would JIT a new kernel pair each
    # time. Powers of two keep it to a handful: round the target up, then back
    # down under every cap.
    pow2 = 1 << (max(want, 1) - 1).bit_length()
    while pow2 > limit:
        pow2 //= 2
    # Each split's pages must fit the block-table LDS window; never round down
    # past that, or a shape that ran would start raising.
    pages = ceildiv(max(seqlen_kv, 1), _PAGED_PAGE_SIZE)
    if ceildiv(pages, pow2) > _PAGED_BT_LDS_SIZE:
        return splits
    return pow2


# Windowed paged split-K sizing. A window bounds the KV walk to a short range,
# so chain length stops being the constraint and the fixed per-split cost
# dominates; size by grid instead, aiming for about _PAGED_WINDOW_TARGET_WG
# workgroups in total.
_PAGED_WINDOW_TARGET_WG = 1024
_PAGED_WINDOW_MIN_SPLITS = 2
_PAGED_WINDOW_MAX_SPLITS = 32


def _paged_window_num_kv_splits(
    num_batches: int, num_heads: int, seqlen_q: int, reach: int, head_dim: int
) -> int:
    """KV splits for a windowed paged call that can see `reach` keys."""
    kv_tiles = -(-max(reach, 1) // _PAGED_BLOCK_N_OUT)
    units = num_batches * num_heads * -(-max(seqlen_q, 1) // 64)
    affordable = _paged_affordable_splits(num_batches, num_heads, seqlen_q, head_dim)
    splits = max(_PAGED_WINDOW_TARGET_WG // max(units, 1), _PAGED_WINDOW_MIN_SPLITS)
    splits = max(1, min(splits, _PAGED_WINDOW_MAX_SPLITS, kv_tiles, affordable))
    return 1 << (splits.bit_length() - 1)


# Workgroups below which the grid cannot keep the device busy on its own.
_PAGED_OCC_TARGET_WG = 512


def _paged_block_m(gqa_packed: bool, seqlen_q: int) -> int:
    """Height of the Q tile the paged builder uses.

    Widened for packed GQA so the packed rows stay in one Q tile; a second tile
    re-reads the whole KV range.
    """
    return 128 if (gqa_packed and seqlen_q > 64) else _PAGED_LIGHT_BLOCK_M


def _paged_combine_cap(head_dim: int) -> int:
    """Split count past which the combine costs more than the shorter chain saves."""
    return 128 if head_dim >= 128 else 256


def _paged_occupancy_topup(
    splits: int,
    num_batches: int,
    num_heads: int,
    seqlen_q: int,
    span: int,
    head_dim: int,
    gqa_packed: bool,
) -> int:
    """Raise the split count when the grid alone cannot fill the device.

    Under _PAGED_OCC_TARGET_WG workgroups the kernel sits at one wave per SIMD
    with nothing to hide memory latency behind, and splitting is the only lever.
    The raise is bounded by the KV span, by _paged_combine_cap and by the
    workspace budget, and never lowers the count it is given.
    """

    def ceildiv(a: int, b: int) -> int:
        return -(-a // b)

    grid = (
        num_batches
        * num_heads
        * ceildiv(max(seqlen_q, 1), _paged_block_m(gqa_packed, seqlen_q))
    )
    if grid * splits >= _PAGED_OCC_TARGET_WG:
        return splits
    want = min(
        ceildiv(_PAGED_OCC_TARGET_WG, max(grid, 1)),
        _paged_combine_cap(head_dim),
        max(ceildiv(max(span, 1), _PAGED_BLOCK_N_OUT), 1),
        _paged_affordable_splits(num_batches, num_heads, seqlen_q, head_dim),
    )
    # Compiled into the kernel, so keep it a power of two.
    want = 1 << (want.bit_length() - 1) if want > 0 else 1
    return max(splits, want)


def _paged_wide_tile_split_cap(
    splits: int,
    num_batches: int,
    num_heads: int,
    seqlen_q: int,
    kv_tiles: int,
    head_dim: int,
    gqa_packed: bool,
) -> int:
    """Cap the split count for multi-token (> 32 row) full-attention tiles.

    Targets ~512 workgroups in flight at D=128 and ~2048 at D=64, with >= 4 KV
    tiles per split. Only ever lowers the chain-length sizing.
    """
    rows = seqlen_q
    if rows <= 32:
        return splits
    grid = num_batches * num_heads * -(-rows // _paged_block_m(gqa_packed, rows))
    target = 512 if head_dim >= 128 else 2048
    want = max(target / max(grid, 1), 1)
    lo = 1 << (int(want).bit_length() - 1)
    by_grid = lo * 2 if want / lo > 1.414 else lo
    by_chain = 1 << (max(kv_tiles // 4, 1).bit_length() - 1)
    return min(splits, max(8, min(by_grid, by_chain)))


def _paged_narrow_tile(
    rows: int,
    window_left: int,
    q_len: int,
    kv_len: int,
    num_kv_splits: int,
    head_dim: int,
    groups: int,
) -> bool:
    """Use a 32-row, single-wave Q tile instead of the 64-row default.

    With <= 32 packed rows the 64-row tile's second wave is all padding; halving
    the tile lets more workgroups share a CU. Only used for long KV walks, since
    a single wave issues each tile's loads alone. A single-token D=128 decode of
    one request (``groups`` = batch x KV heads) keeps the 64-row tile for wave
    split-K over KV columns.
    """
    if rows > 32:
        return False
    if window_left < 0 and q_len == 1 and head_dim >= WAVE_SPLITK_MIN_D and groups <= 4:
        return False
    span = min(kv_len, window_left + q_len) if window_left >= 0 else kv_len
    tiles = -(-max(span, 1) // _PAGED_BLOCK_N_OUT)
    return -(-tiles // max(num_kv_splits, 1)) >= 16


# Single-wave tiles resident per CU (LDS-bound): the narrow-chain trade keeps
# at least this many per CU in flight.
_PAGED_NARROW_WG_PER_CU = 4


def _paged_narrow_chain_splits(
    splits: int,
    groups: int,
    rows: int,
    window_left: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    page_size: int,
    min_wg: int,
) -> int:
    """Trade splits for chain length when that earns the single-wave tile.

    Packed rows that fit 32 run on one wave, with the next K tile prefetched,
    once a split walks >= 16 tiles. Chain-length sizing stops near 8, which
    keeps the 64-row tile whose second wave is all padding. Halve the split
    count until the narrow tile applies, while the grid (``groups`` workgroups
    per split at <= 32 rows) still fills ``min_wg`` workgroups.
    """
    if rows > 32:
        return splits

    def narrow(s: int) -> bool:
        return _paged_narrow_tile(rows, window_left, q_len, kv_len, s, head_dim, groups)

    s = splits
    pages = -(-max(kv_len, 1) // page_size)
    while s > 1 and not narrow(s):
        half = s // 2
        if groups * half < min_wg or -(-pages // half) > _PAGED_BT_LDS_SIZE:
            break
        s = half
    return s if narrow(s) else splits


# Minimum KV tiles per workgroup for paged K double buffering.
_PAGED_KV_DMA_MIN_ITERS = 63


def _paged_kv_dma(
    block_m: int, head_dim: int, window_left: int, kv_len: int, num_kv_splits: int
) -> bool:
    """Double-buffer K for a tall Q tile at D=64 over enough tiles, no window."""
    if block_m <= _PAGED_LIGHT_BLOCK_M or head_dim != 64 or window_left >= 0:
        return False
    tiles = -(-max(kv_len, 1) // _PAGED_BLOCK_N_OUT)
    return -(-tiles // max(num_kv_splits, 1)) >= _PAGED_KV_DMA_MIN_ITERS


# Minimum KV tiles per workgroup for the paged K register prefetch. D=128 holds
# twice the prefetched K in registers and needs a longer walk to pay for it.
_PAGED_K_PREFETCH_MIN_ITERS_D64 = 3
_PAGED_K_PREFETCH_MIN_ITERS_D128 = 16


def _paged_k_prefetch(
    head_dim: int, window_left: int, q_len: int, kv_len: int, num_kv_splits: int
) -> bool:
    """Prefetch the next K tile into registers when each workgroup walks enough tiles."""
    span = min(kv_len, window_left + q_len) if window_left >= 0 else kv_len
    tiles = -(-max(span, 1) // _PAGED_BLOCK_N_OUT)
    need = (
        _PAGED_K_PREFETCH_MIN_ITERS_D64
        if head_dim == 64
        else _PAGED_K_PREFETCH_MIN_ITERS_D128
    )
    return -(-tiles // max(num_kv_splits, 1)) >= need


# Routes the paged decodes chosen below to pa_decode_swa when set.
_PAGED_SWA_KERNEL = True


def _paged_swa_kernel_config(
    q,
    k,
    v,
    causal,
    window_left,
    return_lse,
    varlen,
    vectorized,
    kv_seqstart,
    seqlen_k,
    block_table,
    out,
    num_kv_splits=0,
):
    """(num_waves, num_splits) to run a paged decode on pa_decode_swa, or None.

    Each CTA streams its slice of the keys through one CU. Windowed decodes use
    it once the (request, KV head) groups, split as needed, fill the GPU;
    full-attention decodes over a few groups split each group across the GPU.
    Other shapes stay on the generic split-K kernel.
    """
    if (
        not _PAGED_SWA_KERNEL
        or varlen
        or vectorized
        or not causal
        or return_lse
        or kv_seqstart is not None
        or seqlen_k is None
        or block_table is None
        or q.dim() != 4
        or k.dim() != 4
        or v.shape != k.shape
        # The checks the other routes make later: page size, head dim.
        or k.shape[1] != _PAGED_PAGE_SIZE
        or k.shape[3] != q.shape[3]
        # A caller-set split count is honoured by the generic kernel.
        or num_kv_splits > 1
        or seqlen_k.dtype != torch.int32
        or not seqlen_k.is_contiguous()
        or block_table.dtype != torch.int32
        or not block_table.is_contiguous()
        or (out is not None and out.stride() != q.stride())
    ):
        return None
    from .decode.pa_decode_swa import kv_span_elems, pa_decode_swa_supported

    B, Sq, H, D = q.shape
    H_kv = k.shape[2]
    if H_kv == 0 or H % H_kv or 16 % (H // H_kv):
        return None
    groups = B * H_kv
    m_tiles = -(-Sq // (16 // (H // H_kv)))
    cus = _dense_light_cu(q.device)
    # Split counts are compiled in; powers of two keep the variants few.

    def pow2_floor(n: int) -> int:
        return 1 << (n.bit_length() - 1)

    # Splits bring the grid to about one CTA per CU.
    splits = pow2_floor(max(1, min(8, cus // groups)))
    cfg = None
    if window_left < 0:
        # Full attention over a few requests and moderate contexts: the CTAs
        # of a (request, KV head) split its keys, merged by the reduce kernel.
        # With more groups than CUs there is nothing to split across.
        keys = block_table.shape[1] * k.shape[1]
        if groups > cus or groups * keys > 1 << 18:
            cfg = None
        elif m_tiles == 1:
            cfg = (2, pow2_floor(cus // groups))
        elif m_tiles == 2 and D == 64:
            cfg = (4, pow2_floor(cus // groups))
    elif m_tiles == 1:
        if D == 128:
            # Below cus/4 groups, 16 splits each merged by the reduce kernel.
            if groups >= cus:
                cfg = (4, 1)
            else:
                cfg = (2, splits if groups >= cus // 4 else 16)
        elif D == 64 and groups >= cus // 8:
            if groups >= cus:
                cfg = (2, 1)
            else:
                cfg = (4, 1) if groups >= cus // 2 else (4, splits)
    elif groups >= cus // (8 if D == 64 else 4):
        cfg = (4, splits)
    if cfg is None or not pa_decode_swa_supported(
        head_dim=D,
        num_heads=H,
        num_kv_heads=H_kv,
        seqlen_q=Sq,
        page_size=k.shape[1],
        window_left=window_left,
        dtype=q.dtype,
        kv_capacity=block_table.shape[1] * k.shape[1],
        num_waves=cfg[0],
        num_splits=cfg[1],
        kv_span=max(kv_span_elems(k), kv_span_elems(v)),
    ):
        return None
    return cfg


def _paged_pid_regs(max_pages_per_split: int) -> bool:
    """Hold a split's page ids in registers, one tile per lane, when they fit.

    Not used with K prefetch, which already loads each page id a tile early.
    """
    return max_pages_per_split + 1 <= 64


def _paged_fused_combine(
    splitk: bool,
    return_lse: bool,
    workgroups_x: int,
    num_kv_splits: int,
    head_dim: int,
    tile_rows: int,
) -> bool:
    """Merge split-K partials in the attention kernel instead of a combine kernel.

    The merge runs in the last workgroup to finish, so it is only used when there
    are many tiles and few splits per tile to merge.
    """
    if not splitk or return_lse or workgroups_x > ARRIVAL_COUNTERS:
        return False
    return (
        num_kv_splits <= 8 and workgroups_x >= 64 and (head_dim <= 64 or tile_rows <= 8)
    )


# Streaming tile of the head-packed decode kernel
# (TILE_N in decode/pa_decode_gfx950.py).
# A page must be a whole number of tiles so a tile never straddles two pages.
_PAGED_DECODE_TILE_N = 32

# Routes paged short-query decode to the head-packed decode kernel
# (`decode/pa_decode_gfx950.py`) when set to False.
_DISABLE_PAGED_DECODE_HP = False

# The head-packed decode kernel maps the MFMA M-axis to (query-token, head) pairs,
# so it needs `ratio` to divide MFMA_M and `ratio * Sq` to fit the M-tile budget.
# Mirrors MFMA_M / MAX_M_TILES_BY_HEAD_DIM in decode/pa_decode_gfx950.py.
_HP_MFMA_M = 16

# One CTA is one wave here, so CTA count is the wave count. Below this the kernel
# cannot hide memory latency: more M-tiles cost occupancy, and with too few CTAs
# there is nothing else resident to cover the stall.
_HP_MIN_CTAS_PER_CU = 2

# Tile count at which the occupancy cost becomes worth guarding. Shallow tiling
# keeps enough waves resident to hide latency on its own, so the floor must not
# reject it.
_HP_FLOOR_MIN_TILES = 3

# Query groups to use when one pass would exceed the register budget; more
# groups cost more launches than the occupancy they buy.
_HP_QUERY_GROUPS = 2

# Lets shapes that only fit the head-packed kernel via query grouping use it.
_DISABLE_PAGED_QGROUPS = False


def _hp_decode_ok(
    num_heads: int,
    num_kv_heads: int,
    seqlen_q: int,
    head_dim: int,
    num_batches: int,
    max_kv_pages: int,
    device,
) -> int:
    """How should this shape run on the head-packed decode kernel?

    Returns the number of query groups to split the block into, or 0 to decline.
    1 is the ordinary single pass.

    Three derived gates -- no tuned table:

    1. **Register budget.** `ratio * Sq` pairs must fit in `MFMA_M` slots across
       at most `max_m_tiles(head_dim)` tiles, the point before the compiler
       spills.
    2. **Query grouping.** A block too deep for that budget can instead be run as
       `_HP_QUERY_GROUPS` shallower passes (see `query_group_seqlen`). Costs one
       extra KV pass per group but avoids the spill. Only used to rescue a shape
       the budget would reject: where single-pass already fits, the gain does
       not pay for the extra traffic.
    3. **Parallelism floor.** The resulting CTA count must reach
       `_HP_MIN_CTAS_PER_CU * CUs`. More tiles buy fewer KV passes but cost
       occupancy, and that trade only pays when enough CTAs are resident.
    """
    from .decode.pa_decode_dense import auto_split_k_hp
    from .decode.pa_decode_gfx950 import max_m_tiles

    if _DISABLE_PAGED_QGROUPS:
        groups_allowed = 1
    else:
        groups_allowed = _HP_QUERY_GROUPS

    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return 0
    ratio = num_heads // num_kv_heads
    if not (1 <= ratio <= _HP_MFMA_M) or _HP_MFMA_M % ratio != 0:
        return 0
    t_pack = _HP_MFMA_M // ratio  # query tokens per M-tile
    budget = t_pack * max_m_tiles(head_dim)

    groups = 1
    if seqlen_q > budget:
        # Too deep for one pass: can grouping bring it under the budget?
        if (
            groups_allowed > 1
            and seqlen_q % groups_allowed == 0
            and seqlen_q // groups_allowed <= budget
        ):
            groups = groups_allowed
        else:
            return 0
    if seqlen_q < 1:
        return 0

    # Only deep tiling trades enough occupancy away to need the floor; shallow
    # tiling still leaves enough waves resident. Grouping makes each pass
    # shallower, so the depth that matters is the per-group one.
    tiles = -(-(seqlen_q // groups) // t_pack)
    if tiles < _HP_FLOOR_MIN_TILES:
        return groups
    num_cu = _dense_light_cu(device)
    split_k = auto_split_k_hp(
        num_batches, 1, num_heads, num_kv_heads, max_kv_pages * _PAGED_PAGE_SIZE
    )
    ctas = num_batches * num_kv_heads * split_k
    return groups if ctas >= _HP_MIN_CTAS_PER_CU * num_cu else 0


def _paged_auto_num_kv_splits(
    num_batches: int,
    num_heads: int,
    seqlen_q: int,
    q_len: int,
    seqlen_kv: int,
    head_dim: int,
    page_size: int,
    window_left: int,
    gqa_packed: bool,
    num_cu: int,
) -> int:
    """Split count for a generic paged call that left it to us.

    ``seqlen_q`` counts the (possibly GQA-packed) rows and ``q_len`` the query
    tokens. Sized by chain length (or, windowed, by the keys the window
    reaches), topped up to fill the device, capped for wide tiles, traded for
    the narrow tile where that pays, and finally raised so every split's pages
    fit the block-table LDS window.
    """
    if window_left >= 0:
        # The kernel only walks the keys the window can reach and splits that
        # range, so size by it rather than by the whole context.
        span = min(seqlen_kv, window_left + q_len)
        splits = _paged_window_num_kv_splits(
            num_batches, num_heads, seqlen_q, span, head_dim
        )
    else:
        span = seqlen_kv
        splits = _paged_num_kv_splits(
            num_batches, num_heads, seqlen_q, seqlen_kv, head_dim
        )
    splits = _paged_occupancy_topup(
        splits, num_batches, num_heads, seqlen_q, span, head_dim, gqa_packed
    )
    if window_left < 0:
        splits = _paged_wide_tile_split_cap(
            splits,
            num_batches,
            num_heads,
            seqlen_q,
            -(-seqlen_kv // _PAGED_BLOCK_N_OUT),
            head_dim,
            gqa_packed,
        )
    splits = _paged_narrow_chain_splits(
        splits,
        num_batches * num_heads,
        seqlen_q,
        window_left,
        q_len,
        seqlen_kv,
        head_dim,
        page_size,
        _PAGED_NARROW_WG_PER_CU * num_cu,
    )
    # The block-table window spans the whole context, windowed or not. Raise
    # the count to cover it only while the workspace stays in budget; past that
    # the LDS-window validation rejects the shape before allocating.
    floor = -(-(-(-seqlen_kv // page_size)) // _PAGED_BT_LDS_SIZE)
    if splits < floor:
        floor_pow2 = 1 << (floor - 1).bit_length()
        affordable = _paged_affordable_splits(
            num_batches, num_heads, seqlen_q, head_dim
        )
        if floor_pow2 <= affordable:
            splits = floor_pow2
        elif floor <= affordable:
            splits = floor
    return splits


def _flydsl_flash_attn_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    num_kv_heads: Optional[int],
    block_table: Optional[torch.Tensor],
    seqlen_k: Optional[torch.Tensor],
    max_seqlen_kv: Optional[int],
    kv_cache_layout: str,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_kv: Optional[torch.Tensor],
    kv_seqstart: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    cross_seqlen: Optional[bool],
    num_kv_splits: int,
    return_lse: bool,
    sm_scale: Optional[float],
    out: Optional[torch.Tensor],
    waves_per_eu: int,
    daz: bool,
    dualwave_swp_lazy_rescale: bool,
    dualwave_swp_setprio: bool,
    dualwave_swp_enable_stagger: bool,
    stream,
    # Sliding window (exclusive left bound), forwarded to the generic (light)
    # paged kernel and pa_decode_swa.
    window_left: int = -1,
) -> torch.Tensor:
    """Native paged-KV attention on the gfx950 dualwave kernel.

    Supported config ONLY (anything else raises): linear/vectorized cache layout
    [NumBlocks, PageSize=64, NumKVHeads, HeadDim], vLLM lookup (block_table +
    seqlen_k), causal, D=64/128, dtype bf16/f16.
    - Dense 4D Q ``[B, Sq, H, D]``: split-K supported at any Sq. On the light
      route ``num_kv_splits`` 0 and 1 size the count from KV-chain length
      (``_paged_auto_num_kv_splits``); ``>1`` forces.
    - Varlen packed Q ``[total_q, H, D]`` (cu_seqlens_q given): paged K/V looked up
      per kv-tile via block_table; split-K only when every sequence has the same
      query length.
    """
    if kv_cache_layout not in ("linear", "vectorized"):
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports kv_cache_layout in ('linear','vectorized'), "
            f"got {kv_cache_layout!r}"
        )
    # Gappy paged uses kv_seqstart + cu_seqlens_kv (per-seq lengths) instead of the
    # vLLM seqlen_k bound.
    if block_table is None or (seqlen_k is None and kv_seqstart is None):
        raise ValueError(
            "flydsl_flash_attn_func: native paged KV (vllm) requires block_table and "
            "seqlen_k (or kv_seqstart for gappy paged)"
        )
    vectorized = kv_cache_layout == "vectorized"
    if vectorized:
        # aiter 5D: K [NumBlocks, Hkv, D/kVS, PageSize, kVS], V [NumBlocks, Hkv, PageSize/kVS, D, kVS].
        if k.dim() != 5 or v.dim() != 5:
            raise ValueError(
                f"flydsl_flash_attn_func: vectorized paged K/V must be 5D, got K{k.dim()}D V{v.dim()}D"
            )
    else:
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                "flydsl_flash_attn_func: linear paged K/V must both be 4D "
                f"[NumBlocks,PageSize,Hkv,D], got K{k.dim()}D V{v.dim()}D"
            )
        if k.shape != v.shape:
            raise ValueError(
                "flydsl_flash_attn_func: linear paged K/V must have matching "
                f"shapes, got K{tuple(k.shape)} V{tuple(v.shape)}"
            )

    varlen = cu_seqlens_q is not None
    # Dense paged callers hand us a per-request `seqlen_k` but the launch only
    # ever forwarded the batch maximum, so every request was masked to that
    # maximum and any shorter one read stale cache. Feed the kernel the real
    # lengths; VARLEN already does this, dense simply never did.
    _dense_kv_lens = not varlen and seqlen_k is not None
    dtype_str = _dtype_str(q)
    if varlen:
        # Packed varlen Q: [total_q, H, D]. Per-batch ranges come from cu_seqlens
        # inside the kernel; grid_y is sized by max_seqlen_q.
        if cu_seqlens_kv is None:
            raise ValueError(
                "flydsl_flash_attn_func: varlen paged KV requires cu_seqlens_kv"
            )
        if max_seqlen_q is None:
            raise ValueError(
                "flydsl_flash_attn_func: varlen paged KV requires max_seqlen_q"
            )
        if q.dim() != 3:
            raise ValueError(
                f"flydsl_flash_attn_func: varlen paged q must be 3D [total_q,H,D], got {q.dim()}D"
            )
        _total_q, H, D = q.shape
        B = cu_seqlens_q.numel() - 1
        Sq = int(max_seqlen_q)
        _varlen_uniform_q = _total_q == B * Sq
    else:
        _varlen_uniform_q = True
        if q.dim() != 4:
            raise ValueError(
                f"flydsl_flash_attn_func: paged dense q must be 4D [B,Sq,H,D], got {q.dim()}D"
            )
        B, Sq, H, D = q.shape
    # A non-native head dim runs the generic kernel at the next native one (D_run),
    # which reads Q/K/V at their stored width D -- the cache in place -- and masks
    # the missing columns.
    _masked_hd = D not in _GENERIC_HEAD_DIMS
    D_run = _run_head_dim(D)
    if _masked_hd and (D % 2 or not D_run):
        raise ValueError(
            f"flydsl_flash_attn_func: head_dim ({D}) must be even and <= 256"
        )
    if vectorized and _masked_hd:
        raise NotImplementedError(
            "flydsl_flash_attn_func: vectorized paged KV supports head_dim 64 or 128 "
            f"only, got {D}"
        )

    _validate_paged_metadata("block_table", block_table, q.device, 2, B)
    if seqlen_k is not None:
        _validate_paged_metadata("seqlen_k", seqlen_k, q.device, 1, B)
    if cu_seqlens_q is not None:
        _validate_paged_metadata("cu_seqlens_q", cu_seqlens_q, q.device, 1, B + 1)
    if cu_seqlens_kv is not None:
        _validate_paged_metadata("cu_seqlens_kv", cu_seqlens_kv, q.device, 1, B + 1)
    if kv_seqstart is not None:
        _validate_paged_metadata("kv_seqstart", kv_seqstart, q.device, 1, B)
    if B > 0 and int(block_table.shape[1]) == 0:
        raise ValueError(
            "flydsl_flash_attn_func: block_table must contain at least one page column"
        )

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if torch.device(launch_stream.device) != q.device:
            raise ValueError(
                "flydsl_flash_attn_func: stream must be on "
                f"{q.device}, got {launch_stream.device}"
            )

    requested_num_kv_splits = int(num_kv_splits)
    if requested_num_kv_splits < 0:
        raise ValueError(
            "flydsl_flash_attn_func: num_kv_splits must be nonnegative, "
            f"got {requested_num_kv_splits}"
        )
    block_table_stride = int(block_table.shape[1])
    block_table_capacity = block_table_stride * int(
        k.shape[3] if vectorized else k.shape[1]
    )
    if block_table_capacity > torch.iinfo(torch.int32).max:
        raise ValueError(
            "flydsl_flash_attn_func: block-table capacity exceeds the int32 "
            f"kernel ABI limit; got {block_table_capacity}"
        )
    skv = block_table_capacity if max_seqlen_kv is None else int(max_seqlen_kv)
    if not 0 <= skv <= block_table_capacity:
        raise ValueError(
            "flydsl_flash_attn_func: max_seqlen_kv must be between 0 and "
            f"block-table capacity {block_table_capacity}; got {skv}"
        )
    # The kernels read int32 lengths and clamp each to [0, skv] themselves, so
    # int32 metadata goes to them as is. Wider lengths are clamped here, before
    # narrowing; gappy lengths also depend on their starts (see the launch).
    if seqlen_k is not None and kv_seqstart is None and not _is_i32_vector(seqlen_k):
        with torch.cuda.device(q.device.index), torch.cuda.stream(launch_stream):
            seqlen_k, _ = _normalize_paged_lengths(
                seqlen_k,
                launch_kv_bound=skv,
                block_table_capacity=block_table_capacity,
            )

    if B == 0:
        empty_out = torch.empty_like(q) if out is None else out
        if return_lse:
            empty_lse = torch.empty((B, H, Sq), dtype=torch.float32, device=q.device)
            return empty_out, empty_lse
        return empty_out

    # The decode kernels below build native head dims only; a masked head dim
    # takes the generic kernel.
    _swa_cfg = (
        None
        if _masked_hd
        else _paged_swa_kernel_config(
            q,
            k,
            v,
            causal,
            window_left,
            return_lse,
            varlen,
            vectorized,
            kv_seqstart,
            seqlen_k,
            block_table,
            out,
            requested_num_kv_splits,
        )
    )
    if _swa_cfg is not None:
        from .decode.pa_decode_swa import pa_decode_swa_launch

        with torch.cuda.device(q.device.index), torch.cuda.stream(launch_stream):
            if out is None:
                # The kernel addresses out with q's strides.
                out = torch.empty_strided(
                    q.size(), q.stride(), dtype=q.dtype, device=q.device
                )
            return pa_decode_swa_launch(
                q,
                k,
                v,
                block_table,
                seqlen_k,
                window_left,
                sm_scale if sm_scale is not None else q.shape[-1] ** -0.5,
                out,
                num_waves=_swa_cfg[0],
                num_splits=_swa_cfg[1],
                max_seqlen_kv=skv,
            )

    # ── GQA head packing ────────────────────────────────────────────────────
    # The grid is one workgroup per (request, QUERY head), so the group_size
    # query heads sharing a KV head each stream that head's entire KV range
    # independently, so the kernel is L2-bound on redundant reads.
    #
    # Fold the group's query heads into M instead: one workgroup per
    # (request, KV head), KV read once and shared across the group. Rows are
    # laid out head-major, row = h * q_len + t, and the kernel is built with
    # Q_PACK_QLEN = q_len so its causal bound uses t = row % q_len rather than
    # the row index. Without that trait this is only expressible at q_len == 1.
    # Resolve the KV head count before any route that keys on it (GQA packing
    # below would otherwise decline every caller that leaves it unset).
    if num_kv_heads is None:
        num_kv_heads = int(k.shape[1] if vectorized else k.shape[2])
    _paged_window = window_left >= 0
    _gqa_packed = False
    _gqa_group = 0
    _gqa_qlen = 0
    _gqa_B = B
    _gqa_heads = H
    _gqa_varlen = varlen
    _gqa_user_out = None
    _gqa_stride_packed = False
    _gqa_rebuild_cu_q = False
    if (
        _PAGED_GQA_PACK
        and not return_lse
        and kv_seqstart is None
        and 1 <= Sq <= 64
        and _varlen_uniform_q
        and num_kv_heads is not None
        and num_kv_heads > 0
        and H > num_kv_heads
        and H % num_kv_heads == 0
        and (Sq == 1 or causal)
        and (causal or not _paged_window)
    ):
        _gqa_group = H // num_kv_heads
        _gqa_qlen = Sq
        rows = _gqa_group * Sq
        # At q_len == 1 a packed row *is* a group member, and the caller's
        # [B, 1, H_q, D] buffer already stores the group contiguously: row g sits
        # at g * D and KV head h at h * group * D. That is exactly what the
        # kernel addresses once it is told the head stride (Q_PACK_GROUP), so the
        # permute + copy is unnecessary. CUTLASS's Blackwell gen kernel does the
        # same thing, folding the GQA group into the M mode and pointing M's
        # stride at Q's head stride (sm100_fmha_gen_kernel).
        # Longer query blocks work too: a row encodes (group, token), which is
        # two terms rather than one stride, but both divisors are compile-time
        # constants -- see _qo_row_offset. The outer gate already requires
        # `causal` whenever Sq > 1, so Q_PACK_QLEN is set and the row decode is
        # well defined.
        #
        # Varlen is fine here too: the outer guard already requires uniform
        # q_seqlens, so at q_len == 1 the packed [total_q, H_q, D] buffer has the
        # same bytes in the same order as the dense one. The kernel derives the
        # batch offset from batch_idx under Q_PACK_GROUP rather than from
        # cu_seqlens, which counts packed rows and would be short by NUM_HEADS_Q.
        _gqa_stride_packed = _PAGED_GQA_STRIDE
        # [.., q_len, Hkv, group, D] -> [.., group, q_len, Hkv, D]; the flat M
        # index is h * q_len + t, matching Q_PACK_QLEN's row % q_len.
        if not _gqa_stride_packed:
            with torch.cuda.stream(launch_stream):
                q = (
                    q.view(B, Sq, num_kv_heads, _gqa_group, D)
                    .permute(0, 3, 1, 2, 4)
                    .reshape(
                        (B * rows, num_kv_heads, D)
                        if varlen
                        else (B, rows, num_kv_heads, D)
                    )
                    .contiguous()
                )
        if varlen:
            _gqa_rebuild_cu_q = True
            max_seqlen_q = rows
            _total_q = B * rows
        if not _gqa_stride_packed:
            # The caller's `out` is in the unpacked layout, so compute into a
            # fresh packed buffer and write the result back at the end. Stride
            # packing writes the unpacked layout directly, so `out` is kept.
            _gqa_user_out, out = out, None
        _gqa_heads = H
        H = num_kv_heads
        Sq = rows
        _gqa_packed = True

    if vectorized:
        kvs = 16 // q.element_size()
        page_size = int(k.shape[3])
        k_head_dim = int(k.shape[2]) * int(k.shape[4])  # (D/kVS) * kVS
        if int(k.shape[4]) != kvs:
            raise ValueError(
                "flydsl_flash_attn_func: vectorized K last dim "
                f"({k.shape[4]}) must equal kVS={kvs}"
            )
    else:
        page_size = int(k.shape[1])
        k_head_dim = int(k.shape[3])
    # Gappy paged uses the generic kernel (any D the generic path builds; the op
    # reshapes the cache to 64-row sub-pages), so skip the native-paged D/page gates.
    _gappy = kv_seqstart is not None
    if page_size != _PAGED_PAGE_SIZE and not _gappy:
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports page_size={_PAGED_PAGE_SIZE} only, got {page_size}"
        )
    if D_run not in (64, 128) and not _gappy:
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports head_dim=64 or 128, got {D}"
        )
    if k_head_dim != D:
        raise ValueError(
            f"flydsl_flash_attn_func: paged K head_dim ({k_head_dim}) must match q head_dim ({D})"
        )

    if H % num_kv_heads != 0:
        raise ValueError(
            f"flydsl_flash_attn_func: num_heads ({H}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    # ── route + split-K sizing ──────────────────────────────────────────────
    # Two paged kernels: the generic "light" one and the gfx950 dualwave one. The
    # dualwave paged softmax overflows for f16 (narrow exponent) and on the causal
    # path at large logits, so those route to light regardless of seqlen; bf16
    # non-causal keeps the faster dualwave kernel. Decided here (rather than at the
    # launch site) because split-K sizing depends on which kernel runs.
    _arch = _gpu_arch(q.device)
    _splitk_dtype_ok = D_run in (64, 128) and dtype_str in ("bf16", "f16")
    _paged_light_ok = (
        _gappy
        or _paged_window
        or _masked_hd
        or (
            _splitk_dtype_ok
            and (
                # Dualwave takes per-request KV lengths only via its varlen
                # variant, which has no split-K.
                (_dense_kv_lens and requested_num_kv_splits > 1)
                or dtype_str == "f16"
                or causal
                or not _arch.startswith("gfx950")
                or Sq <= _VARLEN_LIGHT_MAX_SEQ
            )
        )
    )

    if varlen and not _varlen_uniform_q and requested_num_kv_splits > 1:
        raise NotImplementedError(
            "flydsl_flash_attn_func: paged split-K requires uniform varlen Q"
        )
    _auto_sized = False
    if varlen and not _varlen_uniform_q:
        num_kv_splits = 1
    elif (
        _paged_light_ok
        and requested_num_kv_splits <= 1
        and _splitk_dtype_ok
        and not _gappy
    ):
        # Split-K is the lever on decode-shaped paged calls: each workgroup
        # walks its whole KV range as a serial chain of dependent loads, so
        # shortening the chain matters however idle the GPU is. No caller
        # sizes the count for paged, so 1 is treated like the 0 sentinel.
        num_kv_splits = _paged_auto_num_kv_splits(
            B,
            H,
            Sq,
            _gqa_qlen if _gqa_packed else Sq,
            skv,
            D_run,
            page_size,
            window_left,
            _gqa_packed,
            _dense_light_cu(q.device),
        )
        _auto_sized = True
    elif requested_num_kv_splits <= 1:
        num_kv_splits = 1
    else:
        num_kv_splits = requested_num_kv_splits
    splitk = num_kv_splits > 1
    if splitk and not _splitk_dtype_ok:
        raise ValueError(
            f"flydsl_flash_attn_func: paged split-K requires D=64/128, dtype "
            f"bf16/f16; got D={D}, dtype={dtype_str}"
        )
    # The dualwave split-K route needs enough Q rows to amortise its pipeline; the
    # generic light route has no such floor, and short-q is exactly where splitting
    # pays. Keep the old requirement only for the route it was written for.
    if splitk and not _paged_light_ok and Sq < 384:
        raise ValueError(
            f"flydsl_flash_attn_func: dualwave paged split-K requires seq_len>=384; "
            f"got seq_len={Sq}"
        )
    if splitk:
        _validate_splitk_workspace_descriptor(H, Sq, D_run, int(num_kv_splits))

    # Per-batch KV lengths differ in general → bottom-right cross-length masking. Varlen
    # paged always uses cross masking (per-batch seqlen_q/seqlen_kv come from cu_seqlens).
    max_kv_pages = (skv + page_size - 1) // page_size

    # The dualwave kernel maps the MFMA M-axis to query rows. The decode kernel
    # instead packs (query-token, head) pairs onto M for short query blocks.
    _hp_groups = 0
    if (
        not _DISABLE_PAGED_DECODE_HP
        and not _paged_window
        and not _masked_hd
        and (causal or Sq == 1)
        # Generic GQA packing removes the same fan-out and takes precedence.
        and not _gqa_packed
        and not varlen
        and kv_seqstart is None
        and not return_lse
        and not vectorized
        and D in (64, 128)
        and dtype_str in ("bf16", "f16")
        and page_size % _PAGED_DECODE_TILE_N == 0
        and _arch.startswith("gfx950")
    ):
        _hp_groups = _hp_decode_ok(H, num_kv_heads, Sq, D, B, max_kv_pages, q.device)
    if _hp_groups:
        from .decode.pa_decode_dense import pa_decode_paged_launch

        with torch.cuda.device(q.device.index):
            with torch.cuda.stream(launch_stream):
                hp_out = pa_decode_paged_launch(
                    q.view(B, Sq, 1, H, D),
                    k,
                    v,
                    block_table,
                    seqlen_k,
                    float(sm_scale) if sm_scale is not None else float(D**-0.5),
                    page_size=page_size,
                    max_seqlen_kv=skv,
                    # A count sized for the light kernel's chain means nothing to
                    # this kernel (and need not be a compiled power of two).
                    split_k=(
                        0
                        if _auto_sized or requested_num_kv_splits == 0
                        else int(num_kv_splits)
                    ),
                    output_dtype=q.dtype,
                    query_groups=_hp_groups,
                ).view(B, Sq, H, D)
                if out is not None:
                    out.copy_(hp_out)
                    return out
                return hp_out

    # Split-K (paged, dense only): split the KV dimension across
    # grid_z = B*num_kv_splits workgroups + a combine pass. Fills the GPU for
    # low-occupancy shapes (small B / few heads), where single-split paged
    # underutilizes the device. The count was resolved with the route above.
    max_pages_per_split = (max_kv_pages + int(num_kv_splits) - 1) // int(num_kv_splits)
    if max_pages_per_split > _PAGED_BT_LDS_SIZE:
        max_supported_kv = _PAGED_BT_LDS_SIZE * int(num_kv_splits) * page_size
        raise NotImplementedError(
            f"flydsl_flash_attn_func: paged KV length {skv} exceeds block-table LDS window "
            f"({_PAGED_BT_LDS_SIZE} pages/split, max_kv_len={max_supported_kv} for "
            f"num_kv_splits={num_kv_splits}, page_size={page_size})"
        )
    # Only the light kernel reads KV_LENS. Dualwave has no dense per-request KV
    # length, but its varlen variant reads cu_seqlens_kv, and dense Q/O are the
    # same bytes as uniform packed varlen, so run that instead.
    _dw_kv_lens = _dense_kv_lens and not _paged_light_ok
    if varlen:
        cross = bool(cross_seqlen) if cross_seqlen is not None else True
    else:
        cross = _dw_kv_lens or skv != Sq
    with torch.cuda.device(q.device.index):
        with torch.cuda.stream(launch_stream):
            # Flatten so the kernel's row-major index addresses the table correctly.
            block_table_i32 = block_table.to(torch.int32).contiguous().reshape(-1)
            # The light kernel clamps int32 lengths to [0, skv] itself and writes
            # an empty request as 0 (LSE -inf). Dualwave and gappy lengths are
            # bounded here and their empty requests masked after the launch.
            _host_bounds = not _paged_light_ok or kv_seqstart is not None
            bounded_seqlen_k = seqlen_k
            if seqlen_k is not None and (_host_bounds or not _is_i32_vector(seqlen_k)):
                bounded_seqlen_k, _ = _normalize_paged_lengths(
                    seqlen_k,
                    launch_kv_bound=skv,
                    block_table_capacity=block_table_capacity,
                )
            bounded_cu_seqlens_kv = cu_seqlens_kv
            bounded_kv_seqstart = None
            if cu_seqlens_kv is not None and (
                _host_bounds or not _is_i32_vector(cu_seqlens_kv)
            ):
                bounded_cu_seqlens_kv, bounded_kv_seqstart = (
                    _normalize_paged_cumulative_lengths(
                        cu_seqlens_kv,
                        launch_kv_bound=skv,
                        block_table_capacity=block_table_capacity,
                        kv_seqstart=kv_seqstart,
                    )
                )
            if return_lse and not _paged_light_ok:
                raise NotImplementedError(
                    "flydsl_flash_attn_func: return_lse for native paged KV is only "
                    "supported on the generic (short-seq / gfx942) path"
                )
            _paged_bm = _paged_block_m(_gqa_packed, Sq)
            if _paged_narrow_tile(
                Sq,
                window_left,
                _gqa_qlen if _gqa_packed else Sq,
                skv,
                num_kv_splits,
                D_run,
                B * H,
            ):
                _paged_bm = 32
            fused_combine = _paged_light_ok and _paged_fused_combine(
                splitk,
                return_lse,
                B * H * (-(-Sq // _paged_bm)),
                int(num_kv_splits),
                D_run,
                min(Sq, _paged_bm),
            )
            # A masked build reads K/V through the cooperative loads only.
            _k_prefetch = not _masked_hd and _paged_k_prefetch(
                D_run, window_left, _gqa_qlen if _gqa_packed else Sq, skv, num_kv_splits
            )
            if _paged_light_ok:
                exe = _build_paged_light(
                    num_heads=H,
                    num_kv_heads=num_kv_heads,
                    head_dim=D_run,
                    causal=causal,
                    dtype_str=dtype_str,
                    cross_seqlen=cross,
                    varlen=varlen,
                    kv_cache_layout=kv_cache_layout,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    gappy_kv=_gappy,
                    return_lse=return_lse,
                    sm_scale=sm_scale,
                    q_pack_qlen=_gqa_qlen if (_gqa_packed and causal) else 0,
                    q_pack_group=_gqa_group if _gqa_stride_packed else 0,
                    # Resolved here because the builder cannot see batch. Passed as
                    # a bool, not B, so the lru_cache forks once rather than per
                    # distinct batch; 0 disables, keeping the head-dim bound alone.
                    high_concurrency=(
                        _WAVE_SPLITK_MIN_BATCH > 0 and B >= _WAVE_SPLITK_MIN_BATCH
                    ),
                    kv_lens=_dense_kv_lens,
                    block_m=_paged_bm,
                    kv_dma=not _masked_hd
                    and _paged_kv_dma(
                        _paged_bm, D_run, window_left, skv, num_kv_splits
                    ),
                    k_prefetch=_k_prefetch,
                    window_left=window_left,
                    num_kv_splits=int(num_kv_splits),
                    fused_combine=fused_combine,
                    pid_regs=_paged_pid_regs(max_pages_per_split) and not _k_prefetch,
                    head_dim_qk=D,
                    head_dim_v=D,
                )
            else:
                exe = _build_paged(
                    num_heads=H,
                    num_kv_heads=num_kv_heads,
                    head_dim=D,
                    causal=causal,
                    dtype_str=dtype_str,
                    cross_seqlen=cross,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    lazy_rescale=dualwave_swp_lazy_rescale,
                    setprio=dualwave_swp_setprio,
                    enable_stagger=dualwave_swp_enable_stagger,
                    num_kv_splits=int(num_kv_splits),
                    varlen=varlen or _dw_kv_lens,
                    kv_cache_layout=kv_cache_layout,
                )
            if out is None:
                out = torch.empty_like(q)
            q_flat = q.contiguous()
            k_flat = k.contiguous()
            v_flat = v.contiguous()
            o_flat = out.contiguous()
            if _dw_kv_lens:
                q_flat = q_flat.view(B * Sq, H, D)
                o_flat = o_flat.view(B * Sq, H, D)
            kwargs = {
                "block_table": block_table_i32,
                "block_table_stride": block_table_stride,
                "stream": launch_stream,
            }
            output_cu_seqlens_q = None
            if varlen:
                original_cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
                kernel_cu_seqlens_q = (
                    _uniform_cu_seqlens(B, Sq, str(q.device))
                    if _gqa_rebuild_cu_q
                    else original_cu_seqlens_q
                )
                kwargs["cu_seqlens_q"] = kernel_cu_seqlens_q
                output_cu_seqlens_q = (
                    original_cu_seqlens_q if _gqa_stride_packed else kernel_cu_seqlens_q
                )
                kwargs["cu_seqlens_kv"] = bounded_cu_seqlens_kv
            if kv_seqstart is not None:
                kwargs["kv_seqstart"] = bounded_kv_seqstart
            if cross or _paged_light_ok:
                # The light kernel also bounds every request's length by it.
                kwargs["seq_len_kv"] = skv
            lse = (
                torch.empty((B, H, Sq), dtype=torch.float32, device=q.device)
                if return_lse
                else None
            )
            if return_lse:
                kwargs["lse"] = lse
            if splitk:
                ws_elems = dualwave_splitk_workspace_elems(
                    B, H, Sq, int(num_kv_splits), head_dim=D_run
                )
                if fused_combine:
                    workspace = arrival_workspace(ws_elems, q.device, launch_stream)
                else:
                    workspace = torch.empty(
                        ws_elems, dtype=torch.float32, device=q.device
                    )
                kwargs["workspace"] = workspace
            if _dw_kv_lens:
                kwargs["cu_seqlens_kv"] = _paged_kv_cumsum(bounded_seqlen_k)
                kwargs["cu_seqlens_q"] = _uniform_cu_seqlens(B, Sq, str(q.device))
            elif _dense_kv_lens:
                kwargs["cu_seqlens_kv"] = bounded_seqlen_k
            # Stride-packed builds compile the row count (q_len * group) in.
            assert not (_gqa_stride_packed and causal) or Sq == _gqa_qlen * _gqa_group
            exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)
            if o_flat.data_ptr() != out.data_ptr():
                out.copy_(o_flat.view_as(out))
            if _host_bounds and not varlen and bounded_seqlen_k is not None:
                empty_kv = bounded_seqlen_k.eq(0)
                out.masked_fill_(empty_kv.view(B, 1, 1, 1), 0)
                if lse is not None:
                    lse.masked_fill_(empty_kv.view(B, 1, 1), -torch.inf)
            elif _host_bounds and varlen and bounded_cu_seqlens_kv is not None:
                empty_kv = bounded_cu_seqlens_kv[1:].eq(bounded_cu_seqlens_kv[:-1])
                q_lengths = output_cu_seqlens_q[1:] - output_cu_seqlens_q[:-1]
                empty_q_rows = torch.repeat_interleave(
                    empty_kv,
                    q_lengths,
                    output_size=q_flat.shape[0],
                )
                out.masked_fill_(empty_q_rows.view(-1, 1, 1), 0)
                if lse is not None:
                    lse.masked_fill_(empty_kv.view(B, 1, 1), -torch.inf)
            if _gqa_packed and not _gqa_stride_packed:
                # Invert the pack permutation: (group, token, kv-head) ->
                # (token, head). Keep materialization and copy-back ordered after
                # the kernel when the caller supplies a noncurrent stream.
                unpacked = out.view(_gqa_B, _gqa_group, _gqa_qlen, H, D).permute(
                    0, 2, 3, 1, 4
                )
                out = unpacked.reshape(
                    (_gqa_B * _gqa_qlen, _gqa_heads, D)
                    if _gqa_varlen
                    else (_gqa_B, _gqa_qlen, _gqa_heads, D)
                )
                if _gqa_user_out is not None:
                    _gqa_user_out.view(out.shape).copy_(out)
                    out = _gqa_user_out

    return (out, lse) if return_lse else out


@functools.lru_cache(maxsize=256)
def _build_paged_light(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    cross_seqlen: bool,
    varlen: bool,
    kv_cache_layout: str,
    waves_per_eu: int,
    daz: bool,
    num_kv_splits: int,
    gappy_kv: bool = False,
    return_lse: bool = False,
    sm_scale: Optional[float] = None,
    q_pack_qlen: int = 0,
    q_pack_group: int = 0,
    kv_lens: bool = False,
    block_m: int = 0,
    window_left: int = -1,
    kv_dma: bool = False,
    k_prefetch: bool = False,
    high_concurrency: bool = False,
    fused_combine: bool = False,
    pid_regs: bool = False,
    head_dim_qk: Optional[int] = None,
    head_dim_v: Optional[int] = None,
):
    """Build a lightweight paged-varlen launcher for short attention.

    ``num_kv_splits > 1`` partitions the KV loop across grid.y. Decode-shaped
    paged calls (q_len 1-16) otherwise walk the whole context in one workgroup,
    which is latency-bound end to end; splitting shortens that serial chain.
    """
    from .flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        num_kv_heads=num_kv_heads,
        cross_seqlen=cross_seqlen,
        varlen=varlen,
        paged=True,
        kv_cache_layout=kv_cache_layout,
        block_m=block_m or _PAGED_LIGHT_BLOCK_M,
        flat_work_group_size=(block_m or _PAGED_LIGHT_BLOCK_M) * 2,
        path_tag="N32",
        # N32 is the only path carrying the window mask, and it is already what
        # this builder selects -- so the window costs no path change here.
        window_left=window_left,
        waves_per_eu=waves_per_eu,
        daz=daz,
        gappy_kv=gappy_kv,
        return_lse=return_lse,
        sm_scale=sm_scale,
        q_pack_qlen=q_pack_qlen,
        q_pack_group=q_pack_group,
        high_concurrency=high_concurrency,
        kv_lens=kv_lens,
        num_kv_splits=num_kv_splits,
        paged_kv_dma=kv_dma,
        paged_k_prefetch=k_prefetch,
        fused_combine=fused_combine,
        paged_pid_regs=pid_regs,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
    )


# ── public API ─────────────────────────────────────────────────────────────


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    num_kv_heads: Optional[int] = None,
    # Varlen (packed cu_seqlens): pass both to enable the varlen path.
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    # Gappy KV: per-seq absolute KV start [B] into an un-repacked K/V store. With
    # cu_seqlens_kv giving per-seq lengths, the kernel gathers each seq in-place.
    kv_seqstart: Optional[torch.Tensor] = None,
    # Max per-batch Q seqlen (varlen only). Required for varlen to size grid_y
    # without synchronizing on cu_seqlens_q.
    max_seqlen_q: Optional[int] = None,
    # Max per-batch KV seqlen (varlen cross-attn only). Used to size the KV grid
    # when seqlen_q != seqlen_kv per batch.
    max_seqlen_kv: Optional[int] = None,
    # Whether per-batch Sq and Skv can differ. Dense mode infers this from shapes;
    # varlen mode requires it explicitly to choose the correct build variant.
    cross_seqlen: Optional[bool] = None,
    # Paged KV cache ABI: vLLM-style block_table + seqlen_k.
    block_table: Optional[torch.Tensor] = None,
    seqlen_k: Optional[torch.Tensor] = None,
    kv_cache_layout: str = "linear",
    # Split-K (gfx950 only, D=64/128, bf16/f16). The dense path additionally
    # requires seq_len >= 384. 0 and 1 = auto on the paged light route and a
    # single pass elsewhere (see below), >1 = force that many splits.
    num_kv_splits: int = 0,
    # fp8 dense ABI: per-tensor descales for pre-quantized e4m3fn Q/K/V.
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    # Output tensor; allocated if None.
    out: Optional[torch.Tensor] = None,
    # Kernel build options.
    waves_per_eu: int = 2,
    daz: bool = True,
    dualwave_swp_lazy_rescale: bool = True,
    dualwave_swp_setprio: bool = True,
    dualwave_swp_enable_stagger: bool = True,
    # Debug: pass a pre-allocated float32[2] tensor to enable the lazy-rescale
    # branch counter (dualwave_swp_debug_lazy_counts=True). Only for dense mode.
    debug_counts: Optional[torch.Tensor] = None,
    # When True, also return per-row log-sum-exp (natural log, sm_scale folded in,
    # shape [B, H, Sq]; varlen: [B, H, max_seqlen_q]). Not supported for fp8/paged KV.
    return_lse: bool = False,
    # Sliding-window, flash-attention convention: keep KV in
    # [q_row - window_left, q_row] (window_left + 1 keys); -1 disables. Non-causal
    # (dense/varlen) keeps only the left bound: KV >= q_row - window_left, i.e.
    # flash-attention's window_size=(window_left, -1). Rows are bottom-right
    # aligned (q_row counts from seqlen_kv - seqlen_q).
    window_left: int = -1,
    # Softmax scale (None -> 1/sqrt(head_dim)); non-default forces the generic path.
    sm_scale: Optional[float] = None,
    # Additive bias broadcastable to [B, H, Sq, Skv], added to QK*scale.
    bias: Optional[torch.Tensor] = None,
    # Top-left causal (kv_col <= q_row) vs default bottom-right; cross-length only.
    causal_top_left: bool = False,
    # Dropout mask broadcastable to [B, H, Sq, Skv]; 0 or 1/(1-p), applied to post-
    # softmax P.
    dropout_mask: Optional[torch.Tensor] = None,
    # CUDA/HIP stream; defaults to the current stream for q.device.
    stream: Optional[torch.cuda.Stream] = None,
):
    """Run FlyDSL Flash Attention (gfx950 DUALWAVE_SWP / gfx942 generic fallback).

    Args:
        q: Query tensor. Dense: ``[B, Sq, H, D]`` (BSHD).
           Varlen: ``[total_q, H, D]`` (packed, cu_seqlens_q required).
        k: Key tensor. Dense: ``[B, Skv, Hkv, D]``.
           Varlen: ``[total_kv, Hkv, D]``.
        v: Value tensor, same shape as k except possibly the head dim (non-paged
           f16/bf16; both head dims even and <= 256).
           Paged KV cache (future ABI): physical K/V cache tensors. Supported
           ``kv_cache_layout`` values:
           - ``linear``: 4D paged K/V, ``[NumBlocks, PageSize, NumKVHeads, HeadDim]``.
           - ``linear3d``: page_size=1 special case,
             ``[NumBlocks, NumKVHeads, HeadDim]``.
           - ``vectorized``: aiter-style 5D K/V, where
             ``K = [NumBlocks, NumKVHeads, HeadDim / kVectorSize, PageSize, kVectorSize]``
             and
             ``V = [NumBlocks, NumKVHeads, PageSize / kVectorSize, HeadDim, kVectorSize]``.
             Here ``kVectorSize = 16 / element_size`` (bf16/fp16: 8, fp8: 16);
             page_size and head_dim must be divisible by it.
        causal: Bottom-right aligned causal mask when True.
        num_kv_heads: KV head count for GQA/MQA; defaults to q num_heads (MHA).
        cu_seqlens_q: Int32 ``[B+1]`` cumulative Q token counts (varlen).
        cu_seqlens_kv: Int32 ``[B+1]`` cumulative KV token counts (varlen).
        max_seqlen_q: Maximum per-batch Q seqlen (varlen). Required in varlen mode.
        max_seqlen_kv: Maximum per-batch KV seqlen (varlen cross-attn). Required when
            seqlen_q != seqlen_kv per batch.
        cross_seqlen: Whether seqlen_q and seqlen_kv differ. Required in varlen mode;
            dense mode infers it from ``q.shape[1] != k.shape[1]``.
        block_table / seqlen_k: vLLM-style 2D block table metadata.
        num_kv_splits: Split-K factor. ``0`` (default) and ``1`` both mean auto
            on the paged light route (D=64/128, bf16/f16, uniform-q, not
            gappy): the split count is sized from KV-chain length by
            ``_paged_auto_num_kv_splits``. Everywhere else they run a single pass,
            except the dense non-paged path, which applies its own occupancy
            heuristic. ``>1`` forces that many splits (gfx950 only, D=64/128,
            bf16/f16; the dense non-paged path additionally requires seq>=384).
            Dense calls with a window, custom sm_scale, bias, dropout or cross
            length run the forced count on the generic kernel's split-K.
            Negative values are invalid.
        q_descale / k_descale / v_descale: fp32 shape-[1] descales required
            for dense fp8 e4m3fn inputs.
        out: Optional pre-allocated output tensor. For fp8, output is bf16;
            otherwise it has the same dtype as q.
        waves_per_eu: Kernel occupancy hint.
        daz: Enable denormals-are-zero.
        dualwave_swp_lazy_rescale: Enable lazy online softmax rescale.
        dualwave_swp_setprio: Enable s_setprio scheduling hints.
        dualwave_swp_enable_stagger: Enable wave-group phase stagger.
        debug_counts: Float32[2] tensor; when given, counts lazy-rescale branches
            (debug_counts[0] = all-below-true, debug_counts[1] = all-below-false).
        return_lse: When True, also compute and return per-row log-sum-exp
            (natural log, sm_scale folded in). Not supported for fp8 or paged KV.
        stream: CUDA/HIP stream to launch on.

    Returns:
        Output tensor shaped like q with v's head dim (dtype bf16 for fp8 inputs,
        otherwise the same dtype as q). When ``return_lse=True``, returns a
        ``(out, lse)`` tuple instead, where ``lse`` is float32 ``[B, H, Sq]``
        (varlen: ``[B, H, max_seqlen_q]``; padded rows undefined).

    Head dims 64, 128 and 256 run natively. Other even head dims up to 256 run
    on the generic kernel at the next of those, which masks the missing columns
    in-kernel: nothing is copied, and paged KV caches are read in place.
    """
    # ── validation ──────────────────────────────────────────────────────────
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func: q/k/v must be CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            f"flydsl_flash_attn_func: q/k/v must share device; got {q.device}/{k.device}/{v.device}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            f"flydsl_flash_attn_func: q/k/v must share dtype; got {q.dtype}/{k.dtype}/{v.dtype}"
        )

    dtype_str = _dtype_str(q)
    paged_kv = any(x is not None for x in (block_table, seqlen_k))
    if dtype_str == "fp8" and paged_kv:
        raise NotImplementedError(
            "flydsl_flash_attn_func: fp8 flash_attn does not support paged KV"
        )
    if return_lse and dtype_str == "fp8":
        raise NotImplementedError(
            "flydsl_flash_attn_func: return_lse is not supported for fp8"
        )
    # LSE support for paged KV depends on which kernel the shape routes to: the
    # generic light path (paged-gappy, or short-q / gfx942 native paged) produces
    # LSE; the dualwave native-paged path does not. The precise check lives in
    # _flydsl_flash_attn_paged where the light-vs-dualwave decision is made.
    if window_left >= 0:
        # Window lives in apply_kv_mask, which is the generic kernel's N32 mask
        # path. Dense/varlen and paged both reach it -- paged light already
        # builds with path_tag="N32" -- so the window is expressible there. fp8
        # has no mask path at all.
        #
        # The *native* paged kernels (dualwave, pa_decode_gfx950) do their own
        # masking and have no window, so they must not take a windowed shape:
        # they would silently drop the window and return a full-context answer.
        # `_paged_window` in the paged helper forces the light route and declines
        # fast paths for exactly that reason.
        if dtype_str == "fp8":
            raise NotImplementedError(
                "flydsl_flash_attn_func: sliding-window (window_left>=0) is not "
                "supported for fp8"
            )
        if paged_kv and not causal:
            raise NotImplementedError(
                "flydsl_flash_attn_func: sliding-window paged KV requires causal=True"
            )
        if paged_kv and kv_seqstart is not None:
            # Gappy paged builds a different KV index space; the window bound is
            # expressed in flat KV columns and would not line up.
            raise NotImplementedError(
                "flydsl_flash_attn_func: sliding-window (window_left>=0) is not "
                "supported for gappy paged KV (kv_seqstart)"
            )
        # The kernels keep KV in (q_row - w, q_row], an exclusive left bound.
        window_left += 1
    if sm_scale is not None:
        # fp8 and native paged hardcode 1/sqrt(head_dim); gappy paged uses the
        # generic kernel and folds sm_scale.
        _gappy_paged = paged_kv and kv_seqstart is not None
        if dtype_str == "fp8" or (paged_kv and not _gappy_paged):
            raise NotImplementedError(
                "flydsl_flash_attn_func: custom sm_scale is only supported on the "
                "dense/varlen f16/bf16 paths (fp8/paged hardcode 1/sqrt(head_dim))"
            )
    if bias is not None:
        _varlen_req = cu_seqlens_q is not None
        if dtype_str == "fp8" or paged_kv or _varlen_req:
            raise NotImplementedError(
                "flydsl_flash_attn_func: attention bias is only supported on the "
                "dense f16/bf16 path (not fp8/paged/varlen)"
            )
    if dropout_mask is not None:
        _varlen_req = cu_seqlens_q is not None
        if dtype_str == "fp8" or paged_kv or _varlen_req:
            raise NotImplementedError(
                "flydsl_flash_attn_func: dropout is only supported on the "
                "dense f16/bf16 path (not fp8/paged/varlen)"
            )
    if paged_kv:
        return _flydsl_flash_attn_paged(
            q,
            k,
            v,
            causal=causal,
            num_kv_heads=num_kv_heads,
            block_table=block_table,
            seqlen_k=seqlen_k,
            max_seqlen_kv=max_seqlen_kv,
            kv_cache_layout=kv_cache_layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            kv_seqstart=kv_seqstart,
            max_seqlen_q=max_seqlen_q,
            cross_seqlen=cross_seqlen,
            num_kv_splits=num_kv_splits,
            return_lse=return_lse,
            sm_scale=sm_scale,
            out=out,
            waves_per_eu=waves_per_eu,
            daz=daz,
            dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
            dualwave_swp_setprio=dualwave_swp_setprio,
            dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
            stream=stream,
            window_left=window_left,
        )

    varlen = cu_seqlens_q is not None

    # `0` (auto) only selects a split count on the dense paged path, which has
    # already returned above. Everything from here on is non-paged, where the
    # existing per-shape `generic_splitk` selection is the auto mechanism, so
    # collapse the sentinel to the historical default before anything reads it.
    if num_kv_splits == 0:
        num_kv_splits = 1

    if dtype_str == "fp8":
        # _build_dense_fp8 compiles a D=128 kernel whatever the input: D=64 reads
        # and writes past each row, D=256 drops half of it.
        if q.shape[-1] != 128:
            raise NotImplementedError(
                "flydsl_flash_attn_func: fp8 flash_attn supports head_dim=128 only, "
                f"got {q.shape[-1]}"
            )
        if varlen:
            raise NotImplementedError(
                "flydsl_flash_attn_func: fp8 flash_attn does not support varlen"
            )
        if num_kv_splits > 1:
            raise NotImplementedError(
                "flydsl_flash_attn_func: fp8 flash_attn does not support split-K"
            )
        if debug_counts is not None:
            raise NotImplementedError(
                "flydsl_flash_attn_func: fp8 flash_attn does not support debug_counts"
            )
        if any(x is None for x in (q_descale, k_descale, v_descale)):
            raise ValueError(
                "flydsl_flash_attn_func: fp8 requires q_descale, k_descale, and v_descale"
            )
        for name, scale in (
            ("q_descale", q_descale),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
        ):
            if not scale.is_cuda:
                raise ValueError(
                    f"flydsl_flash_attn_func: {name} must be a CUDA tensor"
                )
            if scale.device != q.device:
                raise ValueError(
                    f"flydsl_flash_attn_func: {name} must be on {q.device}, got {scale.device}"
                )
            if scale.dtype != torch.float32 or scale.numel() != 1:
                raise ValueError(
                    f"flydsl_flash_attn_func: {name} must be a shape-[1] float32 tensor"
                )

    if varlen and cu_seqlens_kv is None:
        raise ValueError(
            "flydsl_flash_attn_func: cu_seqlens_kv required when cu_seqlens_q is given"
        )
    if not varlen and cu_seqlens_kv is not None:
        raise ValueError(
            "flydsl_flash_attn_func: cu_seqlens_q required when cu_seqlens_kv is given"
        )
    if varlen and num_kv_splits > 1:
        raise ValueError(
            "flydsl_flash_attn_func: varlen + split-K (num_kv_splits>1) is not supported"
        )

    # ── shape inference ─────────────────────────────────────────────────────
    if varlen:
        if q.dim() != 3:
            raise ValueError(
                f"flydsl_flash_attn_func: varlen q must be 3D [total,H,D], got {q.dim()}D"
            )
        _total_q, H, D = q.shape
        Hkv = k.shape[1]
        B = cu_seqlens_q.numel() - 1
        if max_seqlen_q is None:
            raise ValueError(
                "flydsl_flash_attn_func: max_seqlen_q is required in varlen mode"
            )
        if cross_seqlen is None:
            raise ValueError(
                "flydsl_flash_attn_func: cross_seqlen is required in varlen mode"
            )
        Sq = int(max_seqlen_q)
        cross = bool(cross_seqlen)
        if cross and max_seqlen_kv is None:
            raise ValueError(
                "flydsl_flash_attn_func: max_seqlen_kv is required when varlen cross_seqlen=True"
            )
    else:
        if q.dim() != 4:
            raise ValueError(
                f"flydsl_flash_attn_func: dense q must be 4D [B,Sq,H,D], got {q.dim()}D"
            )
        B, Sq, H, D = q.shape
        Skv = k.shape[1]
        Hkv = k.shape[2]
        cross = Sq != Skv if cross_seqlen is None else bool(cross_seqlen)

    if num_kv_heads is None:
        num_kv_heads = Hkv
    if H % num_kv_heads != 0:
        raise ValueError(
            f"flydsl_flash_attn_func: num_heads ({H}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    # Stored head dims: Q/K rows are Dqk wide, V/O rows Dv wide. The kernels
    # build D = 64/128/256; for anything else (or Dv != Dqk) the generic kernel
    # computes at the next of those and masks the missing columns in-kernel.
    Dqk = D
    Dv = int(v.shape[-1])
    if int(k.shape[-1]) != Dqk:
        raise ValueError(
            f"flydsl_flash_attn_func: k head_dim ({k.shape[-1]}) must match q ({Dqk})"
        )
    masked_hd = Dv != Dqk or D not in _GENERIC_HEAD_DIMS
    if masked_hd:
        D = _run_head_dim(max(Dqk, Dv))
        if not D or Dqk % 2 or Dv % 2:
            raise ValueError(
                f"flydsl_flash_attn_func: head dims (qk={Dqk}, v={Dv}) must be "
                "even and <= 256"
            )
        if dtype_str == "fp8" or num_kv_splits > 1 or debug_counts is not None:
            raise NotImplementedError(
                f"flydsl_flash_attn_func: head dims (qk={Dqk}, v={Dv}) need the "
                "generic f16/bf16 kernel (no fp8, explicit split-K or debug_counts)"
            )

    splitk = num_kv_splits > 1
    # generic_splitk is chosen internally per-shape (below); it uses the generic
    # BLOCK_M=64 kernel + dense combine, distinct from the dualwave `splitk` above.
    generic_splitk = False
    # An explicit split count the dualwave split-K cannot honour (see below) runs
    # on the generic split-K instead.
    _forced_generic_splits = 0

    # ── split-K eligibility guard (SKIP analogous to run_splitk_config) ────
    if splitk:
        if D not in (64, 128) or dtype_str not in ("bf16", "f16") or Sq < 384:
            raise ValueError(
                f"flydsl_flash_attn_func: split-K requires D=64/128, dtype bf16/f16, seq_len>=384; "
                f"got D={D}, dtype={dtype_str}, seq_len={Sq}"
            )
        # The dualwave split-K launcher is built and launched without any of
        # these, so they would be silently dropped (cross length: wrong result).
        _dropped = [
            name
            for name, on in (
                ("window_left", window_left >= 0),
                ("sm_scale", sm_scale is not None),
                ("bias", bias is not None),
                ("dropout_mask", dropout_mask is not None),
                ("seq_len_q != seq_len_kv", cross),
            )
            if on
        ]
        if _dropped:
            # The generic kernel's split-K applies every one of these (its combine
            # is exact for them), so run the requested count there.
            _forced_generic_splits = int(num_kv_splits)
            _validate_splitk_workspace_descriptor(H, Sq, D, _forced_generic_splits)
            splitk = False
        else:
            ws_elems = dualwave_splitk_workspace_elems(
                B, H, Sq, int(num_kv_splits), head_dim=D
            )
            _validate_splitk_workspace_descriptor(H, Sq, D, int(num_kv_splits))

    # ── build (cached) ──────────────────────────────────────────────────────
    debug_lazy = debug_counts is not None

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )

        if splitk:
            exe = _build_splitk(
                num_heads=H,
                num_kv_heads=num_kv_heads,
                head_dim=D,
                causal=causal,
                dtype_str=dtype_str,
                num_kv_splits=int(num_kv_splits),
                waves_per_eu=waves_per_eu,
                daz=daz,
                lazy_rescale=dualwave_swp_lazy_rescale,
                setprio=dualwave_swp_setprio,
                enable_stagger=dualwave_swp_enable_stagger,
                return_lse=return_lse,
            )
        elif varlen:
            # Generic light for short/cross/D256 varlen; long self-attn uses dualwave.
            # (The dualwave cross path NaNs for >=5 KV tiles; D=256 is generic-only.)
            _arch = _gpu_arch(q.device)
            _prefer_light = (
                (not debug_lazy)
                and D in (64, 128, 256)
                and dtype_str in ("bf16", "f16")
                and (
                    cross
                    or D == 256
                    or window_left >= 0
                    # dualwave varlen hardcodes 1/sqrt(head_dim), so custom scale
                    # must use light.
                    or sm_scale is not None
                    # f16 overflows the dualwave softmax; use light.
                    or dtype_str == "f16"
                    or not _arch.startswith("gfx950")
                    or Sq <= _VARLEN_LIGHT_MAX_SEQ
                    # Gappy KV is only implemented on the generic (light) path.
                    or kv_seqstart is not None
                    # Only the light kernel masks stored head dims.
                    or masked_hd
                )
            )
            if _prefer_light:
                exe = _build_varlen_light(
                    num_heads=H,
                    num_kv_heads=num_kv_heads,
                    head_dim=D,
                    causal=causal,
                    dtype_str=dtype_str,
                    cross_seqlen=cross,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    lazy_rescale=dualwave_swp_lazy_rescale,
                    setprio=dualwave_swp_setprio,
                    debug_lazy_counts=debug_lazy,
                    enable_stagger=dualwave_swp_enable_stagger,
                    return_lse=return_lse,
                    causal_top_left=causal_top_left,
                    window_left=window_left,
                    sm_scale=sm_scale,
                    gappy_kv=kv_seqstart is not None,
                    head_dim_qk=Dqk,
                    head_dim_v=Dv,
                )
            else:
                exe = _build_varlen(
                    num_heads=H,
                    num_kv_heads=num_kv_heads,
                    head_dim=D,
                    causal=causal,
                    dtype_str=dtype_str,
                    cross_seqlen=cross,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    lazy_rescale=dualwave_swp_lazy_rescale,
                    setprio=dualwave_swp_setprio,
                    debug_lazy_counts=debug_lazy,
                    enable_stagger=dualwave_swp_enable_stagger,
                    return_lse=return_lse,
                )
        else:
            _arch = _gpu_arch(q.device)
            if dtype_str == "fp8":
                if not _arch.startswith("gfx950"):
                    raise ValueError(
                        f"flydsl_flash_attn_func: fp8 requires gfx950, got '{_arch or 'unknown'}'"
                    )
                exe = _build_dense_fp8(
                    num_heads=H,
                    num_kv_heads=num_kv_heads,
                    causal=causal,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    lazy_rescale=dualwave_swp_lazy_rescale,
                    setprio=dualwave_swp_setprio,
                    enable_stagger=dualwave_swp_enable_stagger,
                )
            else:
                # f16 excluded from dualwave: its narrow exponent overflows the
                # dualwave softmax at large logits -> NaN; bf16 is safe.
                can_dualwave = (
                    D in (64, 128)
                    and dtype_str == "bf16"
                    and _arch.startswith("gfx950")
                )
                if debug_lazy and not can_dualwave:
                    raise NotImplementedError(
                        "flydsl_flash_attn_func: debug_counts requires the gfx950 DUALWAVE_SWP path"
                    )
                # Window / custom scale / bias / dropout / cross are generic-only;
                # never route them to dualwave (hardcodes 1/sqrt(D), NaNs on cross).
                _windowed = window_left >= 0
                _custom_scale = sm_scale is not None
                _has_bias = bias is not None
                _has_dropout = dropout_mask is not None
                # Top-left only differs from the default bottom-right alignment
                # when the lengths differ; keep self-attention on one build.
                _top_left = causal and cross and causal_top_left
                if (
                    not masked_hd
                    and not _windowed
                    and not _custom_scale
                    and not _has_bias
                    and not _has_dropout
                    and not cross
                    and (
                        debug_lazy
                        or (can_dualwave and _dense_routes_to_dualwave(B, Sq))
                    )
                ):
                    exe = _build_dense_dualwave(
                        num_heads=H,
                        num_kv_heads=num_kv_heads,
                        head_dim=D,
                        causal=causal,
                        dtype_str=dtype_str,
                        cross_seqlen=cross,
                        waves_per_eu=waves_per_eu,
                        daz=daz,
                        lazy_rescale=dualwave_swp_lazy_rescale,
                        setprio=dualwave_swp_setprio,
                        debug_lazy_counts=debug_lazy,
                        enable_stagger=dualwave_swp_enable_stagger,
                        return_lse=return_lse,
                    )
                else:
                    # Generic split-K for short-q dense self-attention that underfills
                    # the GPU: BLOCK_M=64, gated by CK's occupancy heuristic. Bias,
                    # window, scale and dropout compose: dropout is an explicit keep
                    # mask indexed by absolute KV column, applied after l/LSE (no
                    # philox offsets to split, unlike CK), so the combine is exact.
                    _gen_splits = _forced_generic_splits or 1
                    _split_min_kv = 0
                    if _has_dropout and D >= 128:
                        _split_min_kv = _DROPOUT_SPLITK_MIN_KV
                    elif _has_dropout or cross:
                        _split_min_kv = _CROSS_SPLITK_MIN_KV
                    if (
                        not _forced_generic_splits
                        and Skv >= _split_min_kv
                        and D in (64, 128, 256)
                        and dtype_str in ("bf16", "f16")
                    ):
                        _gen_splits = _num_kv_splits_heuristic(
                            B, H, Sq, D, _dense_light_cu(q.device)
                        )
                    if _gen_splits > 1:
                        generic_splitk = True
                        num_kv_splits = _gen_splits
                        _validate_splitk_workspace_descriptor(H, Sq, D, _gen_splits)
                        # BLOCK_M=64 short-q tile; N128 only for causal D128, else
                        # N32 (matches the generic builder's own path selection).
                        _sk_tag = "N128" if (causal and D == 128) else "N32"
                        exe = _build_dense(
                            num_heads=H,
                            num_kv_heads=num_kv_heads,
                            head_dim=D,
                            causal=causal,
                            dtype_str=dtype_str,
                            cross_seqlen=cross,
                            block_m=64,
                            flat_work_group_size=128,
                            path_tag=_sk_tag,
                            waves_per_eu=waves_per_eu,
                            daz=daz,
                            return_lse=return_lse,
                            window_left=window_left,
                            sm_scale=sm_scale,
                            has_bias=_has_bias,
                            has_dropout=_has_dropout,
                            num_kv_splits=_gen_splits,
                            causal_top_left=_top_left,
                            head_dim_qk=Dqk,
                            head_dim_v=Dv,
                        )
                    else:
                        block_m, flat_work_group_size, path_tag = _dense_generic_tile(
                            B, Sq, H, D, dtype_str, q.device
                        )
                        exe = _build_dense(
                            num_heads=H,
                            num_kv_heads=num_kv_heads,
                            head_dim=D,
                            causal=causal,
                            dtype_str=dtype_str,
                            cross_seqlen=cross,
                            block_m=block_m,
                            flat_work_group_size=flat_work_group_size,
                            path_tag=path_tag,
                            waves_per_eu=waves_per_eu,
                            daz=daz,
                            return_lse=return_lse,
                            window_left=window_left,
                            sm_scale=sm_scale,
                            has_bias=_has_bias,
                            has_dropout=_has_dropout,
                            causal_top_left=_top_left,
                            head_dim_qk=Dqk,
                            head_dim_v=Dv,
                        )

        # ── allocate output ─────────────────────────────────────────────────
        if out is None:
            out_dtype = torch.bfloat16 if dtype_str == "fp8" else q.dtype
            out = torch.empty((*q.shape[:-1], Dv), dtype=out_dtype, device=q.device)
        elif out.shape[-1] != Dv:
            raise ValueError(
                f"flydsl_flash_attn_func: output head_dim must be {Dv}, got {out.shape[-1]}"
            )
        elif dtype_str == "fp8" and out.dtype != torch.bfloat16:
            raise ValueError(
                f"flydsl_flash_attn_func: fp8 output must be bf16, got {out.dtype}"
            )
        elif dtype_str != "fp8" and out.dtype != q.dtype:
            raise ValueError(
                f"flydsl_flash_attn_func: output dtype must match q dtype {q.dtype}, got {out.dtype}"
            )
        # Keep natural shape; flattening can overflow int32 C-ABI dims.
        # Kernels rebuild per-batch descriptors from base pointers and strides.
        if dtype_str == "fp8":
            # The fp8 gfx950 module preserves the original dense ABI from 711.diff:
            # flattened Q/K/V/O tensors plus descale kwargs.
            q_flat = q.contiguous().view(-1)
            k_flat = k.contiguous().view(-1)
            v_flat = v.contiguous().view(-1)
            o_flat = out.contiguous().view(-1)
        else:
            q_flat = q.contiguous()
            k_flat = k.contiguous()
            v_flat = v.contiguous()
            o_flat = out.contiguous()

        lse = (
            torch.empty((B, H, Sq), dtype=torch.float32, device=q.device)
            if return_lse
            else None
        )
        lse_kwargs = {"lse": lse} if return_lse else {}

        # ── launch ──────────────────────────────────────────────────────────
        if splitk:
            _ws = torch.empty(ws_elems, dtype=torch.float32, device=q.device)
            exe(
                q_flat,
                k_flat,
                v_flat,
                o_flat,
                B,
                Sq,
                workspace=_ws,
                stream=launch_stream,
                **lse_kwargs,
            )
        elif varlen:
            kwargs = dict(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                stream=launch_stream,
                **lse_kwargs,
            )
            if kv_seqstart is not None:
                kwargs["kv_seqstart"] = kv_seqstart
            if cross:
                kwargs["seq_len_kv"] = int(max_seqlen_kv)
            if debug_lazy:
                exe(
                    q_flat,
                    k_flat,
                    v_flat,
                    o_flat,
                    B,
                    Sq,
                    debug_counts=debug_counts,
                    **kwargs,
                )
            else:
                exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)
        else:
            kwargs: dict = dict(stream=launch_stream, **lse_kwargs)
            if cross:
                kwargs["seq_len_kv"] = Skv
            if debug_lazy:
                kwargs["debug_counts"] = debug_counts
            if dtype_str == "fp8":
                kwargs.update(
                    q_descale=q_descale, k_descale=k_descale, v_descale=v_descale
                )
            if bias is not None:
                # Contiguous + broadcast to [B, H, Sq, Skv] (unit kv stride; a
                # size-1 head axis gives stride 0 => head-broadcast).
                bias_e = bias.contiguous().expand(B, H, Sq, Skv)
                assert bias_e.stride(3) == 1, "bias kv axis must be unit-stride"
                kwargs.update(
                    bias=bias_e,
                    bias_stride_b=int(bias_e.stride(0)),
                    bias_stride_h=int(bias_e.stride(1)),
                    bias_stride_q=int(bias_e.stride(2)),
                )
            if dropout_mask is not None:
                # Same ABI as bias: unit kv stride, broadcast to [B, H, Sq, Skv].
                drop_e = dropout_mask.contiguous().expand(B, H, Sq, Skv)
                assert drop_e.stride(3) == 1, "dropout kv axis must be unit-stride"
                kwargs.update(
                    dropout=drop_e,
                    dropout_stride_b=int(drop_e.stride(0)),
                    dropout_stride_h=int(drop_e.stride(1)),
                    dropout_stride_q=int(drop_e.stride(2)),
                )
            if generic_splitk:
                # Dense combine writes O in [B, Sq, H, Dv] layout (stride_q_n = H*Dv).
                _ws = torch.empty(
                    dualwave_splitk_workspace_elems(
                        B, H, Sq, int(num_kv_splits), head_dim=D
                    ),
                    dtype=torch.float32,
                    device=q.device,
                )
                kwargs["workspace"] = _ws
                kwargs["stride_q_n"] = H * Dv
            exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)

    return (out, lse) if return_lse else out
