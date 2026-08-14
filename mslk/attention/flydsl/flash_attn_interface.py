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
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: F401  (imported for callers' convenience)

# Re-export so callers only need to import from this module.
from .flash_attn_utils import dualwave_splitk_workspace_elems

__all__ = ["flydsl_flash_attn_func", "dualwave_splitk_workspace_elems"]

_DTYPE_MAP = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float8_e4m3fn: "fp8"}

# Short varlen/paged cases use the lightweight generic path.
_VARLEN_LIGHT_MAX_SEQ = 256
_DENSE_LIGHT_CU_FALLBACK = 256
_DENSE_DUALWAVE_MIN_SEQ = 256
_DENSE_DUALWAVE_LARGE_BATCH = 8
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
    has_bias: bool = False,
):
    if head_dim in (64, 128) and dtype_str in ("bf16", "f16"):
        main_blocks = batch * num_heads * ((seq_len + 127) // 128)
        if main_blocks < _dense_light_cu(device):
            # Additive bias forces the N32 path and reloads the bias plane per Q
            # tile; the block_m=64 light tile reloads it twice as often, so prefer
            # block_m=128 when bias is present (ties at small S, wins at large S).
            if has_bias:
                return 128, 256, "N32"
            return 64, 128, "N32"
    if num_heads >= 32 and batch * seq_len >= _DENSE_M256_MIN_TOKENS:
        return 256, 512, "auto"
    return 128, 256, "auto"


# Generic split-K uses BLOCK_M=64, so the "mtile" (Q rows per workgroup) is 64.
_GENERIC_SPLITK_BLOCK_M = 64


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
):
    """Build (and cache) one dense generic launcher variant."""
    from .flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
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
):
    """Build a lightweight packed-varlen launcher for short attention."""
    from .flash_attn_generic import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
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
) -> torch.Tensor:
    """Native paged-KV attention on the gfx950 dualwave kernel.

    Supported config ONLY (anything else raises): linear/vectorized cache layout
    [NumBlocks, PageSize=64, NumKVHeads, HeadDim], vLLM lookup (block_table +
    seqlen_k), causal, D=64/128, dtype bf16/f16.
    - Dense 4D Q ``[B, Sq, H, D]``: split-K (num_kv_splits>1) supported (seq_len>=384).
    - Varlen packed Q ``[total_q, H, D]`` (cu_seqlens_q given): paged K/V looked up
      per kv-tile via block_table; split-K not supported (matches dense varlen).
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
    elif k.dim() != 4:
        raise ValueError(
            f"flydsl_flash_attn_func: linear paged K/V must be 4D [NumBlocks,PageSize,Hkv,D], got {k.dim()}D"
        )

    varlen = cu_seqlens_q is not None
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
        if num_kv_splits > 1:
            raise NotImplementedError(
                "flydsl_flash_attn_func: varlen paged KV does not support split-K"
            )
        if q.dim() != 3:
            raise ValueError(
                f"flydsl_flash_attn_func: varlen paged q must be 3D [total_q,H,D], got {q.dim()}D"
            )
        _total_q, H, D = q.shape
        B = cu_seqlens_q.numel() - 1
        Sq = int(max_seqlen_q)
    else:
        if q.dim() != 4:
            raise ValueError(
                f"flydsl_flash_attn_func: paged dense q must be 4D [B,Sq,H,D], got {q.dim()}D"
            )
        B, Sq, H, D = q.shape
    if vectorized:
        kvs = 16 // q.element_size()
        Hkv = int(k.shape[1])
        page_size = int(k.shape[3])
        k_head_dim = int(k.shape[2]) * int(k.shape[4])  # (D/kVS) * kVS
        if int(k.shape[4]) != kvs:
            raise ValueError(
                f"flydsl_flash_attn_func: vectorized K last dim ({k.shape[4]}) must equal kVS={kvs}"
            )
    else:
        page_size = int(k.shape[1])
        Hkv = int(k.shape[2])
        k_head_dim = int(k.shape[3])
    # Gappy paged uses the generic kernel (any D the generic path builds; the op
    # reshapes the cache to 64-row sub-pages), so skip the native-paged D/page gates.
    _gappy_paged = kv_seqstart is not None
    if page_size != _PAGED_PAGE_SIZE and not _gappy_paged:
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports page_size={_PAGED_PAGE_SIZE} only, got {page_size}"
        )
    if D not in (64, 128) and not _gappy_paged:
        raise NotImplementedError(
            f"flydsl_flash_attn_func: native paged KV supports head_dim=64 or 128, got {D}"
        )
    if k_head_dim != D:
        raise ValueError(
            f"flydsl_flash_attn_func: paged K head_dim ({k_head_dim}) must match q head_dim ({D})"
        )

    if num_kv_heads is None:
        num_kv_heads = Hkv
    if H % num_kv_heads != 0:
        raise ValueError(
            f"flydsl_flash_attn_func: num_heads ({H}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    # Split-K (paged, dense only): split the KV dimension across grid_z = B*num_kv_splits
    # workgroups + a combine pass. Fills the GPU for low-occupancy shapes (small B / few
    # heads), where single-split paged underutilizes the device.
    splitk = num_kv_splits > 1
    if splitk and (D not in (64, 128) or dtype_str not in ("bf16", "f16") or Sq < 384):
        raise ValueError(
            f"flydsl_flash_attn_func: paged split-K requires D=64/128, dtype bf16/f16, seq_len>=384; "
            f"got D={D}, dtype={dtype_str}, seq_len={Sq}"
        )

    # Per-batch KV lengths differ in general → bottom-right cross-length masking. Varlen
    # paged always uses cross masking (per-batch seqlen_q/seqlen_kv come from cu_seqlens).
    skv = (
        int(max_seqlen_kv) if max_seqlen_kv is not None else int(seqlen_k.max().item())
    )
    max_kv_pages = (skv + page_size - 1) // page_size
    max_pages_per_split = (max_kv_pages + int(num_kv_splits) - 1) // int(num_kv_splits)
    if max_pages_per_split > _PAGED_BT_LDS_SIZE:
        max_supported_kv = _PAGED_BT_LDS_SIZE * int(num_kv_splits) * page_size
        raise NotImplementedError(
            f"flydsl_flash_attn_func: paged KV length {skv} exceeds block-table LDS window "
            f"({_PAGED_BT_LDS_SIZE} pages/split, max_kv_len={max_supported_kv} for "
            f"num_kv_splits={num_kv_splits}, page_size={page_size})"
        )
    if varlen:
        cross = bool(cross_seqlen) if cross_seqlen is not None else True
    else:
        cross = skv != Sq
    block_table_stride = int(block_table.shape[1])
    # Flatten so the kernel's flat row-major index addresses block_table correctly.
    block_table_i32 = (
        (
            block_table
            if block_table.dtype == torch.int32
            else block_table.to(torch.int32)
        )
        .contiguous()
        .reshape(-1)
    )

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        # The dualwave paged softmax overflows for f16 (narrow exponent) and for the
        # causal path at large logits, so route those to the generic light kernel
        # regardless of seqlen; bf16 non-causal keeps the faster dualwave kernel.
        # Mirrors the varlen path's f16 escape.
        _arch = _gpu_arch(q.device)
        _gappy = kv_seqstart is not None
        _paged_light_ok = _gappy or (
            (num_kv_splits <= 1)
            and D in (64, 128)
            and dtype_str in ("bf16", "f16")
            and (
                dtype_str == "f16"
                or causal
                or not _arch.startswith("gfx950")
                or Sq <= _VARLEN_LIGHT_MAX_SEQ
            )
        )
        if return_lse and not _paged_light_ok:
            # Only the generic light path produces LSE; the dualwave native-paged
            # kernel does not. (Short-q / gfx942 / paged-gappy take the light path.)
            raise NotImplementedError(
                "flydsl_flash_attn_func: return_lse for native paged KV is only "
                "supported on the generic (short-seq / gfx942) path"
            )
        if _paged_light_ok:
            exe = _build_paged_light(
                num_heads=H,
                num_kv_heads=num_kv_heads,
                head_dim=D,
                causal=causal,
                dtype_str=dtype_str,
                cross_seqlen=cross,
                varlen=varlen,
                kv_cache_layout=kv_cache_layout,
                waves_per_eu=waves_per_eu,
                daz=daz,
                lazy_rescale=dualwave_swp_lazy_rescale,
                setprio=dualwave_swp_setprio,
                debug_lazy_counts=False,
                enable_stagger=dualwave_swp_enable_stagger,
                gappy_kv=_gappy,
                return_lse=return_lse,
                sm_scale=sm_scale,
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
                varlen=varlen,
                kv_cache_layout=kv_cache_layout,
            )
        if out is None:
            out = torch.empty_like(q)
        # Keep tensors in natural shape; flattening can overflow int32 C-ABI dims.
        # The paged kernel rebuilds per-batch/page descriptors from base pointers.
        q_flat = q.contiguous()
        k_flat = k.contiguous()
        v_flat = v.contiguous()
        o_flat = out.contiguous()
        kwargs = dict(
            block_table=block_table_i32,
            block_table_stride=block_table_stride,
            stream=launch_stream,
        )
        if varlen:
            kwargs["cu_seqlens_q"] = cu_seqlens_q
            kwargs["cu_seqlens_kv"] = cu_seqlens_kv
        if kv_seqstart is not None:
            kwargs["kv_seqstart"] = kv_seqstart
        if cross:
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
                B, H, Sq, int(num_kv_splits), head_dim=D
            )
            _ws = torch.empty(ws_elems, dtype=torch.float32, device=q.device)
            kwargs["workspace"] = _ws
        exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)

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
    lazy_rescale: bool,
    setprio: bool,
    debug_lazy_counts: bool,
    enable_stagger: bool,
    gappy_kv: bool = False,
    return_lse: bool = False,
    sm_scale: Optional[float] = None,
):
    """Build a lightweight paged-varlen launcher for short attention."""
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
        block_m=64,
        flat_work_group_size=128,
        path_tag="N32",
        waves_per_eu=waves_per_eu,
        daz=daz,
        gappy_kv=gappy_kv,
        return_lse=return_lse,
        sm_scale=sm_scale,
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
    # Split-K (gfx950 only, seq_len >= 384, D=64/128, bf16/f16).
    num_kv_splits: int = 1,
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
    # Sliding-window: keep KV in (q_row - window_left, q_row]; -1 disables.
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
        v: Value tensor, same shape as k.
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
        num_kv_splits: Split-K factor (>1: gfx950 only, D=64/128, bf16/f16, seq>=384).
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
        Output tensor with the same shape as q (dtype bf16 for fp8 inputs,
        otherwise the same dtype as q). When ``return_lse=True``, returns a
        ``(out, lse)`` tuple instead, where ``lse`` is float32 ``[B, H, Sq]``
        (varlen: ``[B, H, max_seqlen_q]``; padded rows undefined).
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
        # Window lives in apply_kv_mask (dense + generic varlen); fp8/paged lack it.
        if dtype_str == "fp8" or paged_kv:
            raise NotImplementedError(
                "flydsl_flash_attn_func: sliding-window (window_left>=0) is not "
                "supported for fp8 or paged KV"
            )
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
        )

    varlen = cu_seqlens_q is not None

    if dtype_str == "fp8":
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
    if D < 64 or D % 32 != 0:
        raise ValueError(
            f"flydsl_flash_attn_func: head_dim ({D}) must be >= 64 and a multiple of 32"
        )

    splitk = num_kv_splits > 1
    # generic_splitk is chosen internally per-shape (below); it uses the generic
    # BLOCK_M=64 kernel + dense combine, distinct from the dualwave `splitk` above.
    generic_splitk = False

    # ── split-K eligibility guard (SKIP analogous to run_splitk_config) ────
    if splitk:
        if D not in (64, 128) or dtype_str not in ("bf16", "f16") or Sq < 384:
            raise ValueError(
                f"flydsl_flash_attn_func: split-K requires D=64/128, dtype bf16/f16, seq_len>=384; "
                f"got D={D}, dtype={dtype_str}, seq_len={Sq}"
            )
        ws_elems = dualwave_splitk_workspace_elems(
            B, H, Sq, int(num_kv_splits), head_dim=D
        )

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
                if (
                    not _windowed
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
                    # the GPU: BLOCK_M=64, gated by CK's occupancy heuristic. Dropout
                    # disables it (matches CK); bias/window/scale compose fine.
                    _gen_splits = 1
                    if (
                        not _has_dropout
                        and not cross
                        and D in (64, 128, 256)
                        and dtype_str in ("bf16", "f16")
                    ):
                        _gen_splits = _num_kv_splits_heuristic(
                            B, H, Sq, D, _dense_light_cu(q.device)
                        )
                    if _gen_splits > 1:
                        generic_splitk = True
                        num_kv_splits = _gen_splits
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
                        )
                    else:
                        block_m, flat_work_group_size, path_tag = _dense_generic_tile(
                            B, Sq, H, D, dtype_str, q.device, has_bias=_has_bias
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
                        )

        # ── allocate output ─────────────────────────────────────────────────
        if out is None:
            out_dtype = torch.bfloat16 if dtype_str == "fp8" else q.dtype
            out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
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
        lse_kwargs = dict(lse=lse) if return_lse else {}

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
                # Dense combine writes O in [B, Sq, H, D] layout (stride_q_n = H*D).
                _ws = torch.empty(
                    dualwave_splitk_workspace_elems(
                        B, H, Sq, int(num_kv_splits), head_dim=D
                    ),
                    dtype=torch.float32,
                    device=q.device,
                )
                kwargs["workspace"] = _ws
                kwargs["stride_q_n"] = H * D
            exe(q_flat, k_flat, v_flat, o_flat, B, Sq, **kwargs)

    return (out, lse) if return_lse else out
