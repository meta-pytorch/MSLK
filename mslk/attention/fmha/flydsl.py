# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe
"""FlyDSL FMHA backward, registered as an opt-in `AttentionBwOpBase`.

This lets FlyDSL's MFMA backward kernel (`mslk.attention.flydsl.fmha_bwd_mfma`)
be exercised through the SAME `test/attention/fmha/test_backward.py` machinery
that judges CK's `ck.BwOp` (both are covered by `test_backward`'s `ALL_BW_OPS`
parametrization there).

Part of `ALL_BW_OPS` in `mslk/attention/fmha/__init__.py` (guarded by
`torch.version.hip`) for TEST-ENUMERATION purposes only -- this broadens which
test files (`test_mem_eff_attention.py`, `test_forward.py`) exercise this op.
It is deliberately NOT wired into `dispatch.py`'s live `_dispatch_bw()`
priority list, which still hardcodes `[ck.BwOp]` on ROCm -- this op is not a
live production backend.

Stride-aware addressing: the kernel supports Q/K/V/dO with a possibly-non-
contiguous per-tensor row pitch — e.g. a packed-`qkv` tensor sliced/unbound
into Q/K/V views. The kernel still requires each tensor to be flat *within* a
row (last dim D stride-1, H-axis stride exactly D — i.e. only the outer
B/M-or-N row pitch may differ from the contiguous `H*D`); this is checked in
`not_supported_reasons()` below. Real per-axis arbitrary strides (transposed H,
non-unit D stride) remain unsupported.

Non-goals (intentionally excluded rather than left as silent gaps -- each
entry below is enforced generically, see the cited mechanism, and has its own
negative-path test in `test_backward.py`):
- Dropout (`d.p != 0.0`): no MSLK caller passes dropout through this op;
  rejected generically by `AttentionOpBase.not_supported_reasons()`'s
  `(d.p != 0.0) and not cls.SUPPORTS_DROPOUT` check (`SUPPORTS_DROPOUT = False`
  below). Revisit if `flydsl.BwOp` is ever added to a live dispatch path and a
  caller trips this.
- True 5D BMGHK (`d.query.ndim == 5`, multi-query-group layout distinct from
  GQA-via-4D-broadcast, which IS supported): no MSLK caller found; rejected
  generically by the same `not_supported_reasons()`'s
  `not cls.SUPPORTS_BMGHK and d.query.ndim == 5` check (`SUPPORTS_BMGHK = False`
  below). Revisit under the same condition as dropout.
- Paged-KV / gappy-keys backward (`PagedBlockDiagonal*Mask`,
  `BlockDiagonal*GappyKeys*Mask`): these are inference-serving-only shapes in
  MSLK (`tree_attention.py`'s forward-only code path never reaches
  `flydsl.BwOp`); no backward caller found. Rejected generically via
  `SUPPORTED_ATTN_BIAS_TYPES` not listing them (falls through to the base
  class's `type(d.attn_bias) not in cls.SUPPORTED_ATTN_BIAS_TYPES` check).
  Revisit if a backward caller for paged/gappy attention appears.
- Tensor-bias / ALiBi (`LowerTriangularMaskWithTensorBias` and similar): no
  MSLK caller found for this backward op. Rejected the same way, via
  `SUPPORTED_ATTN_BIAS_TYPES` omission.
- Bottom-right / local-window varlen (`BlockDiagonalCausalFromBottomRightMask`
  and other non-top-left alignment variants): only top-left causal alignment
  (`LowerTriangularMask`, `BlockDiagonalCausalMask`) is implemented;
  bottom-right/local-window semantics are separately scoped and not
  implemented. Rejected the same way, via `SUPPORTED_ATTN_BIAS_TYPES`
  omission. Revisit if a caller needs bottom-right or local-window varlen
  causal masking specifically.
"""

import math
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple

import torch

from .attn_bias import (
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalMask,
    BlockDiagonalPaddedKeysMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)
from .common import AttentionBwOpBase, AttentionFwOpBase, Context, Gradients, Inputs
from .utils.op_common import get_operator, register_operator


def _uniform_row_pitch_reason(
    name: str, t: torch.Tensor, allow_broadcast_heads: bool = False
) -> Optional[str]:
    """Validate the "flat within a row, possibly-non-contiguous row pitch"
    assumption the kernel's stride-aware addressing relies on: last dim (D)
    must be stride-1, and the H-axis must be flat relative to D
    (stride(2) == D) — only the outer B/M(or N)-axis row pitch may exceed the
    contiguous H*D. Returns a reason string if unsupported, else None. A
    stride-0 (broadcast/expand) row pitch is explicitly rejected: it would
    alias every row onto row 0 under the `row_pos * stride` formula.

    allow_broadcast_heads (GQA): key/value may ALSO have stride(-2) == 0 (a
    `.expand()`-broadcast H axis, e.g. `key[:, :, :1].expand(-1, -1, Hq, -1)`)
    -- this means the tensor's REAL KV-head count is 1, not shape[2]. Query
    never gets this exception (no broadcast-Q case exists).
    """
    D = t.shape[-1]
    if t.stride(-1) != 1:
        return f"{name}'s last dim (head_dim) must be stride-1"
    if allow_broadcast_heads and t.stride(-2) == 0:
        return None
    if t.stride(-2) != D:
        return f"{name}'s head axis must be flat relative to head_dim (stride(-2) == D)"
    if t.stride(-3) % D != 0:
        return f"{name}'s row pitch (stride(-3)) must be a multiple of head_dim"
    if t.stride(-3) == 0:
        return f"{name} has a broadcast (stride-0) row pitch, not supported"
    return None


def _num_kv_heads(key: torch.Tensor) -> int:
    """Real number of distinct KV heads (GQA).

    `key.stride(2) == 0` means every logical head slot aliases the same
    underlying head (a `.expand()` broadcast, e.g. MQA-via-broadcast) -- the
    real count is 1, not `key.shape[2]`. Otherwise key is a genuinely
    distinctly-shaped `(B, N, Hkv, D)` tensor (contiguous, stride(2) == D).
    """
    return 1 if key.stride(2) == 0 else key.shape[2]


# Cache of flyc.compile()'d kernels, keyed on the compile-time-constant params
# baked into the kernel body (D, dtype, tile sizes, scale, causal, GQA ratio,
# varlen). `flyc.compile()`'s underlying MLIR/LLVM artifact is itself cached
# (FlyDSL's own on-disk+memory cache, keyed by kernel source + compile-time
# closure values), but `flyc.compile()` ITSELF re-traces/re-binds the Python
# call on every invocation, dominated by `JitFunction.__call__`'s signature
# binding, not by GPU work -- see `CompiledFunction`'s docstring in FlyDSL:
# the whole point of caching the returned `CompiledFunction` object ourselves
# is to skip straight to its cheap `__call__` hot path instead of re-entering
# `flyc.compile()` every backward call. B/M/N/H/n_tiles/data-pointers are all
# runtime `fx.Int32`/tensor args in the kernel signature (not baked at compile
# time), so ONE cached `CompiledFunction` per key correctly serves any
# shape/batch at that (D, dtype, tile, scale, causal, heads_per_kv, varlen,
# arch) combination.
_dvdk_kernel_cache: dict = {}
_dq_kernel_cache: dict = {}
_dqdkdv_kernel_cache: dict = {}
_gfx950_kernel_cache: dict = {}
_preprocess_kernel_cache: dict = {}
_convert_dq_kernel_cache: dict = {}


@torch.library.custom_op(
    "mslk_flydsl::fmha_bwd",
    mutates_args=(),
    device_types=["cuda"],
)
def _flydsl_bwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad: torch.Tensor,
    scale: float,
    causal: bool,
    seqstart_q: Optional[torch.Tensor] = None,
    seqstart_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pyre-ignore[21]: FlyDSL is available only in AMD builds.
    import flydsl.compiler as flyc
    from mslk.attention.flydsl.fmha_bwd_mfma import (
        compile_fmha_bwd_dq_mfma,
        compile_fmha_bwd_dqdkdv_mfma,
        compile_fmha_bwd_dvdk_mfma,
    )

    # Varlen (non-causal or causal BlockDiagonalMask): query/key/value arrive
    # already reshaped to (1, sum_M_i, H, D) -- B_logical (the real batch
    # count) is `seqstart_q.shape[0] - 1`, NOT query.shape[0] (always 1 under
    # group-mode). Non-varlen: B == query.shape[0].
    varlen = seqstart_q is not None
    H, D = query.shape[2], query.shape[3]
    total_m = query.shape[1]  # real physical M extent (== sum_M_i under varlen)
    total_n = key.shape[1]  # real physical N extent (== sum_N_i under varlen)
    if varlen:
        assert seqstart_k is not None
        B = seqstart_q.shape[0] - 1
        M = int(seqstart_q.diff().max().item())  # max_seqlen_q -- grid/loop sizing only
        N = int(seqstart_k.diff().max().item())  # max_seqlen_k -- grid/loop sizing only
    else:
        B, M = query.shape[0], query.shape[1]
        N = total_n
    device = query.device
    dtype = query.dtype
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    # GQA: H_kv < H is either a genuine distinctly-shaped (B,N,Hkv,D) tensor
    # or a stride-0 `.expand()` broadcast (H_kv=1 in that case, regardless of
    # key.shape[2]) -- see _num_kv_heads. heads_per_kv is passed to the
    # kernel as a COMPILE-TIME constant (like `causal`), not a runtime arg.
    H_kv = _num_kv_heads(key)
    heads_per_kv = H // H_kv

    def _as_i16(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.int16)

    # Stride-aware addressing: Q/K/V/dO may have a non-contiguous row pitch
    # (e.g. a packed-qkv unbind view, stride(1) > H*D) as long as they're
    # flat within a row (stride(-1)==1, stride(2)==D — this is enforced for
    # query/key/value by `not_supported_reasons` before we get here). `grad`
    # isn't visible to `not_supported_reasons` (it's not part of `Inputs`),
    # so validate it here too.
    grad_reason = _uniform_row_pitch_reason("grad", grad)
    if grad_reason is not None:
        raise NotImplementedError(grad_reason)

    # Keep Q/K/V/dO 4D (never collapse (B,M,H,D) -> (B*M*H,D) via `.view()`,
    # which would raise on the non-contiguous case) and pass the real row
    # pitch (in row units, i.e. elem_stride // D) as a runtime kernel arg
    # instead.
    Q_4d = _as_i16(query)
    K_4d = _as_i16(key)
    V_4d = _as_i16(value)
    dO_4d = _as_i16(grad)
    q_stride_m = query.stride(1) // D
    kv_stride_n = key.stride(1) // D
    do_stride_m = grad.stride(1) // D

    # `lse` is [B, H, M] float32 (non-varlen, B*H*M elements) or packed
    # [1, H, sum_M] under varlen (VARLEN_LSE_PACKED=True convention, H*
    # total_m elements -- see fmha_bwd_mfma.py's _lse_row). `out`/`grad` are
    # [B, M, H, D] (non-varlen) or [1, total_m, H, D] (varlen), so their
    # per-row-summed D_vec has the same B*H*M vs H*total_m element count
    # split. Non-varlen's `total_m` is already per-batch M (== query.shape[1]),
    # NOT B*M, so the varlen-only element count (H*total_m) would silently
    # undercount by a factor of B here -- must branch on `varlen` explicitly.
    # These are plain elementwise/reduction ops producing fresh contiguous
    # tensors, so `.view()` on their results is always safe regardless of the
    # strides of `out`/`grad`/`lse` themselves.
    _lse_dvec_rows = H * total_m if varlen else B * H * total_m
    LSE_2d = lse.contiguous().view(_lse_dvec_rows, 1)
    stream = torch.cuda.current_stream()
    # D_vec = rowsum(dO * O) per query row. Use the FlyDSL preprocess kernel
    # when D is compatible (D%64==0), otherwise fall back to PyTorch ops.
    _use_preprocess_kernel = D % 64 == 0
    if _use_preprocess_kernel:
        from mslk.attention.flydsl.fmha_bwd_preprocess import (
            compile_fmha_bwd_preprocess,
        )

        D_vec = torch.empty(_lse_dvec_rows, 1, device=device, dtype=torch.float32)
        dO_2d = _as_i16(grad.contiguous()).view(_lse_dvec_rows, D)
        O_2d = _as_i16(out.contiguous()).view(_lse_dvec_rows, D)
        preprocess_key = (D, dtype_str)
        compiled_preprocess = _preprocess_kernel_cache.get(preprocess_key)
        if compiled_preprocess is None:
            launch_preprocess = compile_fmha_bwd_preprocess(D=D, dtype_str=dtype_str)
            tmp_dvec = torch.empty_like(D_vec)
            compiled_preprocess = flyc.compile(
                launch_preprocess, dO_2d, O_2d, tmp_dvec, _lse_dvec_rows, stream
            )
            _preprocess_kernel_cache[preprocess_key] = compiled_preprocess
        compiled_preprocess(dO_2d, O_2d, D_vec, _lse_dvec_rows, stream)
    else:
        D_vec = (grad.float() * out.float()).sum(dim=-1).view(_lse_dvec_rows, 1)

    # dV/dK are shaped [B, N, H_kv, D] under GQA (one gradient per KV head, summed
    # over its heads_per_kv group by the kernel's grid-regroup-by-KV-head -- see
    # fmha_bwd_mfma.py's compile_fmha_bwd_dvdk_mfma docstring); H_kv == H otherwise.
    # Physical output-buffer element count: under varlen, `total_m`/`total_n` are
    # already the real packed extent (B collapsed to 1 in the tensor's own
    # shape); under non-varlen, `total_m`/`total_n` are `query.shape[1]`/
    # `key.shape[1]` -- i.e. PER-BATCH M/N, not the B*M/B*N physical element
    # count -- so B must be multiplied in explicitly here (same trap as
    # `_lse_dvec_rows` above).
    alloc_m = total_m if varlen else B * total_m
    alloc_n = total_n if varlen else B * total_n
    dV_out = torch.zeros(alloc_n * H_kv * D, 1, device=device, dtype=torch.float32)
    dK_out = torch.zeros(alloc_n * H_kv * D, 1, device=device, dtype=torch.float32)
    dQ_out = torch.zeros(alloc_m * H * D, 1, device=device, dtype=torch.float32)

    gpu_arch = torch.cuda.get_device_properties(device).gcnArchName
    _is_gfx950 = "gfx950" in gpu_arch
    # Varlen: pass the real seqstart tensors; non-varlen passes dummies
    # (unused since varlen=False at compile time -- see
    # compile_fmha_bwd_dvdk_mfma/compile_fmha_bwd_dqdkdv_mfma's `varlen` kwarg).
    _dummy_seqstart = torch.zeros(1, device=device, dtype=torch.int32)
    _seqstart_q_arg = seqstart_q if varlen else _dummy_seqstart
    _seqstart_k_arg = seqstart_k if varlen else _dummy_seqstart

    # gfx950 production path (mslk.attention.flydsl.fmha_bwd_mfma_gfx950),
    # replacing the older fmha_bwd_mfma.py dqdkdv kernel on this arch (gfx942
    # is untouched below -- this kernel hard-asserts gfx950). The kernel
    # writes dV/dK PER QUERY HEAD, uncompacted (`ck_scope_dvdk=True`), and the
    # GQA-combine reduction happens HERE, outside the kernel, via
    # `.unflatten().sum()` -- see the reduce below. dQ stays atomic-add
    # (`deterministic=False`); the deterministic branch is not yet the
    # default for any caller.
    if _is_gfx950:
        from mslk.attention.flydsl.fmha_bwd_mfma_gfx950 import (
            compile_fmha_bwd_dqdkdv_mfma_gfx950,
            gfx950_tile_defaults,
        )

        BLOCK_M_CK, BLOCK_N_CK = gfx950_tile_defaults(D, N, gpu_arch)
        n_M_tiles = (M + BLOCK_M_CK - 1) // BLOCK_M_CK
        # dV/dK per-query-head (uncompacted [B,N,H,D]) under ck_scope_dvdk --
        # see compile_fmha_bwd_dqdkdv_mfma_gfx950's own docstring/kwarg doc.
        dV_out = torch.zeros(alloc_n * H * D, 1, device=device, dtype=torch.float32)
        dK_out = torch.zeros(alloc_n * H * D, 1, device=device, dtype=torch.float32)
        args_gfx950 = (
            Q_4d,
            K_4d,
            V_4d,
            dO_4d,
            dV_out,
            dK_out,
            dQ_out,
            LSE_2d,
            D_vec,
            B,
            M,
            N,
            H,
            n_M_tiles,
            q_stride_m,
            kv_stride_n,
            do_stride_m,
            _seqstart_q_arg,
            _seqstart_k_arg,
            total_m,
            stream,
        )
        gfx950_key = (
            D,
            dtype_str,
            BLOCK_M_CK,
            BLOCK_N_CK,
            scale,
            gpu_arch,
            causal,
            heads_per_kv,
            varlen,
        )
        compiled_gfx950 = _gfx950_kernel_cache.get(gfx950_key)
        if compiled_gfx950 is None:
            launch_gfx950 = compile_fmha_bwd_dqdkdv_mfma_gfx950(
                D=D,
                dtype_str=dtype_str,
                BLOCK_M=BLOCK_M_CK,
                BLOCK_N=BLOCK_N_CK,
                scale=scale,
                causal=causal,
                heads_per_kv=heads_per_kv,
                varlen=varlen,
                gpu_arch=gpu_arch,
                deterministic=False,
                ck_scope_dvdk=True,
            )
            # flyc.compile executes the kernel once (JIT warm run) -- dQ (and
            # dV/dK, always atomic-add under ck_scope_dvdk's per-query-head
            # scope) would otherwise bake a spurious extra atomic contribution
            # into the real output before the real call below even runs (same
            # precedent as the old dqdkdv path above / every gfx950 kernel test).
            tmp_dV = torch.zeros_like(dV_out)
            tmp_dK = torch.zeros_like(dK_out)
            tmp_dQ = torch.zeros_like(dQ_out)
            args_compile = (
                Q_4d,
                K_4d,
                V_4d,
                dO_4d,
                tmp_dV,
                tmp_dK,
                tmp_dQ,
                LSE_2d,
                D_vec,
                B,
                M,
                N,
                H,
                n_M_tiles,
                q_stride_m,
                kv_stride_n,
                do_stride_m,
                _seqstart_q_arg,
                _seqstart_k_arg,
                total_m,
                stream,
            )
            compiled_gfx950 = flyc.compile(launch_gfx950, *args_compile)
            _gfx950_kernel_cache[gfx950_key] = compiled_gfx950
        compiled_gfx950(*args_gfx950)

        out_b = 1 if varlen else B
        out_m = total_m if varlen else M
        out_n = total_n if varlen else N
        # dQ convert: f32 accumulator -> output dtype. Use a dedicated
        # FlyDSL kernel to avoid PyTorch .to() kernel launch overhead.
        dq_n_elems = alloc_m * H * D
        from mslk.attention.flydsl.fmha_bwd_convert_dq import (
            compile_fmha_bwd_convert_dq,
        )

        dq_converted = torch.empty(dq_n_elems, 1, device=device, dtype=dtype)
        convert_key = dtype_str
        compiled_convert = _convert_dq_kernel_cache.get(convert_key)
        if compiled_convert is None:
            launch_convert = compile_fmha_bwd_convert_dq(dtype_str=dtype_str)
            tmp_dq_conv = torch.empty_like(dq_converted)
            compiled_convert = flyc.compile(
                launch_convert, dQ_out, tmp_dq_conv, dq_n_elems, stream
            )
            _convert_dq_kernel_cache[convert_key] = compiled_convert
        compiled_convert(dQ_out, dq_converted, dq_n_elems, stream)
        dq = dq_converted.view(out_b, out_m, H, D)
        # GQA-combine: dV/dK came back per-query-head ([B,N,H,D], H not Hkv).
        # Reduce across the heads_per_kv group and convert to output dtype.
        # heads_per_kv==1 (MHA/no GQA): sum over a size-1 axis, a no-op.
        dk = (
            dK_out.view(out_b, out_n, H, D)
            .unflatten(2, (H_kv, heads_per_kv))
            .sum(3)
            .to(dtype)
        )
        dv = (
            dV_out.view(out_b, out_n, H, D)
            .unflatten(2, (H_kv, heads_per_kv))
            .sum(3)
            .to(dtype)
        )
        return dq, dk, dv

    # Fused (dqdkdv) is the primary path -- measured (A3_ck_flyDSL_compare.md
    # SS7.18) to beat the split dvdk+dq path by ~2x-13x on BOTH gfx950 and
    # gfx942, at every shape tested, despite gfx942 getting none of gfx950's
    # additional wins (use_trload/BLOCK_N=128/M_SPLIT are gfx950-only -- see
    # SS7.14-SS7.17). The split path is now ONLY used as a fallback for the one
    # shape dqdkdv structurally cannot serve: D=256 on gfx942 (its extra LDS_K
    # buffer for the fused dQ contraction doesn't fit gfx942's 64KB at ANY
    # valid BLOCK_M -- see compile_fmha_bwd_dqdkdv_mfma's own docstring and
    # test_fmha_bwd_dqdkdv_mfma.py's D=256/gfx942 skip).
    #
    # gfx942-only from here on (gfx950 always returns above).
    _dqdkdv_unsupported = D >= 256

    if not _dqdkdv_unsupported:
        _GFX950_CU_COUNT = 256

        def _gqa_m_split_gfx950(B_, H_, Hkv_, D_, M_, N_, causal_, block_m_, block_n_):
            heads_per_kv_ = H_ // Hkv_
            if heads_per_kv_ == 1:
                return 1
            if D_ == 128 and not causal_:
                return 1
            if D_ not in (64, 128):
                return 8
            num_N_tiles_ = -(-N_ // block_n_)
            n_M_tiles_ratio_ = -(-M_ // block_m_)
            base_grid_ = B_ * Hkv_ * num_N_tiles_
            ratio_ = (n_M_tiles_ratio_ * heads_per_kv_ * base_grid_) / _GFX950_CU_COUNT
            if causal_:
                return 8
            return 8 if ratio_ <= 32 else 2

        if _is_gfx950:
            BLOCK_M_DQDKDV = 64
            BLOCK_N_DQDKDV = 64 if D == 256 else 128
            USE_TRLOAD_DQDKDV = True
            M_SPLIT_DQDKDV = _gqa_m_split_gfx950(
                B, H, H_kv, D, M, N, causal, BLOCK_M_DQDKDV, BLOCK_N_DQDKDV
            )
        else:
            BLOCK_M_DQDKDV = 32 if D >= 128 else 64
            BLOCK_N_DQDKDV = 64
            USE_TRLOAD_DQDKDV = False
            M_SPLIT_DQDKDV = 1
        n_M_tiles = (M + BLOCK_M_DQDKDV - 1) // BLOCK_M_DQDKDV
        args_dqdkdv = (
            Q_4d,
            K_4d,
            V_4d,
            dO_4d,
            dV_out,
            dK_out,
            dQ_out,
            LSE_2d,
            D_vec,
            B,
            M,
            N,
            H,
            n_M_tiles,
            q_stride_m,
            kv_stride_n,
            do_stride_m,
            _seqstart_q_arg,
            _seqstart_k_arg,
            total_m,
            stream,
        )
        dqdkdv_key = (
            D,
            dtype_str,
            BLOCK_M_DQDKDV,
            BLOCK_N_DQDKDV,
            scale,
            gpu_arch,
            causal,
            heads_per_kv,
            varlen,
            USE_TRLOAD_DQDKDV,
            M_SPLIT_DQDKDV,
        )
        compiled_dqdkdv = _dqdkdv_kernel_cache.get(dqdkdv_key)
        if compiled_dqdkdv is None:
            launch_dqdkdv = compile_fmha_bwd_dqdkdv_mfma(
                D=D,
                dtype_str=dtype_str,
                BLOCK_M=BLOCK_M_DQDKDV,
                BLOCK_N=BLOCK_N_DQDKDV,
                scale=scale,
                use_trload=USE_TRLOAD_DQDKDV,
                M_SPLIT=M_SPLIT_DQDKDV,
                gpu_arch=gpu_arch,
                causal=causal,
                heads_per_kv=heads_per_kv,
                varlen=varlen,
            )
            # flyc.compile executes the kernel once (JIT warm run) as part of
            # compilation -- unlike the old dvdk/dq split path (plain stores,
            # harmless to warm-compile against the real buffers), dqdkdv's dQ
            # (and dV/dK when M_SPLIT>1) is ALWAYS atomic-add, so compiling
            # against the real dV_out/dK_out/dQ_out would bake one spurious
            # extra atomic contribution into the real output before the actual
            # call below even runs. Compile against throwaway buffers instead
            # (mirrors every dqdkdv test file's identical, load-bearing
            # pattern -- see e.g. test_fmha_bwd_dqdkdv_mfma.py's own comment).
            tmp_dV = torch.zeros_like(dV_out)
            tmp_dK = torch.zeros_like(dK_out)
            tmp_dQ = torch.zeros_like(dQ_out)
            args_compile = (
                Q_4d,
                K_4d,
                V_4d,
                dO_4d,
                tmp_dV,
                tmp_dK,
                tmp_dQ,
                LSE_2d,
                D_vec,
                B,
                M,
                N,
                H,
                n_M_tiles,
                q_stride_m,
                kv_stride_n,
                do_stride_m,
                _seqstart_q_arg,
                _seqstart_k_arg,
                total_m,
                stream,
            )
            compiled_dqdkdv = flyc.compile(launch_dqdkdv, *args_compile)
            _dqdkdv_kernel_cache[dqdkdv_key] = compiled_dqdkdv
        compiled_dqdkdv(*args_dqdkdv)
    else:
        # Split kernels (dvdk + dq) -- fallback for D=256/gfx942 only (see
        # _dqdkdv_unsupported above). dvdk's Q/dO/P/dS LDS footprint at
        # BLOCK_M=32 is 42.5KB for D=256 on gfx942, fits within 64KB
        # (confirmed via a real LLVM "local memory exceeds limit" error on
        # gfx942 hardware at larger BLOCK_M -- see compile_fmha_bwd_dvdk_mfma's
        # own gfx942 mitigation, same style as dqdkdv's BLOCK_M shrink above).
        BLOCK_M_DVDK = 32
        BLOCK_N_DVDK = 64
        n_M_tiles = (M + BLOCK_M_DVDK - 1) // BLOCK_M_DVDK
        args_dvdk = (
            Q_4d,
            K_4d,
            V_4d,
            dO_4d,
            dV_out,
            dK_out,
            LSE_2d,
            D_vec,
            B,
            M,
            N,
            H,
            n_M_tiles,
            q_stride_m,
            kv_stride_n,
            do_stride_m,
            _seqstart_q_arg,
            _seqstart_k_arg,
            total_m,
            stream,
        )
        dvdk_key = (
            D,
            dtype_str,
            BLOCK_M_DVDK,
            BLOCK_N_DVDK,
            scale,
            gpu_arch,
            causal,
            heads_per_kv,
            varlen,
        )
        compiled_dvdk = _dvdk_kernel_cache.get(dvdk_key)
        if compiled_dvdk is None:
            launch_dvdk = compile_fmha_bwd_dvdk_mfma(
                D=D,
                dtype_str=dtype_str,
                BLOCK_M=BLOCK_M_DVDK,
                BLOCK_N=BLOCK_N_DVDK,
                scale=scale,
                use_pipeline=True,
                gpu_arch=gpu_arch,
                causal=causal,
                heads_per_kv=heads_per_kv,
                varlen=varlen,
            )
            compiled_dvdk = flyc.compile(launch_dvdk, *args_dvdk)
            _dvdk_kernel_cache[dvdk_key] = compiled_dvdk
        compiled_dvdk(*args_dvdk)

        # dq's K/V/dS LDS footprint at BLOCK_N=32 is 36KB for D=256 on gfx942,
        # fits within 64KB (same mitigation style as dvdk's BLOCK_M above).
        BLOCK_M_DQ = 64
        BLOCK_N_DQ = 32
        n_N_tiles = (N + BLOCK_N_DQ - 1) // BLOCK_N_DQ
        args_dq = (
            Q_4d,
            K_4d,
            V_4d,
            dO_4d,
            dQ_out,
            LSE_2d,
            D_vec,
            B,
            M,
            N,
            H,
            n_N_tiles,
            q_stride_m,
            kv_stride_n,
            do_stride_m,
            _seqstart_q_arg,
            _seqstart_k_arg,
            total_m,
            stream,
        )
        dq_key = (
            D,
            dtype_str,
            BLOCK_M_DQ,
            BLOCK_N_DQ,
            scale,
            gpu_arch,
            causal,
            heads_per_kv,
            varlen,
        )
        compiled_dq = _dq_kernel_cache.get(dq_key)
        if compiled_dq is None:
            launch_dq = compile_fmha_bwd_dq_mfma(
                D=D,
                dtype_str=dtype_str,
                BLOCK_M=BLOCK_M_DQ,
                BLOCK_N=BLOCK_N_DQ,
                scale=scale,
                gpu_arch=gpu_arch,
                causal=causal,
                heads_per_kv=heads_per_kv,
                varlen=varlen,
            )
            compiled_dq = flyc.compile(launch_dq, *args_dq)
            _dq_kernel_cache[dq_key] = compiled_dq
        compiled_dq(*args_dq)

    # Output shapes: under varlen, B collapses to 1 and total_m/total_n are
    # already the real packed sum_M_i/sum_N_i (matching Q/K/V's own physical
    # shape); under non-varlen, the real batch axis is B and the per-batch
    # extent is M/N (== total_m/total_n, which is per-batch here, not B*M/B*N
    # -- see alloc_m/alloc_n above for why that distinction matters).
    out_b = 1 if varlen else B
    out_m = total_m if varlen else M
    out_n = total_n if varlen else N
    dq = dQ_out.view(out_b, out_m, H, D).to(dtype)
    dk = dK_out.view(out_b, out_n, H_kv, D).to(dtype)
    dv = dV_out.view(out_b, out_n, H_kv, D).to(dtype)
    return dq, dk, dv


@torch.library.register_fake("mslk_flydsl::fmha_bwd")
def _flydsl_bwd_abstract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad: torch.Tensor,
    scale: float,
    causal: bool,
    seqstart_q: Optional[torch.Tensor] = None,
    seqstart_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(query),
        torch.empty_like(key),
        torch.empty_like(value),
    )


@register_operator
class BwOp(AttentionBwOpBase):
    """Opt-in backward op wrapping FlyDSL's MFMA dV/dK/dQ kernels.

    Part of `ALL_BW_OPS` (test-enumeration only, ROCm-only) — see
    `test/attention/fmha/test_backward.py`'s `ALL_BW_OPS` parametrization for
    the dedicated matrix, and the module docstring above for why this isn't in
    live dispatch.
    """

    OPERATOR = get_operator("mslk_flydsl", "fmha_bwd")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.bfloat16, torch.float16}
    # Kernel wave-tiling requires D a multiple of 32 (D=32/96 done via
    # ceil-div wave assignment + out-of-range guards, see fmha_bwd_mfma.py's
    # D_SUBS_PER_WAVE comment).
    SUPPORTED_MAX_K = 256
    SUPPORTED_MIN_K = 32
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    # GQA is supported via the plain 4D BMHK path (key/value with Hkv < Hq,
    # either a genuine (B,N,Hkv,D) tensor or a stride-0 `.expand()` broadcast
    # -- see _num_kv_heads/_uniform_row_pitch_reason), matching how
    # test_backward_gqa uses it. SUPPORTS_BMGHK covers a DIFFERENT,
    # unrelated case (true 5D BMGHK tensors) that this kernel does not
    # implement -- stays False.
    SUPPORTS_BMGHK = False
    # False since the internal dispatch switched to the fused dqdkdv kernel:
    # dQ (always) and dV/dK (when M_SPLIT>1, the gfx950 default) are
    # atomic-add, unlike the old split dvdk+dq path's pure
    # register-accumulate-then-store. The D=256/gfx942 fallback (the
    # split path, kept only because dqdkdv structurally can't fit there) IS
    # deterministic, but this flag is a single class-level value covering
    # every shape -- conservatively False for the whole op rather than
    # per-shape-conditional, since AttentionBwOpBase.not_supported_reasons()
    # checks this BEFORE the op knows which internal path a given D will take.
    IS_DETERMINISTIC = False
    VARLEN_LSE_PACKED = (
        True  # matches ck.BwOp's convention (see fmha_bwd_mfma.py's _lse_row)
    )
    # Simple top-left causal (single fixed-length sequence per batch),
    # non-causal `BlockDiagonalMask` varlen, AND per-block top-left causal
    # `BlockDiagonalCausalMask` varlen -- see fmha_bwd_mfma.py's
    # `varlen`/`causal` kwarg docstrings. The `n_row_abs <= m_row_abs` causal
    # term is tile-relative to each block's own batch-local n_start/m_start
    # (never seqstart-global), which already implements per-block causal
    # masking correctly given per-batch tiling -- confirmed via
    # CASES_VARLEN_CAUSAL in test_fmha_bwd_{dvdk,dq}_mfma.py. NOT other
    # causal block-diagonal variants (e.g.
    # `BlockDiagonalCausalFromBottomRightMask`, different alignment
    # semantics) -- separately scoped, not implemented.
    SUPPORTED_ATTN_BIAS_TYPES = (
        type(None),
        LowerTriangularMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
    )
    _TEST_K: List[int] = [32, 64, 96, 128, 256]
    NAME = "flydslB"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(BwOp, cls).not_supported_reasons(d)
        reason = _uniform_row_pitch_reason("query", d.query)
        if reason is not None:
            reasons.append(reason)
        # key/value get the GQA broadcast exception (allow_broadcast_heads) --
        # query never does (no broadcast-Q case exists).
        for name, t in (("key", d.key), ("value", d.value)):
            reason = _uniform_row_pitch_reason(name, t, allow_broadcast_heads=True)
            if reason is not None:
                reasons.append(reason)
        # K and V share a single kv_stride_n kernel arg -- reject if they differ.
        if d.key.stride(1) != d.value.stride(1):
            reasons.append("key and value must share the same row pitch (stride(1))")
        if d.query.shape[-1] % 32 != 0:
            reasons.append("head_dim must be a multiple of 32")
        # GQA: Hq must be a whole multiple of Hkv -- the kernel's
        # heads_per_kv = Hq // Hkv grid-regroup requires this.
        h_kv = _num_kv_heads(d.key)
        if d.query.shape[2] % h_kv != 0:
            reasons.append(
                "query head count must be a multiple of the KV head count (GQA)"
            )
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        # LowerTriangularMask (simple causal) and BlockDiagonalCausalMask
        # (per-block top-left causal) both map to the same causal mask math
        # in this kernel -- mirrors ck.py's _custom_mask_type grouping both
        # under CausalFromTopLeft. BlockDiagonalCausalMask subclasses
        # BlockDiagonalMask, so the isinstance check below (seqstart
        # extraction) already covers it.
        causal = isinstance(
            inp.attn_bias, (LowerTriangularMask, BlockDiagonalCausalMask)
        )
        seqstart_q = seqstart_k = None
        if isinstance(inp.attn_bias, BlockDiagonalMask):
            # Mirrors ck.py's _get_seqlen_info.
            seqstart_q = inp.attn_bias.q_seqinfo.seqstart.to(inp.query.device)
            seqstart_k = inp.attn_bias.k_seqinfo.seqstart.to(inp.query.device)
        dq, dk, dv = cls.OPERATOR(
            inp.query,
            inp.key,
            inp.value,
            ctx.out,
            ctx.lse,
            grad,
            inp.scale_float,
            causal,
            seqstart_q,
            seqstart_k,
        )
        # GQA-via-broadcast (`key`/`value` genuinely Hkv-headed, exposed to
        # the caller as an H-headed stride-0 `.expand()` view -- see
        # _num_kv_heads): the kernel reduces the per-KV-head gradient
        # internally (grid regrouped by KV-head, no atomics -- see the class
        # docstring's GQA note / fmha_bwd_mfma.py's
        # compile_fmha_bwd_dvdk_mfma docstring), so `dk`/`dv` come back
        # Hkv-shaped here, smaller than `inp.key.shape`.
        # `_memory_efficient_attention_backward`
        # (mslk/attention/fmha/__init__.py) unconditionally reshapes every
        # op's returned dk/dv to the ORIGINAL (broadcast, H-shaped)
        # `inp.key.shape`/`inp.value.shape` -- the same contract every other
        # op follows (e.g. flash.py's BwOp always returns dk/dv pre-reshaped
        # to `inp.key.shape`/`inp.value.shape`). Broadcast back up via
        # `.expand()` (stride-0, no extra memory) to satisfy that contract,
        # WITHOUT changing what the kernel itself computes.
        #
        # Dividing by `heads_per_kv` before expanding is required: under the
        # autograd API, PyTorch's own `ExpandBackward` sums the H broadcast
        # copies when reducing back down to the real small-Hkv leaf tensor --
        # returning the already-reduced value undivided would get summed
        # again, overcounting by exactly `heads_per_kv`. Called through the
        # non-autograd `memory_efficient_attention_backward` API directly (no
        # autograd graph, no `ExpandBackward` reduction), this op's contract
        # is "correct after an `ExpandBackward` reduction", matching every
        # other op's broadcast-GQA contract.
        if dk.shape[2] != inp.key.shape[2]:
            heads_per_kv = inp.key.shape[2] // dk.shape[2]
            dk = (dk / heads_per_kv).expand(inp.key.shape)
            dv = (dv / heads_per_kv).expand(inp.value.shape)
        return Gradients(dq=dq, dk=dk, dv=dv)


# ============================ Forward op (flydslF) ============================
# Wraps ``mslk.attention.flydsl.flydsl_flash_attn_func`` as a drop-in for
# ``ck.FwOp``: bf16/f16, head_dim in [1, 256] (non-native dims zero-padded to
# {64,128,256}), causal/non-causal self-attention, additive tensor bias,
# sliding-window, varlen, padded/gappy keys, paged KV, all 18 attn_bias types,
# BMHK and BMGHK. Gradients dispatch to the native BwOp above (or ck.BwOp).

# Causal mask types (LowerTriangularMaskWithTensorBias is causal + tensor bias).
_CAUSAL_BIAS_TYPES = (
    LowerTriangularMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
    LowerTriangularMaskWithTensorBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
)

# Sliding-window (local) mask types carrying a _window_size.
_WINDOW_BIAS_TYPES = (
    LowerTriangularFromBottomRightLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
)

# Varlen-local (windowed) top-left type; drives causal_top_left.
_VARLEN_WINDOW_TOPLEFT_TYPES = (BlockDiagonalCausalLocalAttentionMask,)

# Head dims the kernel builds natively; other dims in [1, 256] are zero-padded up to
# the next value here (padded Q/K add 0 to QK; padded V columns are sliced off O).
_SUPPORTED_HEAD_DIMS = (64, 128, 256)


def _pad_head_dim(d: int) -> int:
    """Smallest natively-buildable head dim >= d (one of _SUPPORTED_HEAD_DIMS)."""
    for hd in _SUPPORTED_HEAD_DIMS:
        if d <= hd:
            return hd
    return d


def _pack_varlen_lse(lse: torch.Tensor, q_seqinfo: Any) -> torch.Tensor:
    """Padded per-batch LSE [B, H, max_seqlen_q] -> packed [1, H, total_q].

    Mirrors ck.FwOp's packed layout so ck.BwOp (VARLEN_LSE_PACKED=True) reads the
    right rows. Padded tail rows (q_row >= seqlen_q) are dropped per batch.
    """
    parts = [
        sl[:, : end - start]
        for sl, (start, end) in zip(lse.unbind(0), q_seqinfo.intervals())
    ]
    return torch.cat(parts, dim=1).unsqueeze(0)


# Varlen (packed cu_seqlens) mask types; q/k/v arrive as [1, total, H, D].
_VARLEN_BIAS_TYPES = (
    BlockDiagonalMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
)

# Paged-KV mask types, driven through the kernel's vLLM paged path (native D 64/128;
# any page_size that is a multiple of 64, remapped to 64-row sub-pages).
_PAGED_BIAS_TYPES = (
    PagedBlockDiagonalPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
)
_PAGED_HEAD_DIMS = (64, 128)
# The generic paged kernel tiles KV at PAGE_SIZE == BLOCK_N == 64; paged-gappy
# reshapes any physical page_size to 64-row sub-pages (see _apply_bmhk).
_KERNEL_PAGE_SIZE = 64

# Paged-gappy: handled by a host gather through block_tables into a tight packing on
# the varlen path (page-size-agnostic, no kernel paged path). Non-causal.
_PAGED_GAPPY_TYPES = (PagedBlockDiagonalGappyKeysMask,)

# Padded/gappy KV: blocks at k_seqinfo.seqstart with valid length k_seqinfo.seqlen,
# gathered into a tight packing on the varlen path. WithOffset causal = bottom-right.
_PADDED_GAPPY_BIAS_TYPES = (
    BlockDiagonalPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
)


def _is_flydsl_available() -> bool:
    try:
        from mslk.flydsl.common import is_flydsl_available

        return is_flydsl_available()
    except Exception:
        return False


@register_operator
class FwOp(AttentionFwOpBase):
    """FlyDSL flash-attention forward (gfx942 generic / gfx950 dualwave)."""

    # Python callable rather than a torch.ops handle (cf. triton_splitk.FwOp).
    OPERATOR = staticmethod(_is_flydsl_available)
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    # Any dim in [1, 256]; non-native dims are padded (see _pad_head_dim).
    SUPPORTED_MAX_K = 256
    SUPPORTED_MIN_K = 1

    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        torch.Tensor,
        LowerTriangularMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularMaskWithTensorBias,
        LowerTriangularFromBottomRightLocalAttentionMask,
        BlockDiagonalMask,
        BlockDiagonalCausalMask,
        BlockDiagonalCausalFromBottomRightMask,
        BlockDiagonalPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalCausalLocalAttentionMask,
        BlockDiagonalCausalLocalAttentionFromBottomRightMask,
        PagedBlockDiagonalPaddedKeysMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalGappyKeysMask,
    )

    # Dropout (dense): reuses CK's philox mask and records [seed, offset] in
    # ctx.rng_state so ck.BwOp reproduces it. See _apply_bmhk.
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    SUPPORTS_BMGHK = True
    SUPPORTS_PARTIAL = False
    # The kernel writes padded per-batch LSE [B, H, max_seqlen_q]; _apply_bmhk
    # repacks varlen LSE to [1, H, total_q] so this matches BwOp/ck.BwOp (True).
    VARLEN_LSE_PACKED = True
    NAME = "flydslF"

    # Match ck.FwOp's tolerances (same MFMA reduced-precision accumulation).
    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 3e-4,
        torch.half: 6e-3,
        torch.bfloat16: 2.8e-2,
    }
    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 2e-5,
        torch.half: 3e-3,
        torch.bfloat16: 2e-2,
    }

    # 96 is non-native, to exercise head-dim padding.
    _TEST_K: List[int] = [64, 96, 128, 256]

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super().not_supported_reasons(d)

        if not _is_flydsl_available():
            reasons.append("FlyDSL unavailable on this device/arch")

        K = d.query.shape[-1]
        Kv = d.value.shape[-1]
        if not (1 <= K <= 256):
            reasons.append(f"query head_dim={K} out of range [1, 256]")
        if Kv != K:
            reasons.append("value head_dim must equal query head_dim")

        # Paged KV kernel fixes PAGE_SIZE 64 and D in {64, 128}; a larger physical
        # page_size (multiple of 64) is remapped to 64-row sub-pages in _apply_bmhk.
        if isinstance(d.attn_bias, _PAGED_BIAS_TYPES):
            if d.attn_bias.page_size % _KERNEL_PAGE_SIZE != 0:
                reasons.append(
                    f"paged page_size={d.attn_bias.page_size} must be a "
                    f"multiple of {_KERNEL_PAGE_SIZE}"
                )
            if K not in _PAGED_HEAD_DIMS:
                reasons.append(
                    f"paged head_dim={K} unsupported (kernel paged: {_PAGED_HEAD_DIMS})"
                )
        # Paged-gappy uses the generic paged kernel (per-row page lookup). D is
        # padded to {64,128,256}; page_size must be a multiple of the kernel's 64.
        if isinstance(d.attn_bias, _PAGED_GAPPY_TYPES):
            if d.attn_bias.page_size % _KERNEL_PAGE_SIZE != 0:
                reasons.append(
                    f"paged-gappy page_size={d.attn_bias.page_size} must be a "
                    f"multiple of {_KERNEL_PAGE_SIZE}"
                )
        # Dense bottom-right causal cross-attention requires Mq <= Mkv; for Mq > Mkv
        # the leading rows are fully masked (undefined softmax).
        elif isinstance(d.attn_bias, LowerTriangularFromBottomRightMask):
            if d.query.shape[1] > d.key.shape[1]:
                reasons.append("LowerTriangularFromBottomRightMask requires Mq <= Mkv")
        # Other dense masks are self-attention only (varlen/paged/gappy handle
        # cross-length via their own paths).
        elif not isinstance(
            d.attn_bias,
            _VARLEN_BIAS_TYPES
            + _PADDED_GAPPY_BIAS_TYPES
            + _PAGED_BIAS_TYPES
            + _PAGED_GAPPY_TYPES,
        ):
            q_len = d.query.shape[1]
            kv_len = d.key.shape[1]
            if q_len != kv_len:
                reasons.append(
                    f"cross-attention Mq={q_len} != Mkv={kv_len} not supported "
                    "(dense self-attention only)"
                )

        # Dropout: dense self-attention only (the in-kernel mask is applied on the
        # generic dense path; varlen/paged/gappy dropout is not wired).
        if d.p != 0.0 and isinstance(
            d.attn_bias,
            _VARLEN_BIAS_TYPES
            + _PADDED_GAPPY_BIAS_TYPES
            + _PAGED_BIAS_TYPES
            + _PAGED_GAPPY_TYPES,
        ):
            reasons.append("dropout only supported on the dense self-attention path")

        # Native paged KV has no backward: the dualwave route emits no LSE and there
        # is no paged backward op, so a gradient request (return_lse on that route)
        # would fail at run time. Decline so grad cases dispatch elsewhere.
        _needs_grad = (
            d.query.requires_grad or d.key.requires_grad or d.value.requires_grad
        )
        if _needs_grad and isinstance(d.attn_bias, _PAGED_BIAS_TYPES):
            reasons.append("native paged KV does not support gradients (no backward)")

        # Non-gappy paged KV hardcodes 1/sqrt(head_dim); a custom scale is silently
        # ignored (only gappy paged folds sm_scale).
        if d.scale is not None and isinstance(d.attn_bias, _PAGED_BIAS_TYPES):
            reasons.append("custom scale is not supported for native paged KV")

        # Q/K/V must be last-dim contiguous and 16-byte aligned for the kernel's
        # 128-bit vector loads (raw buffer-resource loads do not tolerate otherwise).
        for _name, _t in (("query", d.query), ("key", d.key), ("value", d.value)):
            if _t.stride(-1) != 1:
                reasons.append(f"{_name} last dim must be contiguous (stride==1)")
            elif _t.data_ptr() % 16 != 0:
                reasons.append(f"{_name} base pointer must be 16-byte aligned")

        # Tensor bias must match the query dtype: the kernel loads bias bytes as the
        # query dtype, so an fp32 bias would be reinterpreted as bf16/f16.
        _bias_t: Optional[torch.Tensor] = None
        if isinstance(d.attn_bias, torch.Tensor):
            _bias_t = d.attn_bias
        elif isinstance(d.attn_bias, LowerTriangularMaskWithTensorBias):
            _bias_t = d.attn_bias._bias
        if _bias_t is not None and _bias_t.dtype != d.query.dtype:
            reasons.append(
                f"attn_bias dtype {_bias_t.dtype} must match query dtype "
                f"{d.query.dtype}"
            )

        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if type(inp.attn_bias) not in cls.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError("Unsupported attn_bias type")
        if inp.query.ndim == 5:
            return cls._apply_bmghk(inp, needs_gradient)
        if inp.query.ndim != 4:
            raise NotImplementedError("Unsupported number of dimensions")
        return cls._apply_bmhk(inp, needs_gradient)

    # ------------------------------------------------------------------ BMGHK
    @classmethod
    def _apply_bmghk(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        # Flatten the G group axis into H (as ck.FwOp does): expanded 5-D KV
        # (stride[3]==0) -> as_strided drop, otherwise flatten(2, 3).
        [_, _, G, Hq, _] = inp.query.shape
        if inp.key.stride()[3] == 0:
            ks, kst = inp.key.size(), inp.key.stride()
            key = inp.key.as_strided(
                (ks[0], ks[1], ks[2], ks[4]), (kst[0], kst[1], kst[2], kst[4])
            )
            vs, vst = inp.value.size(), inp.value.stride()
            value = inp.value.as_strided(
                (vs[0], vs[1], vs[2], vs[4]), (vst[0], vst[1], vst[2], vst[4])
            )
        else:
            key = inp.key.flatten(2, 3)
            value = inp.value.flatten(2, 3)
        # A 5-D tensor bias [B, G, H, q, kv] flattens (G, H) to match the folded
        # head axis; mask objects pass through unchanged.
        bias_flat = inp.attn_bias
        if isinstance(bias_flat, torch.Tensor) and bias_flat.ndim == 5:
            bias_flat = bias_flat.flatten(1, 2)
        elif isinstance(bias_flat, LowerTriangularMaskWithTensorBias):
            _bt = bias_flat._bias
            if _bt.ndim == 5:
                bias_flat = LowerTriangularMaskWithTensorBias(_bt.flatten(1, 2))
        flat = Inputs(
            query=inp.query.flatten(2, 3),
            key=key,
            value=value,
            attn_bias=bias_flat,
            p=inp.p,
            scale=inp.scale,
            output_dtype=inp.output_dtype,
        )
        out, ctx = cls._apply_bmhk(flat, needs_gradient)
        out = out.unflatten(2, (G, Hq))
        if ctx is not None:
            # Update the existing context in place so dropout rng_state (and any
            # other fields) survive; rebuilding a Context would drop rng_state and
            # break ck.BwOp for 5-D dropout inputs.
            ctx.lse = ctx.lse.unflatten(1, (G, Hq))
            ctx.out = out
        return out, ctx

    # ------------------------------------------------------------------- BMHK
    @classmethod
    def _apply_bmhk(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        from mslk.attention.flydsl import flydsl_flash_attn_func

        bias = inp.attn_bias
        causal = isinstance(bias, _CAUSAL_BIAS_TYPES)

        # Head-dim padding: non-native dims are zero-padded to the next of
        # {64, 128, 256}. Paged KV is never padded (gated to native D upstream).
        D_native = inp.query.shape[-1]
        Dpad = _pad_head_dim(D_native)
        needs_pad = Dpad != D_native

        def _pad_hd(t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(t, (0, Dpad - D_native)) if needs_pad else t

        kw: dict = {"causal": causal, "num_kv_heads": inp.key.shape[-2]}

        # window_left = _window_size (xformers keeps kv > q - window_size).
        if isinstance(bias, _WINDOW_BIAS_TYPES):
            kw["window_left"] = int(bias._window_size)

        if isinstance(bias, torch.Tensor):
            kw["bias"] = bias
        elif isinstance(bias, LowerTriangularMaskWithTensorBias):
            kw["bias"] = bias._bias

        # Dense/varlen fold sm_scale at build time; when padding, pin the native
        # scale so the padded 1/sqrt(Dpad) does not change the result.
        if inp.scale is not None:
            kw["sm_scale"] = float(inp.scale)
        elif needs_pad:
            kw["sm_scale"] = 1.0 / math.sqrt(D_native)

        kw["return_lse"] = needs_gradient

        # Dropout (dense only): Python-only mask via native torch RNG. This is
        # CUDA-graph-safe and advances per replay, and drops the CK philox
        # dependency. NOTE: backward mask regeneration is not implemented yet --
        # training (needs_gradient) with dropout is rejected below until a
        # flydsl dropout backward that reproduces this mask lands.
        if inp.p != 0.0:
            B = inp.query.shape[0]
            Sq = inp.query.shape[1]
            H = inp.query.shape[2]
            Skv = inp.key.shape[1]
            keep = (
                torch.rand(
                    B, H, Sq, Skv, device=inp.query.device, dtype=torch.float32
                )
                >= inp.p
            )
            # uint8 keep-mask (0/1), half the memory of a bf16/f16 mask. The
            # 1/(1-p) inverted-dropout scale is applied to the output in _run
            # (a constant factor, so it factors out of the per-(q,k) sum).
            kw["dropout_mask"] = keep.to(torch.uint8)

        # Shared tail: pad the head dim, call, unpack LSE, slice padded columns off O,
        # restore input shape (slice/reshape are no-ops when a path didn't pad).
        def _run(q, k, v, pad=True):
            if pad:
                q, k, v = _pad_hd(q), _pad_hd(k), _pad_hd(v)
            result = flydsl_flash_attn_func(q, k, v, **kw)
            out, lse = result if needs_gradient else (result, None)
            out = out[..., :D_native].reshape(inp.query.shape)
            if inp.p != 0.0:
                # Inverted-dropout output scale (the uint8 mask carries no scale).
                out = out * (1.0 / (1.0 - inp.p))
            return out, lse

        def _set_varlen_kw(cu_q, cu_kv, max_q, max_kv, cross):
            kw["cu_seqlens_q"] = cu_q
            kw["cu_seqlens_kv"] = cu_kv
            kw["max_seqlen_q"] = max_q
            kw["max_seqlen_kv"] = max_kv
            kw["cross_seqlen"] = cross

        # Pack q/k/v to flat [total, H, D] for the varlen path.
        def _flatten_qkv():
            q = inp.query.reshape(-1, inp.query.shape[-2], inp.query.shape[-1])
            k = inp.key.reshape(-1, inp.key.shape[-2], inp.key.shape[-1])
            v = inp.value.reshape(-1, inp.value.shape[-2], inp.value.shape[-1])
            return q, k, v

        if isinstance(bias, _PAGED_BIAS_TYPES):
            # Paged KV: Q packed [1, total_q, H, D], K/V the physical page cache.
            # Drives the kernel's varlen+paged path. Native-D only, so pad=False.
            H = inp.query.shape[-2]
            Hkv = inp.key.shape[-2]
            Dk = inp.key.shape[-1]
            q = inp.query.reshape(-1, H, inp.query.shape[-1])
            page_size = int(bias.page_size)
            # The paged kernel fixes PAGE_SIZE == 64. For a larger physical page_size
            # (a multiple of 64), reshape the cache into 64-row sub-pages and expand
            # block_table so each logical page p maps to sub-pages [p*sub, p*sub+sub);
            # this preserves the physical layout while giving the kernel 64-row pages.
            sub = page_size // _KERNEL_PAGE_SIZE
            k = inp.key.reshape(-1, _KERNEL_PAGE_SIZE, Hkv, Dk)
            v = inp.value.reshape(-1, _KERNEL_PAGE_SIZE, Hkv, Dk)
            seqlen_k = bias.k_seqinfo.seqlen.to(q.device)
            # cu_seqlens_kv = cumulative per-seq KV lengths; pages via block_table.
            kseq = torch.nn.functional.pad(
                seqlen_k.to(torch.int32).cumsum(0, dtype=torch.int32), (1, 0)
            )
            _set_varlen_kw(
                bias.q_seqinfo.seqstart.to(q.device),
                kseq,
                int(bias.q_seqinfo.max_seqlen),
                int(bias.k_seqinfo.max_seqlen),
                True,
            )
            bt = bias.block_tables.to(torch.int32).to(q.device)
            if sub > 1:
                bt = (
                    bt.unsqueeze(-1) * sub
                    + torch.arange(sub, dtype=torch.int32, device=q.device)
                ).reshape(bt.shape[0], bt.shape[1] * sub)
            kw["block_table"] = bt
            kw["seqlen_k"] = seqlen_k
            kw["kv_cache_layout"] = "linear"
            out, lse = _run(q, k, v, pad=False)
        elif isinstance(bias, _PADDED_GAPPY_BIAS_TYPES):
            # Padded/gappy KV: pass the ORIGINAL K/V store (no repack) plus a per-seq
            # absolute KV start; the kernel gathers each seq in-place. seqstart and
            # seqlen are already device tensors, so no host sync.
            q, k, v = _flatten_qkv()
            seqlen_kv = bias.k_seqinfo.seqlen.to(torch.int32).to(q.device)
            kcum = torch.nn.functional.pad(
                seqlen_kv.cumsum(0, dtype=torch.int32), (1, 0)
            )
            _set_varlen_kw(
                bias.q_seqinfo.seqstart.to(q.device),
                kcum,
                int(bias.q_seqinfo.max_seqlen),
                int(bias.k_seqinfo.max_seqlen),
                True,
            )
            kw["kv_seqstart"] = bias.k_seqinfo.seqstart.to(torch.int32).to(q.device)
            out, lse = _run(q, k, v)
        elif isinstance(bias, _PAGED_GAPPY_TYPES):
            # Paged-gappy KV: pass the physical page cache as-is; the kernel resolves
            # each row's page (per-row, since a non-aligned logical start straddles
            # pages) from block_table + a per-seq logical start. No host gather.
            # k_seqinfo: seqstart_py = per-seq logical start, seqlen_py = logical end.
            H = inp.query.shape[-2]
            Hkv = inp.key.shape[-2]
            Dk = inp.key.shape[-1]
            q = _pad_hd(inp.query.reshape(-1, H, Dk))
            page_size = int(bias.page_size)
            # The generic paged kernel fixes PAGE_SIZE == BLOCK_N == 64. Reshape the
            # physical cache to 64-row sub-pages and expand block_table so each
            # logical page p maps to sub-pages [p*sub, p*sub+sub); this keeps the
            # in-kernel per-row page lookup valid at any physical page_size. Head dim
            # is zero-padded (last axis) like the other paths.
            sub = page_size // _KERNEL_PAGE_SIZE
            k = _pad_hd(inp.key.reshape(-1, _KERNEL_PAGE_SIZE, Hkv, Dk))
            v = _pad_hd(inp.value.reshape(-1, _KERNEL_PAGE_SIZE, Hkv, Dk))
            # sm_scale (native, pinned above when padding) is folded by the generic
            # paged kernel, so no Q pre-scale is needed.
            bt = bias.block_tables.to(torch.int32).to(q.device)
            bt64 = (
                bt.unsqueeze(-1) * sub
                + torch.arange(sub, dtype=torch.int32, device=q.device)
            ).reshape(bt.shape[0], bt.shape[1] * sub)
            # seqstart is [B+1] (per-seq logical start + trailing sentinel); seqlen
            # is [B] logical ends. Per-seq length = end - start.
            B_kv = bias.k_seqinfo.seqlen.shape[0]
            kstart = bias.k_seqinfo.seqstart[:B_kv].to(torch.int32)
            kend = bias.k_seqinfo.seqlen.to(torch.int32)
            kseqlen = (kend - kstart).to(q.device)
            kcum = torch.nn.functional.pad(kseqlen.cumsum(0, dtype=torch.int32), (1, 0))
            kw["cu_seqlens_q"] = bias.q_seqinfo.seqstart.to(q.device)
            kw["cu_seqlens_kv"] = kcum
            kw["max_seqlen_q"] = int(bias.q_seqinfo.max_seqlen)
            kw["max_seqlen_kv"] = int(bias.k_seqinfo.max_seqlen)
            kw["cross_seqlen"] = True
            kw["block_table"] = bt64
            kw["kv_seqstart"] = kstart.to(q.device)
            kw["kv_cache_layout"] = "linear"
            out, lse = _run(q, k, v, pad=False)
        elif isinstance(bias, _VARLEN_BIAS_TYPES):
            # Packed varlen via per-sequence seqstart offsets; per-block q/k lengths
            # may differ (cross_seqlen).
            q, k, v = _flatten_qkv()
            qseq = bias.q_seqinfo.seqstart.to(q.device)
            kseq = bias.k_seqinfo.seqstart.to(q.device)
            max_q = int(bias.q_seqinfo.max_seqlen)
            max_kv = int(bias.k_seqinfo.max_seqlen)
            # cross_seqlen enables the general (per-block q!=k length) path; it is a
            # correctness superset of the equal-length fast path, so err toward True.
            # Decide WITHOUT a device->host sync (torch.equal(.cpu()) breaks CUDA-graph
            # capture): unequal max => cross; identical seqstart object => not cross;
            # otherwise assume cross (rare distinct-but-equal layouts lose only the
            # equal-length optimization, never correctness).
            cross = (max_q != max_kv) or (qseq is not kseq)
            _set_varlen_kw(qseq, kseq, max_q, max_kv, cross)
            # Top-left-aligned masks need causal_top_left (kernel default is
            # bottom-right); the ...FromBottomRight variant keeps the default.
            if isinstance(
                bias, (BlockDiagonalCausalMask,) + _VARLEN_WINDOW_TOPLEFT_TYPES
            ):
                kw["causal_top_left"] = True
            out, lse = _run(q, k, v)
        else:
            # Dense self-attention.
            out, lse = _run(inp.query, inp.key, inp.value)

        # The kernel returns varlen LSE padded per-batch as [B, H, max_seqlen_q].
        # Repack it to the [1, H, total_q] packed layout (VARLEN_LSE_PACKED=True) so
        # ck.BwOp and automatic dispatch see a consistent LSE for varlen gradients.
        if lse is not None and isinstance(
            bias,
            _VARLEN_BIAS_TYPES
            + _PADDED_GAPPY_BIAS_TYPES
            + _PAGED_BIAS_TYPES
            + _PAGED_GAPPY_TYPES,
        ):
            lse = _pack_varlen_lse(lse, bias.q_seqinfo)

        ctx: Optional[Context] = None
        if needs_gradient:
            if inp.p != 0.0:
                # TODO: implement a flydsl dropout backward that regenerates the
                # Python (torch RNG) mask. Until then the mask cannot be
                # reproduced in the backward (ck.BwOp only knows CK philox), so
                # reject training + dropout rather than return wrong gradients.
                raise NotImplementedError(
                    "flydsl dropout backward is not implemented yet for the "
                    "Python-generated mask; dropout is currently forward/"
                    "inference-only (needs_gradient=False)."
                )
            # op_bw=None lets _dispatch_bw pick a compatible backward
            # (VARLEN_LSE_PACKED=True). It currently resolves to ck.BwOp, which
            # matches this forward for bf16 and f16. The native BwOp above is NOT
            # yet wired as the default: the flydsl FwOp+BwOp pair diverges for
            # f16 (never tested together upstream) and needs numerical
            # verification before it can be preferred here.
            ctx = Context(lse=lse, out=out, op_bw=None)
        return out, ctx
