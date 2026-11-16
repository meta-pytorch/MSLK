# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import sys
from typing import Callable, Dict, Tuple, Union

import torch
import triton
import triton.language as tl
from mslk.utils.device import is_gfx950, rocm_version_at_least

from .vararg_kernel import unroll_varargs, VAR_ARGS_ARRAY

# pyre-ignore-all-errors
AUTOTUNER_KEY = [
    "Z",
    "H",
    "G",
    "N_CTX_Q",
    "N_CTX_K",
    "BLOCK_DMODEL",
    "PACKED_PER_VAL",
    "N_GROUPS",
    "BLOCK_N_PER_SPLIT",
    "PAGE_SIZE",
]


@triton.jit
def _tile_ptrs(
    base,
    row_start,
    n_rows,
    row_stride,
    col_start,
    col_stride,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    """Pointers to the ROWS x COLS tile starting at (row_start, col_start) of a
    matrix with n_rows rows, and the mask of its rows which are in bounds.

    The coordinates are int64 and broadcast to the whole tile. Seemingly
    equivalent ways of computing the pointers led to different instruction
    scheduling, which made some decoding configurations up to 25% slower.
    """
    rows = tl.broadcast_to(
        (row_start + tl.zeros((), tl.int64) + tl.arange(0, ROWS))[:, None],
        (ROWS, COLS),
    )
    cols = tl.broadcast_to(
        (col_start + tl.zeros((), tl.int64) + tl.arange(0, COLS))[None, :],
        (ROWS, COLS),
    )
    ptrs = base + rows * row_stride + cols * col_stride
    return ptrs, (rows >= 0) & (rows < n_rows)


@triton.jit
def _fwd_kernel_splitK(  # noqa: C901
    Q,
    K,
    V,
    sm_scale,
    Out_splitK,  # [B, H, split_k, Mq, K]
    LSE_splitk,  # [B, H, split_k, Mq]
    block_tables,
    Seq_len,
    Seq_starts_k,
    Seq_starts_q,
    Seq_starts_q_multiplier,
    additive_bias,
    K_fp8_scale_shift,
    V_fp8_scale_shift,
    q_fp8_scale_shift,
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qk,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kk,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vk,
    stride_osk_z,
    stride_osk_g,
    stride_osk_h,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_lsek_z,
    stride_lsek_g,
    stride_lsek_h,
    stride_lsek_s,
    stride_lsek_m,
    stride_blocktablesz,
    stride_blocktablesl,
    stride_bias_b,
    stride_bias_g,
    stride_bias_h,
    stride_bias_qm,
    stride_bias_km,
    stride_k_fp8_scale_shift_z: tl.constexpr,
    stride_k_fp8_scale_shift_n: tl.constexpr,
    stride_k_fp8_scale_shift_g: tl.constexpr,
    stride_k_fp8_scale_shift_h: tl.constexpr,
    stride_v_fp8_scale_shift_z: tl.constexpr,
    stride_v_fp8_scale_shift_n: tl.constexpr,
    stride_v_fp8_scale_shift_g: tl.constexpr,
    stride_v_fp8_scale_shift_h: tl.constexpr,
    stride_q_fp8_scale_shift_z: tl.constexpr,
    stride_q_fp8_scale_shift_m: tl.constexpr,
    stride_q_fp8_scale_shift_g: tl.constexpr,
    stride_q_fp8_scale_shift_h: tl.constexpr,
    kv_cache_blocks_per_row: tl.constexpr,
    Z: tl.constexpr,
    N_CTX_Q: tl.constexpr,  # The number of queries
    N_CTX_K: tl.constexpr,
    BLOCK_N_PER_SPLIT: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    USE_SEQ_LEN: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr,
    N_GROUPS: tl.constexpr,
    # It's important that BOUNDS_CHECKS_N, BLOCK_M, BLOCK_N come at the end of
    # the argument list, since they are provided by the heuristics/autotune decorator.
    # Otherwise Triton throws IndexError
    BOUNDS_CHECKS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_SPLITK: tl.constexpr,
    SPLIT_K_EARLY_EXIT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    NUM_QUERIES_CAUSAL: tl.constexpr,  # The N_CTX_Q queries are from this many sequence positions
    USE_PAGED_ATTENTION: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    HAS_ADDITIVE_BIAS: tl.constexpr,
    NUM_PROGRAMS_DIM2_CONST: tl.constexpr,
    IS_HIP: tl.constexpr,
    FP8_FNUZ: tl.constexpr,
    QUANTIZE_PV_TO_FP8: tl.constexpr,
    QUANTIZE_QK_TO_FP8: tl.constexpr,
    USE_FP32_SCALES: tl.constexpr,
):
    """This kernel can accept non-quantized or int4-quantized keys/values.
    PACKED_PER_VAL determines the quantization type:
        - PACKED_PER_VAL == 1 means no quantization
        - PACKED_PER_VAL == 8 means 4-bit quantization (8 packed quantized values inside one int32)
    For the quantized case K/V should be int32 tensors.
    Quantization can be row-wise (when N_GROUPS = 1) or group-wise with N_GROUPS = 2, 4, or 8.
    Quantization coefficients are stored at the beginning of the row along the last dimension of K/V
    So K[B, H, M, :] has a form
    [   quant_coef0, quant_coef1, ...|
        group0_quant_value0, group0_quant_value1,... |
        group1_quant_value0, group1_quant_value1,...]
    where each quant_coef is an int32 which should be interpreted as 2 packed float16: scale and offset.

    Note: this kernel needs to be processed by unroll_varargs
    before compilation. That will unroll variables marked with "VAR_ARGS_ARRAY" into lists.
    See how FwOp.apply does it below.

    Set IS_SPLITK=False to indicate the MHA result should be written directly.
    No metadata will be written.
    """
    internal_dtype = (
        tl.float64 if Out_splitK.dtype.element_ty is tl.float64 else tl.float32
    )
    tl.static_assert(
        (PACKED_PER_VAL == 1 and tl.constexpr(K.dtype.element_ty != tl.int32))
        or (
            (PACKED_PER_VAL == 4 or PACKED_PER_VAL == 8)
            and tl.constexpr(K.dtype.element_ty == tl.int32)
        ),
        f"Only int4 and fp8 quantization is supported, K/V should have dtype int32 in "
        f"the quantized case: {PACKED_PER_VAL=} {tl.constexpr(K.dtype)=} {tl.constexpr(K.dtype.element_ty)=}",
    )
    tl.static_assert(
        (((N_GROUPS == 1 or N_GROUPS == 2) or N_GROUPS == 4) or N_GROUPS == 8),
        "Number of quantization groups can be 1 (row-wise quantization), 2, 4, or 8.",
    )
    tl.static_assert(
        N_GROUPS == 1 or K_fp8_scale_shift is None,
        f"Only row-wise fp8 quantization is supported, but got {N_GROUPS=} > 1.",
    )
    FP8_QUANTIZED: tl.constexpr = K_fp8_scale_shift is not None
    # pyrefly: ignore [bad-assignment]
    INT4_QUANTIZED: tl.constexpr = PACKED_PER_VAL > 1 and not FP8_QUANTIZED
    PACKED_D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // PACKED_PER_VAL // N_GROUPS
    D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // N_GROUPS

    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = (off_zhg // (H * G)).to(tl.int64)
    off_hg = off_zhg % (H * G)
    off_h = off_hg // G
    off_g = off_hg % G
    splitk_idx = tl.program_id(2)

    if USE_SEQ_LEN:
        kv_len = tl.load(Seq_len + off_z)
        if SPLIT_K_EARLY_EXIT and kv_len == 0:
            return
    else:
        kv_len = N_CTX_K

    if Seq_starts_k is None:
        start_kv_idx = 0
    else:
        start_kv_idx = tl.load(Seq_starts_k + off_z)
        if USE_SEQ_LEN and PAGE_SIZE > 0:
            # gappy with paged attention stores each "end" instead of each "length"
            # because that's what FA3 needs.
            kv_len -= start_kv_idx

    if Seq_starts_q is None:
        q_len = N_CTX_Q
        queries_use_batch_dim = 1
        off_m = 0
    else:
        queries_use_batch_dim = 0
        off_m = tl.load(Seq_starts_q + off_z) * Seq_starts_q_multiplier
        q_len = tl.load(Seq_starts_q + off_z + 1) * Seq_starts_q_multiplier - off_m
        if q_len == 0:
            return

    k_base = K + off_h * stride_kh + off_g * stride_kg
    v_base = V + off_h * stride_vh + off_g * stride_vg

    if FP8_QUANTIZED:
        k_fp8_scale_shift_base = (
            K_fp8_scale_shift
            + off_h * stride_k_fp8_scale_shift_h
            + off_g * stride_k_fp8_scale_shift_g
        )
        v_fp8_scale_shift_base = (
            V_fp8_scale_shift
            + off_h * stride_v_fp8_scale_shift_h
            + off_g * stride_v_fp8_scale_shift_g
        )
    else:
        k_fp8_scale_shift_base = None
        v_fp8_scale_shift_base = None

    # Boundaries of split-k chunk
    chunk_hi = (splitk_idx + 1) * BLOCK_N_PER_SPLIT
    chunk_lo = splitk_idx * BLOCK_N_PER_SPLIT
    ignore_in_first_block = 0
    if PAGE_SIZE > 0:
        # Page contains several blocks
        BLOCKS_IN_PAGE: tl.constexpr = PAGE_SIZE // BLOCK_N
        # Align boundaries of split-k chunk to block boundaries
        # In the last chunk, shift hi to the right, in the other chunks, shift it to the left
        # TODO: Replace NUM_PROGRAMS_DIM2_CONST with tl.num_programs(2) after
        # the next Triton upgrade.
        is_last_chunk = splitk_idx == NUM_PROGRAMS_DIM2_CONST - 1
        shift = BLOCK_N - 1 if is_last_chunk else 0
        lo = (tl.maximum(chunk_lo, start_kv_idx) // BLOCK_N) * BLOCK_N
        ignore_in_first_block = tl.maximum(0, (start_kv_idx - lo))
        hi = ((chunk_hi + shift) // BLOCK_N) * BLOCK_N
        hi = tl.minimum(hi, kv_len + start_kv_idx)
        block_table = block_tables + stride_blocktablesz * off_z
        # Offset in integer blocks
        logical_block_idx = lo // BLOCK_N
    else:
        lo = chunk_lo
        hi = tl.minimum(chunk_hi, kv_len)
        if Seq_starts_k is not None:
            k_base += start_kv_idx * stride_kn
            v_base += start_kv_idx * stride_vn
            if FP8_QUANTIZED:
                k_fp8_scale_shift_base += start_kv_idx * stride_k_fp8_scale_shift_n
                v_fp8_scale_shift_base += start_kv_idx * stride_v_fp8_scale_shift_n
        else:
            k_base += off_z * stride_kz
            v_base += off_z * stride_vz
            if FP8_QUANTIZED:
                k_fp8_scale_shift_base += off_z * stride_k_fp8_scale_shift_z
                v_fp8_scale_shift_base += off_z * stride_v_fp8_scale_shift_z

    if SPLIT_K_EARLY_EXIT and lo >= hi:
        return

    # Offsets are int64: offsets along the KV sequence can overflow int32 for
    # large caches, and int32 offsets also used more registers.
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_start = start_m * BLOCK_M

    # In the int4 case, the first N_GROUPS elements along the last dimension of K/V
    # contain packed quantization coefficients.
    k_d_offsets = (INT4_QUANTIZED * N_GROUPS + tl.arange(0, PACKED_D_PER_GROUP)).to(
        tl.int64
    ) * stride_kk
    V_D_PER_GROUP: tl.constexpr = (
        D_PER_GROUP if QUANTIZE_PV_TO_FP8 else PACKED_D_PER_GROUP
    )
    v_d_offsets = (INT4_QUANTIZED * N_GROUPS + tl.arange(0, V_D_PER_GROUP)).to(
        tl.int64
    ) * stride_vk

    if HAS_ADDITIVE_BIAS:
        bias_base = (
            additive_bias
            + off_z * stride_bias_b
            + off_g * stride_bias_g
            + off_h * stride_bias_h
        )
        bias_m_offsets = (m_start + offs_m).to(tl.int64) * stride_bias_qm
        bias_m_mask = offs_m[:, None] < N_CTX_Q - m_start

    q_base = (
        Q
        + off_m * stride_qm
        + off_h * stride_qh
        + off_z * stride_qz * queries_use_batch_dim
        + off_g * stride_qg
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Before compilation, this kernel will be processed by unroll_varargs.
    # That turns tensors annotated as the one below into lists of tensors of length N_GROUPS.
    # This is a solution for Triton native lack of support for lists of tensors.
    acc: "VAR_ARGS_ARRAY"  # noqa: F821

    # pyrefly: ignore [unbound-name]
    for i in range(len(acc)):  # noqa: F821
        acc[i] = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=internal_dtype)  # noqa: F821
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    #
    # We declare log2e as a constant with a precisely-specified type to guarantee that
    # triton will use the exact same value in all instances below, rather than sometimes
    # using float32 and sometimes using float64.  For more discussion see:
    # https://github.com/triton-lang/triton/issues/5466
    log2e = tl.full((), 1.44269504, tl.float32)
    qk_scale = sm_scale * log2e
    # load q: it will stay in SRAM throughout
    q: "VAR_ARGS_ARRAY"  # noqa: F821

    if QUANTIZE_QK_TO_FP8:
        # q_scale is per row
        q_scale_ptrs, q_scale_mask = _tile_ptrs(
            q_fp8_scale_shift
            + off_m * stride_q_fp8_scale_shift_m
            + off_h * stride_q_fp8_scale_shift_h
            + off_g * stride_q_fp8_scale_shift_g
            + off_z * stride_q_fp8_scale_shift_z * queries_use_batch_dim,
            m_start,
            q_len,
            stride_q_fp8_scale_shift_m,
            0,
            1,
            BLOCK_M,
            1,
        )
        q_scale = tl.load(q_scale_ptrs, mask=q_scale_mask)

    for i in range(len(acc)):  # noqa: F821
        q_ptrs, q_mask = _tile_ptrs(
            q_base,
            m_start,
            q_len,
            stride_qm,
            i * D_PER_GROUP,
            stride_qk,
            BLOCK_M,
            D_PER_GROUP,
        )
        # pyrefly: ignore [unbound-name]
        q[i] = tl.load(q_ptrs, mask=q_mask)  # noqa: F821

    if IS_CAUSAL or IS_LOCAL:
        # Why does the masking conditon below work as a causal mask?
        # Assuming num_queries <= BLOCK_M:
        # kv_pos = kv_start + range(0, BLOCK_N)
        # q_offset = start_m * BLOCK_M + range(0, BLOCK_M)
        # q_pos = kv_start + kv_len - num_queries + q_offset % num_queries
        # mask = q_pos - kv_pos >= 0
        # So the final masking condition is:
        #   range(0, BLOCK_M) % num_queries - range(0, BLOCK_N) >= num_queries - kv_len

        q_offset = m_start + offs_m
        diag_idx = (q_offset[:, None] % NUM_QUERIES_CAUSAL) - offs_n[None, :]
        diag_idx_shifted = tl.constexpr(diag_idx - NUM_QUERIES_CAUSAL + kv_len)

    if not PAGE_SIZE:
        # Row of K/V where the current block starts. It is carried through the
        # loop rather than computed from start_n, which used more registers.
        kv_row = lo.to(tl.int64)
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        if PAGE_SIZE > 0:
            # Offset in integer blocks from the beginning of the page
            # pyrefly: ignore [unbound-name]
            block_offset_in_page = logical_block_idx % BLOCKS_IN_PAGE
            # Offset in integer pages
            logical_page_idx = logical_block_idx // BLOCKS_IN_PAGE
            physical_page_idx = tl.load(
                # pyrefly: ignore [unbound-name]
                block_table + stride_blocktablesl * logical_page_idx
            ).to(tl.int32)
            kv_row = (
                physical_page_idx * PAGE_SIZE + block_offset_in_page * BLOCK_N
            ).to(tl.int64)
            logical_block_idx += 1
        if not BOUNDS_CHECKS_N:
            kv_rows_end = None
        elif PAGE_SIZE > 0:
            kv_rows_end = kv_row + (hi - start_n)
        else:
            kv_rows_end = hi

        k: "VAR_ARGS_ARRAY"  # noqa: F821
        v: "VAR_ARGS_ARRAY"  # noqa: F821

        if QUANTIZE_PV_TO_FP8:
            v_dtype = tl.float8e4nv
        else:
            v_dtype = Q.dtype.element_ty
        for i in range(len(acc)):  # noqa: F821
            k_group_d_offsets = i * PACKED_D_PER_GROUP * stride_kk + k_d_offsets
            v_group_d_offsets = i * V_D_PER_GROUP * stride_vk + v_d_offsets
            if INT4_QUANTIZED:
                k_scale_shift_base = k_base + i * stride_kk
                v_scale_shift_base = v_base + i * stride_vk
                stride_k_scale_shift_n = stride_kn
                stride_v_scale_shift_n = stride_vn
            elif FP8_QUANTIZED:
                k_scale_shift_base = k_fp8_scale_shift_base
                v_scale_shift_base = v_fp8_scale_shift_base
                stride_k_scale_shift_n = stride_k_fp8_scale_shift_n
                stride_v_scale_shift_n = stride_v_fp8_scale_shift_n
            else:
                k_scale_shift_base = None
                v_scale_shift_base = None
                stride_k_scale_shift_n = 0
                stride_v_scale_shift_n = 0
            # Load and dequantize K/V with appropriate return values based on quantization flags
            result = load_dequantize_k_v_group(  # noqa: F821
                k_base,
                v_base,
                k_scale_shift_base,
                v_scale_shift_base,
                k_group_d_offsets,
                v_group_d_offsets,
                stride_kn,
                stride_vn,
                stride_k_scale_shift_n,
                stride_v_scale_shift_n,
                # pyrefly: ignore [unbound-name]
                kv_row,
                kv_rows_end,
                BLOCK_N,
                PACKED_PER_VAL,
                FP8_QUANTIZED,
                Q.dtype.element_ty,
                # pyrefly: ignore [bad-argument-type]
                v_dtype,
                IS_HIP,
                FP8_FNUZ,
                QUANTIZE_PV_TO_FP8,
                QUANTIZE_QK_TO_FP8,
                USE_FP32_SCALES,
            )

            # Unpack results based on quantization configuration
            if QUANTIZE_PV_TO_FP8 and QUANTIZE_QK_TO_FP8:
                # pyrefly: ignore [bad-unpacking, unbound-name]
                k[i], v[i], v_scale, k_scale = result  # noqa: F821
            elif QUANTIZE_PV_TO_FP8:
                # pyrefly: ignore [bad-unpacking, unbound-name]
                k[i], v[i], v_scale = result  # noqa: F821
            elif QUANTIZE_QK_TO_FP8:
                # pyrefly: ignore [bad-unpacking, unbound-name]
                k[i], v[i], k_scale = result  # noqa: F821
            else:
                # pyrefly: ignore [bad-unpacking, unbound-name]
                k[i], v[i] = result  # noqa: F821
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for i in range(len(acc)):  # noqa: F821
            # pyrefly: ignore [unbound-name]
            qk += tl.dot(q[i], k[i])  # noqa: F821

        if QUANTIZE_QK_TO_FP8:
            # Reshape k_scale for proper broadcasting with qk
            # k_scale has shape (BLOCK_N,), we need to reshape it to (1, BLOCK_N)
            # for proper broadcasting with qk of shape (BLOCK_M, BLOCK_N)
            # pyrefly: ignore [unbound-name]
            k_scale_reshaped = tl.reshape(k_scale, (1, BLOCK_N))

            # Apply k_scale to qk
            qk = qk * k_scale_reshaped
            # pyrefly: ignore [unbound-name]
            qk = qk * tl.reshape(q_scale, (BLOCK_M, 1))  # noqa: F821

        # Apply qk_scale (scalar)
        qk *= qk_scale

        if start_n == lo and ignore_in_first_block > 0:
            qk = tl.where(offs_n < ignore_in_first_block, float("-inf"), qk)

        if HAS_ADDITIVE_BIAS:
            if BOUNDS_CHECKS_N:
                # pyrefly: ignore [unbound-name]
                bias_mask = bias_m_mask & (offs_n[None, :] < hi - start_n)
            else:
                # pyrefly: ignore [unbound-name]
                bias_mask = bias_m_mask
            loaded_bias = tl.load(
                # pyrefly: ignore [unbound-name]
                bias_base
                + bias_m_offsets[:, None]
                + ((start_n + offs_n).to(tl.int64) * stride_bias_km)[None, :],
                mask=bias_mask,
                other=0.0,
            )
            qk += loaded_bias.to(tl.float32) * log2e

        # TODO: This is slow, and only needed at the last iteration.
        # Maybe we can unroll the last iteration instead?
        if BOUNDS_CHECKS_N:
            qk = tl.where(offs_n < hi - start_n, qk, float("-inf"))
        if IS_CAUSAL:
            # -- apply the causal mask --
            # pyrefly: ignore [unbound-name]
            qk = tl.where(diag_idx_shifted >= start_n - start_kv_idx, qk, float("-inf"))
        if IS_LOCAL:
            # -- apply the local window size mask --
            qk = tl.where(
                # pyrefly: ignore [unbound-name]
                diag_idx_shifted < start_n - start_kv_idx + WINDOW_LEFT + 1,
                qk,
                float("-inf"),
            )
            if not IS_CAUSAL and WINDOW_RIGHT >= 0:
                qk = tl.where(
                    diag_idx_shifted >= start_n - start_kv_idx - WINDOW_RIGHT,
                    qk,
                    float("-inf"),
                )

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if HAS_ADDITIVE_BIAS or (IS_CAUSAL or IS_LOCAL):
            # NOTE: It's possible that an entire block is masked out.
            # if this is the case, `m_i_new=nan` and everything becomes nan
            alpha = tl.where(m_i_new == float("-inf"), 0, alpha)
            p = tl.where(m_i_new[:, None] == float("-inf"), 0, p)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        if not QUANTIZE_PV_TO_FP8:
            p = p.to(v_dtype)
        else:
            # Apply v-scale to P
            # pyrefly: ignore [unbound-name]
            p = p * tl.trans(v_scale)

            # Quantize P to FP8
            MAX_FP8 = 448
            amax = tl.max(p, axis=1)  # rowmax(P)
            p_scale = tl.maximum(amax / MAX_FP8, 1e-9)
            p_scaled = p / p_scale[:, None]
            p_clamped = tl.clamp(p_scaled, 0, MAX_FP8)

            # convert P to FP8
            p = p_clamped.to(v_dtype, fp_downcast_rounding="rtne")

        # -- scale and update acc --
        for i in range(len(acc)):  # noqa: F821
            acc[i] *= alpha[:, None]  # noqa: F821
            if not QUANTIZE_PV_TO_FP8:
                # pyrefly: ignore [unbound-name]
                acc[i] += tl.dot(p, v[i])  # noqa: F821
            else:
                # Re-scale PV using p_scale
                # pyrefly: ignore [unbound-name]
                acc[i] += tl.dot(p, v[i]) * p_scale[:, None]  # noqa: F821
        if not PAGE_SIZE:
            # pyrefly: ignore [unbound-name]
            kv_row += BLOCK_N

    # write back O
    o_base = (
        Out_splitK
        + off_z.to(tl.int64) * stride_osk_z * queries_use_batch_dim
        + off_m * stride_osk_m
        + off_g * stride_osk_g
        + off_h * stride_osk_h
        + splitk_idx * stride_osk_s
    )
    for i in range(len(acc)):  # noqa: F821
        # If for the current batch element there are no tokens in the current split-k chunk (because
        # seqlen is too short), l_i will be 0, so we need to make sure attention is filled with zeros and not NaNs.
        attn_out = tl.where(l_i[:, None] == 0, 0.0, acc[i] / l_i[:, None])  # noqa: F821
        o_ptrs, o_mask = _tile_ptrs(
            o_base,
            m_start,
            q_len,
            stride_osk_m,
            i * D_PER_GROUP,
            stride_osk_k,
            BLOCK_M,
            D_PER_GROUP,
        )
        tl.store(
            o_ptrs,
            attn_out.to(Out_splitK.dtype.element_ty),  # noqa: F821
            mask=o_mask,
        )
    if WRITE_LSE:
        LSE_splitk_ptr = (
            LSE_splitk
            + off_z * stride_lsek_z * queries_use_batch_dim
            + off_m * stride_lsek_m
            + off_g * stride_lsek_g
            + off_h * stride_lsek_h
            + splitk_idx * stride_lsek_s
            + (m_start + offs_m) * stride_lsek_m
        )
        # Can be float64 to improve numerics
        lse_dtype = LSE_splitk.dtype.element_ty
        tl.store(
            LSE_splitk_ptr,
            (tl.math.log2(l_i.to(lse_dtype)) + m_i.to(lse_dtype)) / log2e,
            mask=m_start + offs_m < q_len,
        )


def gen_config(
    block_m: int,
    block_n: int,
    stages: int,
    warps: int,
) -> triton.Config:
    """A more compact way to define a triton.Config, so it fits on one line"""

    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
        },
        num_stages=stages,
        num_warps=warps,
    )


def _get_splitk_kernel(num_groups):
    """
    Kernel _fwd_kernel_splitK needs to be post-processed by unroll_varargs
    to specialize it for a given number of quantization groups N_GROUPS
    before we can apply triton.heuristics and triton.autotune, so we
    don't do them as decorators.
    """

    _fwd_kernel_splitK_unrolled = unroll_varargs(_fwd_kernel_splitK, N=num_groups)
    kernel = triton.heuristics(
        {
            "BOUNDS_CHECKS_N": lambda args: bool(
                (args["BLOCK_N_PER_SPLIT"] % args["BLOCK_N"])
                or (
                    args["BLOCK_N_PER_SPLIT"] > 0
                    and args["N_CTX_K"] % args["BLOCK_N_PER_SPLIT"]
                )
                or args["USE_SEQ_LEN"]
            )
        }
    )(_fwd_kernel_splitK_unrolled)
    return kernel


def early_config_prune(configs, named_args, **kwargs):
    use_paged_attention = kwargs["USE_PAGED_ATTENTION"]
    page_size = kwargs["PAGE_SIZE"]
    if use_paged_attention:
        return list(
            filter(lambda config: page_size % config.kwargs["BLOCK_N"] == 0, configs)
        )
    else:
        return configs


@functools.lru_cache(None)
def autotune_kernel(kernel: Callable):
    BLOCK_M_VALUES = [16, 32, 64, 128]
    BLOCK_N_VALUES = [16, 32, 64, 128]
    STAGES_VALUES = [1, 2] if torch.version.hip else [1, 2, 3]
    WARPS_VALUES = [1, 2, 4, 8]

    TRITON_CONFIGS = [
        gen_config(block_m, block_n, stages, warps)
        for block_m in BLOCK_M_VALUES
        for block_n in BLOCK_N_VALUES
        for stages in STAGES_VALUES
        for warps in WARPS_VALUES
        if block_n >= block_m
    ]

    # HIP graph capture faulted (HSA_INVALID_PACKET / GPU faults) during autotuning of
    # this kernel on gfx950 with ROCm < 7.14; fixed in ROCm 7.14. Disable graph-based
    # autotuning only on that affected stack; gfx942, CUDA, and gfx950 on ROCm >= 7.14
    # keep it.
    graph_autotune_broken = is_gfx950() and not rocm_version_at_least(7, 14)

    kernel = triton.autotune(
        configs=TRITON_CONFIGS,
        key=AUTOTUNER_KEY,
        use_cuda_graph=not graph_autotune_broken,
        prune_configs_by={
            "early_config_prune": early_config_prune,
        },
    )(kernel)
    return kernel


# This object contains forward kernels wrapped into autotuner for different number
# of quantization groups.
_fwd_kernel_splitK_autotune: Dict[int, triton.runtime.Autotuner] = {}
# The loop below:
# - transforms the jitted kernel with unroll_varargs producing a new kernel of each value of num_groups
# - wraps the kernel into triton.heuristics
# - wraps kernel into Triton autotuner. Autotuning itself happens the first time the kernel is called
if sys.version_info >= (3, 9):
    # unroll_varargs requires Python 3.9+
    for num_groups in [1, 2, 4, 8]:
        # pyrefly: ignore [unsupported-operation]
        _fwd_kernel_splitK_autotune[num_groups] = autotune_kernel(
            _get_splitk_kernel(num_groups)
        )

    def get_autotuner_cache(
        num_groups: int,
    ) -> Dict[Tuple[Union[int, str]], triton.Config]:
        """Returns a triton.runtime.autotuner.AutoTuner.cache object, which
        represents mappings from kernel autotune keys (tuples describing kernel inputs)
        to triton.Config
        """
        return _fwd_kernel_splitK_autotune[num_groups].cache

    def set_autotuner_cache(
        cache: Dict[Tuple[Union[int, str]], triton.Config], num_groups: int
    ) -> None:
        _fwd_kernel_splitK_autotune[num_groups].cache = cache


@triton.jit
def _masked_load(ptrs, mask):
    if mask is None:
        return tl.load(ptrs)
    else:
        return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def load_dequantize_k_v_group(
    k_base,
    v_base,
    k_scale_shift_base,
    v_scale_shift_base,
    k_d_offsets,
    v_d_offsets,
    stride_kn,
    stride_vn,
    stride_k_scale_shift_n,
    stride_v_scale_shift_n,
    kv_row,
    kv_rows_end,
    BLOCK_N: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr,
    FP8_QUANTIZED: tl.constexpr,
    q_dtype: tl.constexpr,
    v_dtype: tl.constexpr,  # Q.dtype.element_ty
    IS_HIP: tl.constexpr,
    FP8_FNUZ: tl.constexpr,
    QUANTIZE_PV_TO_FP8: tl.constexpr,
    QUANTIZE_QK_TO_FP8: tl.constexpr,
    USE_FP32_SCALES: tl.constexpr,
):
    """Load K/V for a given block and quantization group. In case of
    int4/fp8-quantized K/V, dequantize them after loading.
    The block covers rows kv_row to kv_row + BLOCK_N of K/V, of which those
    from kv_rows_end are masked out, unless kv_rows_end is None.

    Row offsets are int64, since they may overflow int32 for large KV caches.
    Pointers and masks are computed just before each load: anything computed
    earlier, e.g. by the caller, is held in registers across the loads.

    Returns:
        - k, v: loaded and potentially dequantized tensors
        - v_scale (optional): V scale factor if QUANTIZE_PV_TO_FP8 is True
        - k_scale (optional): K scale factor if QUANTIZE_QK_TO_FP8 is True
    """
    rows = kv_row + tl.arange(0, BLOCK_N)
    k_mask = None if kv_rows_end is None else rows[None, :] < kv_rows_end
    v_mask = None if kv_rows_end is None else rows[:, None] < kv_rows_end
    # -- load k, v --
    k = _masked_load(
        k_base + k_d_offsets[:, None] + (rows * stride_kn)[None, :], k_mask
    )
    v = _masked_load(
        v_base + (rows * stride_vn)[:, None] + v_d_offsets[None, :], v_mask
    )
    if k_scale_shift_base is not None:
        k_scale_shift_ptrs = (
            k_scale_shift_base + (rows * stride_k_scale_shift_n)[None, :]
        )
        v_scale_shift_ptrs = (
            v_scale_shift_base + (rows * stride_v_scale_shift_n)[:, None]
        )

    # Initialize return values
    # v_scale = None
    # k_scale = None

    if FP8_QUANTIZED:
        k, v, v_scale, k_scale = _process_fp8_quantization(
            k,
            v,
            k_scale_shift_ptrs,
            v_scale_shift_ptrs,
            k_mask,
            v_mask,
            PACKED_PER_VAL,
            q_dtype,
            v_dtype,
            IS_HIP,
            FP8_FNUZ,
            QUANTIZE_PV_TO_FP8,
            QUANTIZE_QK_TO_FP8,
            USE_FP32_SCALES,
        )

    elif PACKED_PER_VAL > 1:
        # Int4 quantization.
        k, v = _process_int4_quantization(
            k,
            v,
            k_scale_shift_ptrs,
            v_scale_shift_ptrs,
            k_mask,
            v_mask,
            PACKED_PER_VAL,
            q_dtype,
            IS_HIP,
        )

    # Return appropriate values based on quantization flags
    if QUANTIZE_PV_TO_FP8 and QUANTIZE_QK_TO_FP8:
        # Return both v_scale and k_scale for applying to P and K
        # pyrefly: ignore [unbound-name]
        return k, v, v_scale, k_scale
    elif QUANTIZE_PV_TO_FP8:
        # Return v_scale for applying v_scale to P
        # pyrefly: ignore [unbound-name]
        return k, v, v_scale
    elif QUANTIZE_QK_TO_FP8:
        # Return k_scale for applying k_scale to K
        # pyrefly: ignore [unbound-name]
        return k, v, k_scale
    else:
        return k, v


@triton.jit
def _process_fp8_quantization(
    k,
    v,
    k_scale_shift_ptrs,
    v_scale_shift_ptrs,
    k_mask,
    v_mask,
    PACKED_PER_VAL: tl.constexpr,
    q_dtype: tl.constexpr,
    v_dtype: tl.constexpr,
    IS_HIP: tl.constexpr,
    FP8_FNUZ: tl.constexpr,
    QUANTIZE_PV_TO_FP8: tl.constexpr,
    QUANTIZE_QK_TO_FP8: tl.constexpr,
    USE_FP32_SCALES: tl.constexpr,
):
    """Process FP8 quantization for K and V tensors."""
    # Process V tensor
    v_scale_shift = _masked_load(v_scale_shift_ptrs, v_mask)
    v_scale, v_shift = _extract_scale_shift(v_scale_shift, IS_HIP, USE_FP32_SCALES)
    if not QUANTIZE_PV_TO_FP8:
        v = dequantize(
            v,
            v_scale,
            v_shift if not USE_FP32_SCALES else None,
            PACKED_PER_VAL,
            IS_HIP,
            FP8_FNUZ,
            USE_FP32_SCALES,
        ).to(v_dtype)
    else:
        # Do not dequantize V; V needs to be FP8 for PV.
        tl.static_assert(PACKED_PER_VAL == 4, "Assert: int32 packs four FP8 values")

    # Process K tensor
    k_scale_shift = _masked_load(k_scale_shift_ptrs, k_mask)
    k_scale, k_shift = _extract_scale_shift(k_scale_shift, IS_HIP, USE_FP32_SCALES)
    if IS_HIP:
        if not QUANTIZE_QK_TO_FP8:
            k = dequantize_k_hip(k, k_scale, k_shift, PACKED_PER_VAL, FP8_FNUZ).to(
                q_dtype
            )
        else:
            # For QUANTIZE_QK_TO_FP8, unpack int32 to 8-bit entries and interpret as fp8
            tl.static_assert(PACKED_PER_VAL == 4, "Assert: int32 packs four FP8 values")
    else:
        if not QUANTIZE_QK_TO_FP8:
            k_t = dequantize(
                tl.trans(k),
                tl.trans(k_scale),
                # pyrefly: ignore [bad-argument-type]
                tl.trans(k_shift) if not USE_FP32_SCALES else None,
                PACKED_PER_VAL,
                IS_HIP,
                FP8_FNUZ,
                USE_FP32_SCALES,
            ).to(q_dtype)
            k = tl.trans(k_t)
        else:
            # For QUANTIZE_QK_TO_FP8, unpack int32 to 8-bit entries and interpret as fp8
            tl.static_assert(PACKED_PER_VAL == 4, "Assert: int32 packs four FP8 values")
            k_t = tl.trans(k)
            k_t = _unpack_fp8_tensor(k_t, PACKED_PER_VAL, IS_HIP, FP8_FNUZ)
            k = tl.trans(k_t)

    return k, v, v_scale, k_scale


@triton.jit
def _extract_scale_shift(
    scale_shift, IS_HIP: tl.constexpr, USE_FP32_SCALES: tl.constexpr
):
    """Extract scale and shift values from packed representation."""
    if IS_HIP:
        return cast_uint32_to_float(scale_shift)
    elif USE_FP32_SCALES:
        return scale_shift.to(tl.float32, bitcast=True), 0
    else:
        return cast_uint32_to_half2(scale_shift)


@triton.jit
def _unpack_fp8_tensor(
    x_, PACKED_PER_VAL: tl.constexpr, IS_HIP: tl.constexpr, FP8_FNUZ: tl.constexpr
):
    """Unpack FP8 K/V tensor from int32 packed representation."""
    tl.static_assert(PACKED_PER_VAL == 4, "Assert: int32 packs four FP8 values")

    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    BLOCK_N: tl.constexpr = x_.shape[0]
    # Create bit offsets for unpacking (0, 8, 16, 24 bits)
    offsets = tl.arange(0, PACKED_PER_VAL) * 8

    # Extract 8-bit values by right-shifting and masking
    unpacked_values = x_[:, :, None] >> offsets
    unpacked_values = tl.reshape(
        unpacked_values, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL)
    )

    # Convert to FP8 through bitcast. gfx942 uses e4m3fnuz (float8e4b8); gfx950 and
    # CUDA use OCP e4m3fn (float8e4nv). FP8_FNUZ carries the arch decision.
    fp8_type = tl.float8e4b8 if FP8_FNUZ else tl.float8e4nv
    x_ = unpacked_values.to(tl.uint8).to(fp8_type, bitcast=True)

    return x_


@triton.jit
def _process_int4_quantization(
    k,
    v,
    k_scale_shift_ptrs,
    v_scale_shift_ptrs,
    k_mask,
    v_mask,
    PACKED_PER_VAL: tl.constexpr,
    dtype: tl.constexpr,
    IS_HIP: tl.constexpr,
):
    """Process INT4 quantization for K and V tensors."""
    k_scale_shift = _masked_load(k_scale_shift_ptrs, k_mask)
    v_scale_shift = _masked_load(v_scale_shift_ptrs, v_mask)
    if IS_HIP:
        k_scale, k_shift = cast_uint32_to_float(k_scale_shift)
        v_scale, v_shift = cast_uint32_to_float(v_scale_shift)
        # int4 path never reaches the fp8 branch inside dequantize; FP8_FNUZ is unused.
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL, IS_HIP, False).to(dtype)
        k = dequantize_k_hip(k, k_scale, k_shift, PACKED_PER_VAL, False).to(dtype)
    else:
        k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
        v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL, IS_HIP, False).to(dtype)
        k_t = dequantize(
            tl.trans(k),
            tl.trans(k_scale),
            tl.trans(k_shift),
            PACKED_PER_VAL,
            IS_HIP,
            False,
        ).to(dtype)
        k = tl.trans(k_t)

    return k, v


@triton.jit
def cast_uint64_to_float2(scale_shift):
    """Using FP32 scales, so only extract one fp32 from the packed int64"""
    scale = scale_shift & 0xFFFFFFFF
    scale = scale.to(tl.uint32).to(tl.float32, bitcast=True)
    return scale, 0


@triton.jit
def cast_uint32_to_half2(scale_shift):
    """Extract two float16 packed into one int32"""
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True)
    return scale, shift


@triton.jit
def cast_uint32_to_float(scale_shift):
    """Extract two float16 packed into one int32 as float32"""
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return scale, shift


@triton.jit
def dequantize_k_hip(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr,
    FP8_FNUZ: tl.constexpr,
):
    """PACKED_PER_VAL is the number of values packed into each element x_.
    For example, for int4 quantization and x_ of type int32, PACKED_PER_VAL is 8.
    """
    # x_ : (BLOCK_N, D // PACKED_PER_VAL)
    # scale: (BLOCK_N, 1)
    # offsets: (PACKED_PER_VAL,)
    BLOCK_N: tl.constexpr = x_.shape[1]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[0]
    offsets = tl.arange(0, PACKED_PER_VAL) * (32 // PACKED_PER_VAL)
    quant_offset = (
        x_[:, None, :] >> offsets[:, None]
    )  # (BLOCK_N, D // PACKED_PER_VAL, PACKED_PER_VAL)

    quant_offset = tl.reshape(
        quant_offset, (BLOCK_DMODEL_PACKED * PACKED_PER_VAL, BLOCK_N)
    )

    if PACKED_PER_VAL == 4:
        # FP8 quantization. gfx942 -> e4m3fnuz (float8e4b8); gfx950/CUDA -> e4m3fn.
        fp8_type = tl.float8e4b8 if FP8_FNUZ else tl.float8e4nv
        dequant = (
            quant_offset.to(tl.uint8).to(fp8_type, bitcast=True).to(scale.dtype) * scale
            + shift
        )
    else:
        # Int4 quantization.
        # Trick - instead of converting int4 to float16 we view it as float16
        # and then multiply by 32768 * 512 == 2**24
        quant_offset = (
            (quant_offset & 0xF)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        quant_offset = quant_offset * 32768.0
        scale_512 = scale * 512

        dequant = quant_offset * scale_512 + shift
    return dequant


@triton.jit
def dequantize(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr,
    IS_HIP: tl.constexpr,
    FP8_FNUZ: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    USE_FP32_SCALES: tl.constexpr = False,
):
    """PACKED_PER_VAL is the number of values packed into each element x_.
    For example, for int4 quantization and x_ of type int32, PACKED_PER_VAL is 8.
    """
    # x_ : (BLOCK_N, D // PACKED_PER_VAL)
    # scale: (BLOCK_N, 1)
    # offsets: (PACKED_PER_VAL,)
    BLOCK_N: tl.constexpr = x_.shape[0]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * (32 // PACKED_PER_VAL)
    quant_offset = (
        x_[:, :, None] >> offsets
    )  # (BLOCK_N, D // PACKED_PER_VAL, PACKED_PER_VAL)

    quant_offset = tl.reshape(
        quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL)
    )
    if PACKED_PER_VAL == 4:
        # FP8 quantization. gfx942 -> e4m3fnuz (float8e4b8); gfx950/CUDA -> e4m3fn.
        fp8_type = tl.float8e4b8 if FP8_FNUZ else tl.float8e4nv
        dequant = (
            quant_offset.to(tl.uint8).to(fp8_type, bitcast=True).to(scale.dtype) * scale
        )
        if not USE_FP32_SCALES:
            # Use asymmetric quantization only for FP16 scales
            dequant += shift
    else:
        # Int4 quantization.
        # Trick - instead of converting int4 to float16 we view it as float16
        # and then multiply by 32768 * 512 == 2**24
        if IS_HIP:
            # Do final math in float32 to avoid casting to bf16 on MI300. There
            # no direct instructions for this so its less performant on this workload.
            quant_offset = (
                (quant_offset & 0xF)
                .to(tl.uint16)
                .to(tl.float16, bitcast=True)
                .to(tl.float32)
            )
            quant_offset = quant_offset * 32768.0
        else:
            quant_offset = (
                (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
            )
            quant_offset = (quant_offset * 32768.0).to(tl.float16)
        scale_512 = scale * 512

        dequant = quant_offset * scale_512
        if not USE_FP32_SCALES:
            # Use asymmetric quantization only for FP16 scales
            dequant += shift
    return dequant
