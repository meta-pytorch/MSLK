# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Triton MLA (Multi-head Latent Attention) kernels for ROCm.

DeepSeek-V3 style MLA with weight absorption: after absorbing K/V projection
matrices into Q and output projections, the attention reduces to MQA over a
single latent KV head with qk_head_dim=576 and v_head_dim=512.

Two kernel pipelines:
  - **Decode** (qlen=1): split-K with grouped-head tiling
    Stage 1: _fwd_grouped_kernel_stage1  — per-(batch, head-block, kv-split) partial
    Stage 2: _fwd_kernel_stage2  — log-sum-exp reduce across split-K
  - **Prefill** (qlen>1): single-stage flash attention with causal masking

Decode kernels ported from SGLang (sgl-project/sglang):
  - python/sglang/srt/layers/attention/triton_ops/decode_attention.py
  - Weight absorption variant (HAS_MLA=True)

Prefill kernel adapted from:
  - ROCm/aimodels agent/kernel/mla_prefill_triton.py
  - AOTriton tritonsrc/fwd_kernel.py (XCD remapping)

Hardware target: AMD MI300X (gfx942) and MI350X (gfx950).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# MLA constants (DeepSeek-V3 architecture, post weight-absorption)
# ---------------------------------------------------------------------------

NUM_HEADS: int = 128
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 512
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM: int = KV_LORA_RANK  # 512


# ---------------------------------------------------------------------------
# XCD-aware PID remapping (from AOTriton, for MI300X/MI350X 8 XCDs)
# ---------------------------------------------------------------------------


@triton.jit
def _remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    """Remap program IDs to distribute work across XCDs evenly."""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        new_pid = xcd * pids_per_xcd + local_pid
    else:
        new_pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return new_pid


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


# ===================================================================
#  DECODE KERNELS (split-K grouped-head pipeline)
#  Ported from SGLang: sgl-project/sglang
#  python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# ===================================================================

_MIN_BLOCK_KV = 32
_is_hip = True


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    block_tables,
    seqused_k,
    Att_Out,
    Att_Lse,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_bt_b,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    page_size: tl.constexpr,
    FUSED_OUTPUT: tl.constexpr = False,
):
    """Stage 1: per-(batch, head-block, kv-split) partial attention.

    Ported from SGLang _fwd_grouped_kernel_stage1 with HAS_MLA=True.
    Grid: (batch, head_groups, MAX_KV_SPLITS).
    Weight absorption: V = transpose(K[:BLOCK_DMODEL]).
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_seq_len = tl.load(seqused_k + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, MAX_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    base_offs_k = offs_d[:, None]
    if BLOCK_DPE > 0:
        base_offs_kpe = offs_dpe[:, None]

    bt_ptr = block_tables + cur_batch * stride_bt_b

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        q_k = q.to(K_Buffer.dtype.element_ty)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe,
                mask=(mask_h[:, None]) & (mask_dpe[None, :]),
                other=0.0,
            )
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            block_idx = offs_n // page_size
            slot_idx = offs_n % page_size
            page_id = tl.load(
                bt_ptr + block_idx,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = page_id * page_size + slot_idx

            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q_k, k)
            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                qk,
                float("-inf"),
            )

            # MLA weight absorption: V = transpose(K[:BLOCK_DMODEL])
            v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        if FUSED_OUTPUT:
            offs_out = (
                cur_batch * stride_mid_ob
                + cur_head[:, None] * stride_mid_oh
                + offs_dv[None, :]
            )
            tl.store(
                Att_Out + offs_out,
                (acc / e_sum[:, None]).to(Att_Out.dtype.element_ty),
                mask=(mask_h[:, None]) & (mask_dv[None, :]),
            )
        else:
            offs_mid_o = (
                cur_batch * stride_mid_ob
                + cur_head[:, None] * stride_mid_oh
                + split_kv_id * stride_mid_os
                + offs_dv[None, :]
            )
            tl.store(
                Att_Out + offs_mid_o,
                acc / e_sum[:, None],
                mask=(mask_h[:, None]) & (mask_dv[None, :]),
            )

            offs_mid_o_1 = (
                cur_batch * stride_mid_ob
                + cur_head * stride_mid_oh
                + split_kv_id * stride_mid_os
            ) // Lv
            tl.store(
                Att_Lse + offs_mid_o_1,
                e_max + tl.log(e_sum),
                mask=mask_h,
            )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    v_scale,
    seqused_k,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """Stage 2: log-sum-exp reduce across split-K partials.

    Ported from SGLang _fwd_kernel_stage2.
    Grid: (batch, head_num).
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(seqused_k + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, MAX_KV_SPLITS)

    for split_kv_id in tl.range(0, MAX_KV_SPLITS, num_stages=2):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os,
                mask=mask_d,
                other=0.0,
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum * v_scale,
        mask=mask_d,
    )


# ===================================================================
#  PREFILL KERNEL (single-stage flash attention)
# ===================================================================


@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    """Binary-search the sequence index owning the q_block at target_idx."""
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _find_seq_idx_token(
    cu_seqlens_ptr,
    token_idx,
    num_seqs,
):
    """Binary-search for the sequence owning a given global token index."""
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(cu_seqlens_ptr + mid + 1)
        if val <= token_idx:
            left = mid + 1
        else:
            right = mid
    return left


@triton.jit
def _mla_prefill_fwd(
    output_ptr,
    query_ptr,
    kv_buffer_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_tables_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    stride_kv_buf_flat: tl.int64,
    stride_kv_buffer_3: tl.constexpr,
    query_start_len_ptr,
    num_seqs: tl.int32,
    page_size: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    """Single-stage MLA prefill with batched-Q KV reuse.

    Grid: (num_kv_heads, cdiv(total_tokens, BLOCK_Q), num_head_groups).
    Each program processes BLOCK_Q query tokens × BLOCK_H heads.
    KV tiles are loaded ONCE per iteration and reused across all
    BLOCK_Q query tokens, reducing KV bandwidth by BLOCK_Q×.

    Uses BF16 partial accumulators to fit within MI300X 64KB LDS
    when BLOCK_Q > 1.
    """
    tl.program_id(0)  # kv_head_idx (always 0 for MLA single KV head)
    q_block_idx = tl.program_id(1)
    head_group_idx = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    qk_factor: tl.float32 = scale * RCP_LN2

    q_token_global_start = q_block_idx * BLOCK_Q

    offs_h = head_group_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_query_heads
    offs_lora_rank = tl.arange(0, KV_LORA_RANK)
    offs_rope_head_dim = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_t = tl.arange(0, TILE_SIZE)

    seq_idx = _find_seq_idx_token(query_start_len_ptr, q_token_global_start, num_seqs)
    cur_batch_in_all_start = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop - cur_batch_in_all_start

    bt_ptr = block_tables_ptr + seq_idx * block_tables_stride
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    for qi in range(BLOCK_Q):
        q_tok = q_token_global_start + qi
        q_loc = q_tok - cur_batch_in_all_start

        if q_loc >= 0 and q_loc < cur_batch_query_len:
            query_offset = q_tok * query_stride_0 + offs_h[:, None] * query_stride_1

            Q_lora = tl.load(
                query_ptr + query_offset + offs_lora_rank[None, :],
                mask=mask_h[:, None],
                other=0.0,
            )
            Q_rope = tl.load(
                query_ptr + query_offset + (KV_LORA_RANK + offs_rope_head_dim)[None, :],
                mask=mask_h[:, None],
                other=0.0,
            )

            M_val = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
            L_val = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
            acc = tl.zeros([BLOCK_H, KV_LORA_RANK], dtype=tl.float32)

            causal_limit = context_len + q_loc + 1
            causal_limit = tl.minimum(causal_limit, seq_len)
            num_tiles = _cdiv_fn(causal_limit, TILE_SIZE)

            for j in range(0, num_tiles):
                seq_offset = j * TILE_SIZE + offs_t
                block_idx_kv = seq_offset // page_size
                slot_idx = seq_offset % page_size
                page_id = tl.load(
                    bt_ptr + block_idx_kv,
                    mask=seq_offset < causal_limit,
                    other=0,
                )
                kv_flat_idx = page_id * page_size + slot_idx

                kv_lora_offset = (
                    kv_flat_idx[:, None] * stride_kv_buf_flat
                    + offs_lora_rank[None, :] * stride_kv_buffer_3
                )
                k_rope_offset = (
                    kv_flat_idx[None, :] * stride_kv_buf_flat
                    + (KV_LORA_RANK + offs_rope_head_dim)[:, None] * stride_kv_buffer_3
                )

                tok_mask = seq_offset < causal_limit

                KV_lora = tl.load(
                    kv_buffer_ptr + kv_lora_offset,
                    mask=tok_mask[:, None],
                    other=0.0,
                )
                K_rope = tl.load(
                    kv_buffer_ptr + k_rope_offset,
                    mask=tok_mask[None, :],
                    other=0.0,
                )

                seq_mask = seq_offset[None, :] < causal_limit

                S_lora = tl.dot(Q_lora, KV_lora.trans(1, 0).to(Q_lora.dtype))
                S_rope = tl.dot(Q_rope, K_rope.to(Q_lora.dtype))
                S = qk_factor * (S_lora + S_rope)

                S = tl.where(
                    mask_h[:, None] & seq_mask,
                    S,
                    float("-inf"),
                )

                m_j = tl.maximum(M_val, tl.max(S, axis=1))
                m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

                P = tl.math.exp2(S - m_j[:, None])
                l_j = tl.sum(P, axis=1)

                alpha = tl.math.exp2(M_val - m_j)
                acc = acc * alpha[:, None]

                L_val = L_val * alpha + l_j
                M_val = m_j

                acc += tl.dot(P.to(KV_lora.dtype), KV_lora)

            one_over_L = 1.0 / L_val[:, None]
            acc = acc * one_over_L

            output_offset = (
                q_tok * output_stride_0
                + offs_h[:, None] * output_stride_1
                + offs_lora_rank[None, :]
            )
            tl.store(
                output_ptr + output_offset,
                acc,
                mask=mask_h[:, None],
            )


# ===================================================================
#  LAUNCHER FUNCTIONS (called by triton_mla.py)
# ===================================================================


def mla_decode_forward(
    query: torch.Tensor,
    kv_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    max_kv_splits: int = 8,
    v_scale: float = 1.0,
    logit_cap: float = 0.0,
    block_h: int = 64,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """Launch MLA decode (split-K) pipeline.

    Ported from SGLang _decode_grouped_att_m_fwd with HAS_MLA=True.

    Args:
        query: (batch, num_heads, qk_head_dim) — BF16
        kv_buffer: (num_pages, page_size, num_kv_heads, qk_head_dim) — BF16/FP8
        block_tables: (batch, blocks_per_seq) int32
        cu_seqlens_q: (batch + 1,) int32
        seqused_k: (batch,) int32
        softmax_scale: float
        max_kv_splits: max split-K count
        v_scale: FP8 dequant scale for output (1.0 for BF16)
    """
    batch, head_num, Lk = query.shape[0], query.shape[1], query.shape[2]
    page_size = kv_buffer.shape[1]
    num_kv_heads_actual = kv_buffer.shape[2]
    kv_group_num = head_num // num_kv_heads_actual

    # Flatten pages for token-level indexing
    k_buffer = kv_buffer.view(-1, num_kv_heads_actual, Lk)
    v_buffer = k_buffer  # MLA: V = K[:, :BLOCK_DMODEL]

    # SGLang dimension logic for DeepSeek MLA
    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0

    Lv = BLOCK_DMODEL  # MLA: v_head_dim = kv_lora_rank
    BLOCK_DV = triton.next_power_of_2(Lv)

    # HIP-specific block size (SGLang: BLOCK=16 for Lk>=576 on HIP)
    BLOCK_N = 16 if _is_hip and Lk >= 576 else 32

    BLOCK_H = block_h
    MAX_KV_SPLITS = max_kv_splits

    q = query.view(batch, head_num, Lk)
    out = torch.empty(batch, head_num, Lv, dtype=torch.bfloat16, device=query.device)

    fused = MAX_KV_SPLITS == 1
    num_head_groups = triton.cdiv(head_num, min(BLOCK_H, kv_group_num))

    # HIP-specific kernel args
    extra_kargs = {}
    if _is_hip:
        extra_kargs = {
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        }

    if fused:
        grid = (batch, num_head_groups)
        _fwd_grouped_kernel_stage1[grid](
            q,
            k_buffer,
            v_buffer,
            float(softmax_scale),
            block_tables,
            seqused_k,
            out,
            out,  # Att_Lse unused in fused mode
            q.stride(0),
            q.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            block_tables.stride(0),
            out.stride(0),
            out.stride(1),
            0,  # stride_mid_os unused
            kv_group_num=kv_group_num,
            q_head_num=head_num,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DPE=BLOCK_DPE,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK_N,
            BLOCK_H=BLOCK_H,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            MAX_KV_SPLITS=1,
            logit_cap=logit_cap,
            Lk=Lk,
            Lv=Lv,
            page_size=page_size,
            FUSED_OUTPUT=True,
            num_warps=num_warps,
            num_stages=num_stages,
            **extra_kargs,
        )
    else:
        att_out = torch.empty(
            batch,
            head_num,
            MAX_KV_SPLITS,
            Lv,
            dtype=torch.float32,
            device=query.device,
        )
        att_lse = torch.empty(
            batch,
            head_num,
            MAX_KV_SPLITS,
            dtype=torch.float32,
            device=query.device,
        )

        grid = (batch, num_head_groups, MAX_KV_SPLITS)
        _fwd_grouped_kernel_stage1[grid](
            q,
            k_buffer,
            v_buffer,
            float(softmax_scale),
            block_tables,
            seqused_k,
            att_out,
            att_lse,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            block_tables.stride(0),
            att_out.stride(0),
            att_out.stride(1),
            att_out.stride(2),
            kv_group_num=kv_group_num,
            q_head_num=head_num,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DPE=BLOCK_DPE,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK_N,
            BLOCK_H=BLOCK_H,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            MAX_KV_SPLITS=MAX_KV_SPLITS,
            logit_cap=logit_cap,
            Lk=Lk,
            Lv=Lv,
            page_size=page_size,
            FUSED_OUTPUT=False,
            num_warps=num_warps,
            num_stages=num_stages,
            **extra_kargs,
        )

        grid_stage2 = (batch, head_num)
        _fwd_kernel_stage2[grid_stage2](
            att_out,
            att_lse,
            out,
            v_scale,
            seqused_k,
            att_out.stride(0),
            att_out.stride(1),
            att_out.stride(2),
            out.stride(0),
            out.stride(1),
            MAX_KV_SPLITS=MAX_KV_SPLITS,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            BLOCK_DV=BLOCK_DV,
            Lv=Lv,
            num_warps=4,
            num_stages=2,
        )

    return out.view(batch, head_num, Lv)


def mla_prefill_forward(
    query: torch.Tensor,
    kv_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    page_size: int = 16,
    tile_size: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
    block_h: int = 64,
    block_q: int = 4,
) -> torch.Tensor:
    """Launch MLA prefill (single-stage flash attention).

    Args:
        query: (total_tokens, num_heads, qk_head_dim) — FP8 or BF16
        kv_buffer: (num_blocks, page_size, num_kv_heads, qk_head_dim) — FP8 or BF16
        block_tables: (num_seqs, max_blocks_per_seq) int32
        cu_seqlens_q: (num_seqs + 1,) int32
        seqused_k: (num_seqs,) int32
        softmax_scale: float
        page_size: KV cache page size (tokens per page)
        tile_size: KV tokens per inner loop iteration (independent of page_size)
        block_h: heads per program
    """
    total_num_tokens = query.shape[0]
    num_heads = query.shape[1]
    num_kv_heads_actual = kv_buffer.shape[2]
    num_seqs = seqused_k.shape[0]
    kv_lora_rank = query.shape[2] - QK_ROPE_HEAD_DIM
    actual_page_size = kv_buffer.shape[1]

    kv_flat = kv_buffer.view(-1, num_kv_heads_actual, kv_buffer.shape[3])

    num_head_groups = triton.cdiv(num_heads, block_h)

    out = torch.empty(
        total_num_tokens,
        num_heads,
        kv_lora_rank,
        dtype=torch.bfloat16,
        device=query.device,
    )

    num_q_blocks = triton.cdiv(total_num_tokens, block_q)
    _mla_prefill_fwd[(num_kv_heads_actual, num_q_blocks, num_head_groups)](
        output_ptr=out,
        query_ptr=query,
        kv_buffer_ptr=kv_flat,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seqused_k,
        scale=float(softmax_scale),
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads_actual,
        block_tables_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=QK_ROPE_HEAD_DIM,
        stride_kv_buf_flat=kv_flat.stride(0),
        stride_kv_buffer_3=kv_flat.stride(2),
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        page_size=actual_page_size,
        TILE_SIZE=tile_size,
        BLOCK_H=block_h,
        BLOCK_Q=block_q,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out
