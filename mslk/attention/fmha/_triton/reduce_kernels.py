# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl

from .vararg_kernel import VAR_ARGS_ARRAY

# pyre-ignore-all-errors


@triton.jit
def _splitK_reduce(
    Out_splitK,  # [B, G, H, split_k, Mq, K]
    LSE_splitK,  # [B, G, H, split_k, Mq]
    Out,  # [B, H, M, K]
    LSE,  # [B, H, M]
    split_k: tl.constexpr,
    splitK_pow2: tl.constexpr,
    stride_osk_z: tl.constexpr,
    stride_osk_g: tl.constexpr,
    stride_osk_h: tl.constexpr,
    stride_osk_s: tl.constexpr,
    stride_osk_m: tl.constexpr,
    stride_osk_k: tl.constexpr,
    stride_lsek_z: tl.constexpr,
    stride_lsek_g: tl.constexpr,
    stride_lsek_h: tl.constexpr,
    stride_lsek_s: tl.constexpr,
    stride_lsek_m: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_og: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_ok: tl.constexpr,
    stride_lse_z: tl.constexpr,
    stride_lse_g: tl.constexpr,
    stride_lse_h: tl.constexpr,
    stride_lse_m: tl.constexpr,
    head_dim: tl.constexpr,
    head_dim_pow_2: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    # grid = (M, B * G * H, 1)
    off_m = tl.program_id(0).to(tl.int64)
    off_zhg = tl.program_id(1).to(tl.int64)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G

    head_dim_mask = tl.arange(0, head_dim_pow_2) < head_dim

    Out_splitK_ptr = (
        Out_splitK
        + stride_osk_z * off_z
        + stride_osk_g * off_g
        + stride_osk_h * off_h
        + stride_osk_m * off_m
        + tl.arange(0, head_dim_pow_2)[None, :]
        + stride_osk_s * tl.arange(0, splitK_pow2)[:, None]
    )

    LSE_splitK_ptr0 = (
        LSE_splitK
        + stride_lsek_z * off_z
        + stride_lsek_g * off_g
        + stride_lsek_h * off_h
        + stride_lsek_m * off_m
        + stride_lsek_s * tl.arange(0, splitK_pow2)
    )

    if splitK_pow2 > split_k:
        mask_1d = tl.arange(0, splitK_pow2) < split_k
        mask_2d = mask_1d[:, None] & head_dim_mask[None, :]
        lse_splitk = tl.load(LSE_splitK_ptr0, mask=mask_1d, other=float("-inf"))
        lse_max = tl.max(lse_splitk)
        out_splitk = tl.load(
            Out_splitK_ptr, mask=mask_2d, other=0
        )  # (split_k, head_dim_pow_2)
        lse_splitk = tl.load(
            LSE_splitK_ptr0, mask=mask_1d, other=float("-inf")
        )  # (split_k,)
    else:
        lse_splitk = tl.load(LSE_splitK_ptr0)
        lse_max = tl.max(lse_splitk)
        out_splitk = tl.load(Out_splitK_ptr)
        lse_splitk = tl.load(LSE_splitK_ptr0)

    sumexp_normalized_splitk = tl.math.exp2(
        (lse_splitk - lse_max).to(tl.float32) * 1.44269504
    )  # (split_k,)
    sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0)  # scalar
    # Compute numerator
    numerator_normalized = tl.sum(
        out_splitk * sumexp_normalized_splitk[:, None], axis=0
    )
    acc = numerator_normalized / sumexp_normalized
    acc = tl.where(lse_max == float("-inf"), 0.0, acc)

    Out_ptr = (
        Out
        + stride_oz * off_z
        + stride_oh * off_h
        + stride_og * off_g
        + stride_om * off_m
        + tl.arange(0, head_dim_pow_2)
    )
    if acc.dtype is tl.float64 and Out.dtype.element_ty is not tl.float64:
        # must avoid direct cast f64->f16
        acc = acc.to(tl.float32)
    tl.store(Out_ptr, acc, mask=head_dim_mask)

    if WRITE_LSE:
        l_ptrs = (
            LSE
            + off_z * stride_lse_z
            + off_g * stride_lse_g
            + off_h * stride_lse_h
            + off_m * stride_lse_m
        )
        to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
        to_store = tl.where(lse_max == float("-inf"), lse_max, to_store)
        tl.store(l_ptrs, to_store)


@triton.jit
def _splitK_reduce_varargs(
    Out_splitK: "VAR_ARGS_ARRAY",  # list of [B, G, H, Mq, K];
    LSE_splitK: "VAR_ARGS_ARRAY",  # list of [B, G, H, Mq]
    Out,  # [B, G, H, M, K]
    LSE,  # [B, G, H, M]
    stride_osk_z: "VAR_ARGS_ARRAY",
    stride_osk_g: "VAR_ARGS_ARRAY",
    stride_osk_h: "VAR_ARGS_ARRAY",
    stride_osk_m: "VAR_ARGS_ARRAY",
    stride_osk_k: "VAR_ARGS_ARRAY",
    stride_lsek_z: "VAR_ARGS_ARRAY",
    stride_lsek_g: "VAR_ARGS_ARRAY",
    stride_lsek_h: "VAR_ARGS_ARRAY",
    stride_lsek_m: "VAR_ARGS_ARRAY",
    stride_oz,
    stride_og,
    stride_oh,
    stride_om,
    stride_ok,
    stride_lse_z,
    stride_lse_g,
    stride_lse_h,
    stride_lse_m,
    head_dim: tl.constexpr,
    head_dim_pow_2: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    """
    This version of reduce kernel takes attention and LSE of chunks as lists of tensors,
    as opposed to _splitK_reduce, which takes each as a stacked tensor.
    """
    # grid = (M, B * G * H, 1)
    off_m = tl.program_id(0).to(tl.int64)
    off_zhg = tl.program_id(1).to(tl.int64)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G
    head_dim_mask = tl.arange(0, head_dim_pow_2) < head_dim

    out_splitk_offset: "VAR_ARGS_ARRAY"  # noqa
    for i in range(len(Out_splitK)):
        # pyrefly: ignore [unbound-name]
        out_splitk_offset[i] = (  # noqa
            stride_osk_z[i] * off_z  # noqa
            + stride_osk_g[i] * off_g
            + stride_osk_h[i] * off_h
            + stride_osk_m[i] * off_m
            + tl.arange(0, head_dim_pow_2)
        )
    lse_splitk_offset: "VAR_ARGS_ARRAY"  # noqa
    for i in range(len(Out_splitK)):
        # pyrefly: ignore [unbound-name]
        lse_splitk_offset[i] = (  # noqa
            stride_lsek_z[i] * off_z  # noqa
            + stride_lsek_g[i] * off_g
            + stride_lsek_h[i] * off_h
            + stride_lsek_m[i] * off_m
        )

    lse_max = float("-inf")
    for split_k_idx in range(len(Out_splitK)):  # noqa
        # pyrefly: ignore [unbound-name]
        LSE_splitK_ptr = LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx]  # noqa
        lse_splitk = tl.load(LSE_splitK_ptr)
        lse_max = tl.maximum(lse_max, lse_splitk)

    sumexp_normalized = 0.0
    numerator_normalized = tl.zeros([head_dim_pow_2], dtype=tl.float32)

    for split_k_idx in range(len(Out_splitK)):  # noqa
        out_splitk = tl.load(
            # pyrefly: ignore [unbound-name]
            Out_splitK[split_k_idx] + out_splitk_offset[split_k_idx],  # noqa
            mask=head_dim_mask,
        )
        # pyrefly: ignore [unbound-name]
        lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx])  # noqa
        # Compute denominator
        sumexp_normalized_splitk = tl.math.exp2(
            (lse_splitk - lse_max).to(tl.float32) * 1.44269504
        )
        sumexp_normalized += sumexp_normalized_splitk

        # Compute numerator
        numerator_normalized += out_splitk * sumexp_normalized_splitk

    acc = numerator_normalized / sumexp_normalized
    acc = tl.where(lse_max == float("-inf"), 0.0, acc)

    Out_ptr = (
        Out
        + stride_oz * off_z
        + stride_oh * off_h
        + stride_og * off_g
        + stride_om * off_m
        + tl.arange(0, head_dim_pow_2)
    )
    if acc.dtype is tl.float64 and Out.dtype.element_ty is not tl.float64:
        # must avoid direct cast f64->f16
        acc = acc.to(tl.float32)
    tl.store(Out_ptr, acc, mask=head_dim_mask)

    if WRITE_LSE:
        l_ptrs = (
            LSE
            + off_z * stride_lse_z
            + off_g * stride_lse_g
            + off_h * stride_lse_h
            + off_m * stride_lse_m
        )
        to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
        to_store = tl.where(lse_max == float("-inf"), lse_max, to_store)
        tl.store(l_ptrs, to_store)


@triton.jit
def _splitK_reduce_varargs_backward(
    Out_splitK: "VAR_ARGS_ARRAY",  # list of [B, G, H, Mq, K];
    LSE_splitK: "VAR_ARGS_ARRAY",  # list of [B, G, H, Mq]
    Dout_splitK: "VAR_ARGS_ARRAY",  # gradients - same shape as the inputs themselves
    DLSE_splitK: "VAR_ARGS_ARRAY",
    Out,  # [B, G, H, M, K]
    LSE,  # [B, G, H, M]
    DOut,
    DLSE,
    # strides of chunked inputs: attention and LSE
    stride_osk_z: "VAR_ARGS_ARRAY",
    stride_osk_g: "VAR_ARGS_ARRAY",
    stride_osk_h: "VAR_ARGS_ARRAY",
    stride_osk_m: "VAR_ARGS_ARRAY",
    stride_osk_k: "VAR_ARGS_ARRAY",
    stride_lsek_z: "VAR_ARGS_ARRAY",
    stride_lsek_g: "VAR_ARGS_ARRAY",
    stride_lsek_h: "VAR_ARGS_ARRAY",
    stride_lsek_m: "VAR_ARGS_ARRAY",
    # strides of merged outputs: attention and LSE
    stride_oz,
    stride_og,
    stride_oh,
    stride_om,
    stride_ok,
    stride_lse_z,
    stride_lse_g,
    stride_lse_h,
    stride_lse_m,
    # strides of gradients
    stride_doz,
    stride_dog,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_dlse_z,
    stride_dlse_g,
    stride_dlse_h,
    stride_dlse_m,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
):
    """
    Backward for _splitK_reduce_varargs. Similar to forward, it takes
    attention and LSE of chunks as lists of tensors,
    and outputs the corresponding gradients in the same format.
    """

    # grid = (M, B * G * H, 1)
    off_m = tl.program_id(0).to(tl.int64)
    off_zhg = tl.program_id(1).to(tl.int64)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G

    # Compute offsets inside each attention/LSE chunk.
    # Note that each chunk can have different strides, so offsets can also be different.
    out_splitk_offset: "VAR_ARGS_ARRAY"  # noqa
    for i in range(len(Out_splitK)):
        # pyrefly: ignore [unbound-name]
        out_splitk_offset[i] = (  # noqa
            stride_osk_z[i] * off_z
            + stride_osk_g[i] * off_g
            + stride_osk_h[i] * off_h
            + stride_osk_m[i] * off_m
            + tl.arange(0, BLOCK_SIZE)
        )
    lse_splitk_offset: "VAR_ARGS_ARRAY"  # noqa
    for i in range(len(Out_splitK)):
        # pyrefly: ignore [unbound-name]
        lse_splitk_offset[i] = (  # noqa
            stride_lsek_z[i] * off_z
            + stride_lsek_g[i] * off_g
            + stride_lsek_h[i] * off_h
            + stride_lsek_m[i] * off_m
        )

    lse_max = float("-inf")
    for split_k_idx in range(len(Out_splitK)):  # noqa
        # pyrefly: ignore [unbound-name]
        LSE_splitK_ptr = LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx]  # noqa
        lse_splitk = tl.load(LSE_splitK_ptr)
        lse_max = tl.maximum(lse_max, lse_splitk)

    # Load attention and the corresponding gradient
    offset_out = (
        stride_oz * off_z
        + stride_oh * off_h
        + stride_og * off_g
        + stride_om * off_m
        + tl.arange(0, BLOCK_SIZE)
    )
    offset_dout = (
        stride_doz * off_z
        + stride_doh * off_h
        + stride_dog * off_g
        + stride_dom * off_m
        + tl.arange(0, BLOCK_SIZE)
    )
    out = tl.load(Out + offset_out)
    dattn = tl.load(DOut + offset_dout)

    # Load LSE and the corresponding gradient
    offset_lse = (
        stride_lse_z * off_z
        + stride_lse_h * off_h
        + stride_lse_g * off_g
        + stride_lse_m * off_m
    )
    offset_dlse = (
        stride_dlse_z * off_z
        + stride_dlse_h * off_h
        + stride_dlse_g * off_g
        + stride_dlse_m * off_m
    )
    lse = tl.load(LSE + offset_lse)
    dlse = tl.load(DLSE + offset_dlse)

    for split_k_idx in range(len(Out_splitK)):  # noqa
        # Load attention and LSE of chunks
        # pyrefly: ignore [unbound-name]
        out_splitk = tl.load(Out_splitK[split_k_idx] + out_splitk_offset[split_k_idx])  # noqa
        # pyrefly: ignore [unbound-name]
        lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx])  # noqa

        # Pointers to save gradients of attention and LSE of chunks
        dout_splitk_ptr = Dout_splitK[split_k_idx] + out_splitk_offset[split_k_idx]  # noqa
        dlse_splitk_ptr = DLSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx]  # noqa

        # dX/dattn_i = dX/dattn * dattn/dattn_i + dX/dlse * dlse/dattn_i,
        # and dlse/dattn_i == 0
        dattn_dattn_i = tl.exp(lse_splitk - lse_max) / tl.exp(lse - lse_max)
        dX_dattn_i = dattn_dattn_i * dattn
        tl.store(dout_splitk_ptr, dX_dattn_i)

        dattn_dlse_i = (out_splitk - out) * dattn_dattn_i

        # dX/dlse_i = dX/dattn * dattn/dlse_i + dX/dlse * dlse/dlse_i
        dlse_dlse_i = dattn_dattn_i
        dX_dlse_i = dlse_dlse_i * dlse + tl.sum(
            dattn_dlse_i * dattn
        )  # Sum is over the hidden dimension
        tl.store(dlse_splitk_ptr, dX_dlse_i)
