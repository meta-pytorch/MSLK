# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe
# pyre-ignore-all-errors[29]

import logging

import pytest
import torch

try:
    from mtia.re.re_unittest_lib import init_mtia_device  # type: ignore

    init_mtia_device()
except ImportError:
    # Failed to load MTIA libraries, so just keep going without MTIA devices
    pass

from mslk.attention import fmha
from mslk.attention.fmha import ALL_BW_OPS, ALL_FW_OPS
from mslk.attention.fmha.attn_bias_utils import create_attn_bias
from mslk.attention.fmha.unbind import unbind

from .case_generation import (
    _generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    create_tensors,
    get_supported_attn_bias_types,
    make_id,
)
from .utils import (
    _filter_unsupported_ops,
    assert_allclose,
    cuda_only,
    cuda_or_mtia_only,
    nanify_oob_seqlen,
    ref_attention_for_test,
    UNSUPPORTED_OP_PASSES,
)

logger = logging.getLogger(__file__)

ALL_FW_OPS = _filter_unsupported_ops(ALL_FW_OPS)
ALL_BW_OPS = _filter_unsupported_ops(ALL_BW_OPS)

parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv = pytest.mark.parametrize(
    "opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_FW_OPS),
)


def parametrize_test_forward():
    values_ids = _generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_FW_OPS)
    out = []
    out_ids = []
    for value, id in zip(values_ids["argvalues"], values_ids["ids"]):
        out.append(value + (False, "BMHK"))
        out_ids.append(id + "--BMHK")
        (
            op,
            device,
            dtype,
            bias_type,
            batch_size,
            q_len,
            kv_len,
            h,
            k,
            kv,
        ) = value
        do_BMK = bias_type in (
            fmha.attn_bias.LowerTriangularMask,
            torch.Tensor,
            type(None),
        )
        # packed doesn't make sense with paged attention,
        # since q has different shape than k/v.
        # packed incompatible with `k != kv` or `q_len != kv_len`
        do_packed = not issubclass(
            bias_type,
            (
                fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
                fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
            ),
        ) and (k == kv and q_len == kv_len)

        if do_BMK:
            out.append(value + (False, "BMK"))
            out_ids.append(id + "--BMK")

        if do_packed:
            out.append(value + (True, "BMHK"))
            out_ids.append(id + "-packed-BMHK")
            if do_BMK:
                out.append(value + (True, "BMK"))
                out_ids.append(id + "-packed-BMK")
    return {
        "argvalues": out,
        "ids": out_ids,
    }


@pytest.mark.parametrize(
    "opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_packed_fmt", **parametrize_test_forward()
)
def test_forward(opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_packed_fmt, **kwargs):
    (
        op,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
        packed,
        fmt,
    ) = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_packed_fmt

    if op is fmha.ck.FwOp:
        if (k > 256 or kv > 256) and issubclass(
            bias_type,
            (
                fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
                fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
            ),
        ):
            pytest.skip("ck.FwOp hdim-512 is not supported when Paged-KVCache is used!")

    try:
        query, key, value, attn_bias = create_tensors(
            *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_packed_fmt[:-2],
            fmt="BMHK" if packed else fmt,
            **kwargs,
        )
    except pytest.skip.Exception as e:
        if UNSUPPORTED_OP_PASSES:
            logger.warning(
                f"Skipping {opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_packed_fmt}: {e}"
            )
            return
        raise
    if attn_bias is not None:
        assert type(attn_bias.to(query.device)) is type(attn_bias)

    if packed:
        c = torch.stack([query, key, value], 2)
        if fmt == "BMK":
            # bm3hk -> 3bhmk -> 3Bmk
            c = c.permute(2, 0, 3, 1, 4).view([3, -1, q_len, k])
            query, key, value = c[0], c[1], c[2]
            # Re-create bias in the right format
            attn_bias = create_attn_bias(
                bias_type=bias_type,
                batch_size=batch_size,
                num_heads=h,
                num_heads_groups=1,
                q_len=q_len,
                kv_len=kv_len,
                device=device,
                dtype=dtype,
                requires_grad=False,
                fmt=fmt,
                op=op,
            )
        elif fmt == "BMHK":
            # bm3hk -> 3 x bmhk
            query, key, value = unbind(c, 2)
        else:
            raise AssertionError(f"Unsupport fmt {fmt} with packing")
        assert not query.is_contiguous()

    out = fmha.memory_efficient_attention_forward(query, key, value, attn_bias, op=op)
    assert not out.isnan().any(), ("Output has NaNs", attn_bias)
    out2 = fmha.memory_efficient_attention_forward(
        nanify_oob_seqlen(query),
        nanify_oob_seqlen(key),
        nanify_oob_seqlen(value),
        attn_bias,
        op=op,
    )
    assert not out2.isnan().any(), "Output has NaNs - most likely reading out-of-bounds"
    assert torch.allclose(out, out2, atol=0.0, rtol=0.0), (
        "Non-deterministic behavior",
        attn_bias,
    )

    ref = ref_attention_for_test(query, key, value, attn_bias)
    assert out.shape == ref.shape, out.shape
    assert_allclose(
        out.float(),
        ref,
        atol=op.ERROR_ATOL[dtype],
        rtol=op.ERROR_RTOL.get(dtype, 1e-5),
    )


def nqueries_for_test_forward_gqa(biasT):
    if issubclass(
        biasT,
        (fmha.attn_bias.LowerTriangularMask, fmha.attn_bias.BlockDiagonalCausalMask),
    ):
        # avoid undefined upper left
        return [512]
    return [1, 512]


@cuda_or_mtia_only
@pytest.mark.parametrize(
    "opFW_biasT_Mq",
    [
        (op, biasT, Mq)
        for op in ALL_FW_OPS
        for biasT in get_supported_attn_bias_types(op)
        if op.SUPPORTS_BMGHK
        for Mq in nqueries_for_test_forward_gqa(biasT)
    ],
    ids=lambda o: f"{o[0].NAME}-{o[1].__name__}-{o[2]}" if isinstance(o, tuple) else "",
)
def test_forward_gqa(opFW_biasT_Mq):
    opFW, biasT, Mq = opFW_biasT_Mq
    device = torch._C._get_accelerator().type

    B_Mq_Mkv_H_K_Kv = (3, Mq, 512, 16, 128, 128)
    test_forward(
        (
            opFW,
            device,
            torch.float16,
            biasT,
            *B_Mq_Mkv_H_K_Kv,
            False,
            "BMGHK",
        ),
        g=2,
    )


shapes_triton_splitk = [
    (1, 8, 2**16, 1, 128, 128),
    (1, 4, 2**16, 1, 128, 128),
    (1, 16, 2**16, 1, 128, 128),
    (1, 16, 2**16, 1, 32, 32),
    (1, 8, 1025, 1, 128, 128),
    (2, 8, 4096, 1, 128, 128),
    (10, 8, 2**16, 1, 128, 128),
    (10, 15, 2**16, 1, 128, 128),
    (1, 3, 2**16, 1, 128, 128),
    (1, 3, 2**16 - 10, 1, 128, 128),
    (2, 3, 73, 1, 128, 128),
    (2, 7, 7328, 1, 128, 128),
    (2, 7, 7328, 1, 120, 120),
    (2, 7, 63, 1, 120, 120),
]
op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_splitk = [
    (fmha.triton_splitk.FwOp, "cuda", torch.float16, type(None), *s)
    for s in shapes_triton_splitk
] + [
    (fmha.triton_splitk.FwOp, "cuda", torch.bfloat16, type(None), *s)
    for s in shapes_triton_splitk
]


@pytest.mark.parametrize(
    "opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_splitk,
    ids=[make_id(*c) for c in op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_splitk],
)
@cuda_only
def test_forward_splitk(
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    packed=False,
    fmt="BMHK",
):
    test_forward(opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv + (packed, fmt))


@cuda_only
def test_forward_triton_split_k_autotune():
    """
    Generally we disable autotuning in tests for performance reasons,
    add just one test with autotuning.
    """

    class SplitKAutotune(fmha.triton_splitk.FwOp):
        AUTOTUNE = True

    opFW = SplitKAutotune
    biasT = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask
    B_Mq_Mkv_H_K_Kv = (1, 1, 2048, 8, 128, 128)
    test_forward(
        (
            opFW,
            "cuda",
            torch.bfloat16,
            biasT,
            *B_Mq_Mkv_H_K_Kv,
            False,
            "BMHK",
        ),
    )
