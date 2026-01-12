# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe
# pyre-ignore-all-errors[29]

import logging
import random

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
from mslk.attention.fmha.unbind import unbind

from .case_generation import (
    _generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    create_tensors,
    get_bias_grad,
    sample_random_supported_fw,
)
from .utils import (
    _filter_unsupported_ops,
    assert_allclose,
    cuda_or_mtia_only,
    disable_tf32,
    ref_attention_bmhk_for_test,
    ref_attention_for_test,
    use_cpu_ref,
)

logger = logging.getLogger("xformers")

ALL_FW_OPS = _filter_unsupported_ops(ALL_FW_OPS)
ALL_BW_OPS = _filter_unsupported_ops(ALL_BW_OPS)

parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv = pytest.mark.parametrize(
    "opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_BW_OPS),
)


@disable_tf32
@pytest.mark.parametrize("fmt", ["BMK", "BMHK"])
@pytest.mark.parametrize("grad_out_contiguous", [False, True])
@parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
def test_backward(  # noqa: C901
    opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    grad_out_contiguous,
    fmt,
):
    (
        op_bw,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv

    # Big batch sizes can be slow on MTIA, especially older devices because
    # it doesn't have attention fast paths. Testing on very big batch sizes
    # doesn't meaningfully increase our test coverage here, as long as most
    # permutations of parameters are tested on lower batch sizes.
    if device.startswith("mtia") and batch_size >= 11:
        pytest.skip("Skipping big batch test cases on MTIA")

    attn_bias_requires_grad = (
        random.Random(q_len + kv_len * batch_size).randint(0, 1) > 0
    )
    query, key, value, attn_bias = create_tensors(
        *opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        attn_bias_requires_grad=attn_bias_requires_grad,
        fmt=fmt,
    )

    # To understand why we do this, check the comment on the
    # `AttentionBwOpBase` class
    scale = None
    if op_bw.SUPPORTS_CUSTOM_SCALE and query.shape[-1] < 32:
        scale = (1 / 32) ** 0.5
    op_fw = (
        sample_random_supported_fw(
            fmha.Inputs(query=query, key=key, value=value, attn_bias=attn_bias),
            ALL_FW_OPS,
            seed=q_len * kv + kv_len * k,
            op_bw=op_bw,
        )
        if op_bw != fmha.cutlass.BwOp
        else fmha.cutlass.FwOp
    )

    if op_bw == fmha.ck.BwOp:
        op_fw = fmha.ck.FwOp
        if dtype == torch.bfloat16:
            # bfloat16 testing can be enabled by export ENABLE_HIP_FMHA_RTN_BF16_CONVERT=1 when
            # building xformers and get accurate results
            pytest.skip(
                "CK Fmha backward for bfloat16 currently is not very accurate for some cases!"
            )
        if not grad_out_contiguous:
            pytest.skip("CK Fmha does not support non-contiguous layout for grad_out!")
        if k % 2 != 0:
            pytest.skip(
                "CK Fmha currently requires the headdim size of query input be an even value!"
            )

    qkv = None

    if (
        fmt == "BMHK"
        and query.shape[3] == value.shape[3]
        and query.shape[1] == value.shape[1]
    ):
        qkv = torch.stack([query, key, value], 2)
        qkv.requires_grad_(True)
        # bm3hk -> 3 x bmhk
        query, key, value = unbind(qkv, 2)
        assert not query.is_contiguous()

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    if not op_bw.supports(fmha.Inputs(query, key, value, attn_bias)):
        pytest.skip("inputs not supported")

    out = fmha.memory_efficient_attention(
        query, key, value, attn_bias, scale=scale, op=(op_fw, op_bw)
    )

    grad_out = torch.randn_like(out)
    if grad_out_contiguous is False:
        grad_out = torch.tensor([1.0], dtype=query.dtype, device=device)[
            None, None, :
        ].expand_as(out)

    out.backward(grad_out)

    if qkv is None and op_bw == fmha.cutlass.BwOp:
        assert query.stride() == query.grad.stride()

    grads = []
    if qkv is None:
        grads = [query.grad, key.grad, value.grad]
        query.grad = None
        key.grad = None
        value.grad = None
    else:
        grads = [qkv.grad]
        qkv.grad = None
    if attn_bias_requires_grad:
        attn_bias_grad = get_bias_grad(attn_bias, clear=True)
        if attn_bias_grad is not None:
            grads.append(attn_bias_grad)

    if use_cpu_ref(device):
        query = query.detach().cpu()
        key = key.detach().cpu()
        value = value.detach().cpu()
        grad_out = grad_out.detach().cpu()

        if qkv is not None:
            qkv = torch.stack([query, key, value], 2)
            qkv.requires_grad_(True)
            query, key, value = unbind(qkv, 2)

        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        grad_out.requires_grad_(True)

        if isinstance(attn_bias, torch.Tensor):
            attn_bias = attn_bias.cpu()

    ref = ref_attention_for_test(query, key, value, attn_bias, scale=scale)
    ref.backward(grad_out)

    assert_allclose(
        out.float().to(ref.device),
        ref.float(),
        "fw pass",
        atol=op_fw.ERROR_ATOL[dtype],
        rtol=op_fw.ERROR_RTOL[dtype],
    )

    del out
    del grad_out
    del ref

    atol = op_bw.ERROR_ATOL[dtype]
    rtol = op_bw.ERROR_RTOL[dtype]

    # This is a special case without masks where the accumulated numbers become so big that
    # we lose too much precision, especially on bfloat16. For this reason, the default bfloat16
    # tolerance for the backward pass is set to 0.9, but in some cases on MTIA we get up to 1.3,
    # probably due to the fact that the implementation doesn't use the fused kernels yet, which
    # increases the precision loss caused by the accumulation.
    if (
        device.startswith("mtia")
        and issubclass(bias_type, type(None))
        and q_len >= 2**16
    ):
        atol *= 1.6

    grads_ref = []
    grads_name = []
    if qkv is None:
        assert isinstance(query.grad, torch.Tensor)
        assert isinstance(key.grad, torch.Tensor)
        assert isinstance(value.grad, torch.Tensor)
        grads_ref = [query.grad, key.grad, value.grad]
        grads_name = ["query", "key", "value"]
    else:
        assert isinstance(qkv.grad, torch.Tensor)
        grads_ref = [qkv.grad]
        grads_name = ["qkv"]

    if attn_bias_requires_grad:
        attn_bias_grad = get_bias_grad(attn_bias)
        if attn_bias_grad is not None:
            grads_ref.append(attn_bias.grad)
            grads_name.append("bias")

    del query
    del key
    del value
    del qkv

    assert len(grads_ref) == len(grads), (
        "Wrong number of gradients (maybe bias grad didn't backprop?)"
    )
    for name, calc_grad, ref_grad in zip(grads_name, grads, grads_ref):
        assert_allclose(
            calc_grad.to(ref_grad.device),
            ref_grad,
            msg=f"{op_fw.NAME}+{op_bw.NAME}:{name}",
            atol=atol,
            rtol=rtol,
        )


@cuda_or_mtia_only
@pytest.mark.parametrize(
    "opBW",
    [
        fmha.flash.BwOp,
        fmha.ck.BwOp if torch.version.hip else fmha.cutlass.BwOp,
        fmha.cutlass_blackwell.BwOp,
    ],
)
def test_backward_gqa(opBW):
    device = torch._C._get_accelerator().type

    H = 8
    B_Mq_Mkv_H_K_Kv = (3, 512, 512, H, 128, 128)
    dtype = torch.float16
    query, key, value, attn_bias = create_tensors(
        *(opBW, device, dtype, type(None), *B_Mq_Mkv_H_K_Kv),
        attn_bias_requires_grad=False,
        fmt="BMHK",
    )
    op = (fmha.ck.FwOp if torch.version.hip else fmha.cutlass.FwOp, opBW)
    key = key[:, :, :1].expand(-1, -1, H, -1)
    value = value[:, :, :1].expand(-1, -1, H, -1)
    key.requires_grad_(True)
    out = fmha.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
    out.backward(query)
    dk = key.grad
    key.grad = None

    if use_cpu_ref(device):
        query = query.detach().cpu()
        key = key.detach().cpu()
        value = value.detach().cpu()
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

    out_ref = ref_attention_bmhk_for_test(query, key, value, attn_bias=attn_bias)
    out_ref.backward(query)

    assert_allclose(
        out.float().to(out_ref.device),
        out_ref.float(),
        atol=op[0].ERROR_ATOL[dtype],
        rtol=op[0].ERROR_RTOL[dtype],
    )
    assert_allclose(
        dk.float().to(key.grad.device),
        key.grad.float(),
        atol=op[1].ERROR_ATOL[dtype],
        rtol=op[1].ERROR_RTOL[dtype],
    )
