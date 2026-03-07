# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import re
from typing import Optional

import torch
from torch._C import parse_schema


def _flash_attention3_incompatible_reason() -> Optional[str]:
    if not hasattr(torch.ops.flash_attn_3, "fwd") or not hasattr(
        torch.ops.flash_attn_3, "bwd"
    ):
        return (
            "PyTorch has no `flash_attn_3` "
            "- is your Flash-Attention version recent enough?"
        )
    _fwd_schema = torch.ops.flash_attn_3.fwd.default._schema  # type: ignore
    if not _fwd_schema.is_backward_compatible_with(
        parse_schema(
            "flash_attn_3::fwd("
            "Tensor q, Tensor k, Tensor v, "
            "Tensor(k_new!)? k_new=None, "
            "Tensor(v_new!)? v_new=None, "
            "Tensor? q_v=None, Tensor(out!)? out=None, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_k=None, "
            "Tensor? cu_seqlens_k_new=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, "
            "int? max_seqlen_q=None, "
            "int? max_seqlen_k=None, "
            "Tensor? page_table=None, "
            "Tensor? kv_batch_idx=None, "
            "Tensor? leftpad_k=None, "
            "Tensor? rotary_cos=None, "
            "Tensor? rotary_sin=None, "
            "Tensor? seqlens_rotary=None, "
            "Tensor? q_descale=None, "
            "Tensor? k_descale=None, "
            "Tensor? v_descale=None, "
            "float? softmax_scale=None, "
            "bool is_causal=False, "
            "int window_size_left=-1, "
            "int window_size_right=-1, "
            "int attention_chunk=0, "
            "float softcap=0., "
            "bool is_rotary_interleaved=False, "
            "Tensor? scheduler_metadata=None, "
            "int num_splits=0, "
            "bool? pack_gqa=None, "
            "int sm_margin=0) "
            "-> (Tensor(out!), Tensor, Tensor, Tensor)"
        )
    ):
        return "flash_attn_3::fwd operator is not compatible"

    _bwd_schema = torch.ops.flash_attn_3.bwd.default._schema  # type: ignore
    _bwd_expected = parse_schema(
        "flash_attn_3::bwd("
        "Tensor dout, Tensor q, Tensor k, Tensor v, "
        "Tensor out, Tensor softmax_lse, "
        "Tensor(dq!)? dq=None, "
        "Tensor(dk!)? dk=None, "
        "Tensor(dv!)? dv=None, "
        "Tensor? cu_seqlens_q=None, "
        "Tensor? cu_seqlens_k=None, "
        "Tensor? seqused_q=None, "
        "Tensor? seqused_k=None, "
        "int? max_seqlen_q=None, "
        "int? max_seqlen_k=None, "
        "float? softmax_scale=None, "
        "bool is_causal=False, "
        "int window_size_left=-1, "
        "int window_size_right=-1, "
        "float softcap=0., "
        "bool deterministic=False, "
        "int sm_margin=0) "
        "-> (Tensor(dq!), Tensor(dk!), Tensor(dv!), "
        "Tensor, Tensor, Tensor, Tensor, Tensor)"
    )
    _bwd_expected_alt = parse_schema(
        "flash_attn_3::bwd("
        "Tensor dout, Tensor q, Tensor k, Tensor v, "
        "Tensor out, Tensor softmax_lse, "
        "Tensor(dq!)? dq=None, "
        "Tensor(dk!)? dk=None, "
        "Tensor(dv!)? dv=None, "
        "Tensor? cu_seqlens_q=None, "
        "Tensor? cu_seqlens_k=None, "
        "Tensor? seqused_q=None, "
        "Tensor? seqused_k=None, "
        "int? max_seqlen_q=None, "
        "int? max_seqlen_k=None, "
        "float? softmax_scale=None, "
        "bool is_causal=False, "
        "int window_size_left=-1, "
        "int window_size_right=-1, "
        "float softcap=0., "
        "bool deterministic=False, "
        "int sm_margin=0) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
    )
    if not _bwd_schema.is_backward_compatible_with(
        _bwd_expected
    ) and not _bwd_schema.is_backward_compatible_with(_bwd_expected_alt):
        return "flash_attn_3::bwd operator is not compatible"
    return None


def _count_args_from_doc(docstring: str) -> int:
    """Count the number of arguments from a pybind docstring."""
    match = re.search(r"\((.*?)\)", docstring)
    if match:
        args_list = match.group(1).split(",")
        return len(args_list)
    else:
        raise ValueError("No valid argument list found in the docstring.")


def check_ai_codesign_compatibility(c_module: object) -> None:
    """Compatibility check for ai_codesign FAv3 APIs."""
    expected_num_of_args = [
        ("fwd", 33),
        ("bwd", 22),
    ]
    for name, num_of_args in expected_num_of_args:
        num_of_args_from_doc = _count_args_from_doc(getattr(c_module, name).__doc__)
        assert num_of_args_from_doc == num_of_args, (
            f"Found func signature mismatch for {name}. "
            f"Expected {num_of_args}, "
            f"actual: {num_of_args_from_doc} "
            "Please update the version of Flash Attention3."
        )
