# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Paged-attention decode ops for decoder_bench.

Each op wraps an fmha forward operator (FlyDSL / CK / Triton) and is benchmarked
through the standard ``op.apply`` interface, so all backends share one code path
and input layout. Mirrors the ops-registry pattern of ``bench/gemm/gemm_ops.py``.
"""

import abc

import torch
from mslk.attention.fmha import (
    ck_decoder,
    ck_splitk,
    flydsl_decoder,
    flydsl_splitk,
    triton_splitk,
)
from mslk.attention.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from mslk.attention.fmha.common import Inputs, InputsFp8
from mslk.bench.common.utils import BenchOptions, do_bench
from mslk.utils.triton.fp8_utils import get_fp8_constants

try:
    from mslk.attention.fmha.flydsl.fp8_paged_cache import dense_kv_to_fp8_paged
    from mslk.attention.fmha.flydsl.pa_decode_fp8 import (
        get_recommended_splits,
        KV_COMPUTE_BLOCK,
        pa_decode_ps_launch,
    )
    from mslk.attention.fmha.flydsl.pa_decode_fp8_dispatch import (
        is_fp8_paged_decode_available,
    )
    from mslk.flydsl.common import is_flydsl_available

    FLYDSL_ENABLED = True
except ImportError:
    FLYDSL_ENABLED = False

DENSE_DTYPES = ("bf16", "f16")


decode_op_registry: list["DecodeOpBase"] = []


def register_decode_op(op):
    """Decorator that registers a single instance of a decode op."""
    decode_op_registry.append(op())
    return op


def get_decode_ops() -> list["DecodeOpBase"]:
    """Return all registered decode ops."""
    return decode_op_registry


class DecodeOpBase(metaclass=abc.ABCMeta):
    """A paged-decode operator benchmarked through the fmha forward interface."""

    # fmha forward op class (subclass of AttentionFwOpBase).
    OP = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # dtypes (``--dtype``) this op benchmarks. Dense ops handle bf16/f16; fp8 ops
    # quantize the KV cache and only run under ``--dtype fp8``.
    supported_dtypes: tuple[str, ...] = DENSE_DTYPES

    @property
    def supported(self) -> bool:
        """Whether this op can run on the current device/build."""
        return self.OP is not None and self.OP.is_available()

    def setup(
        self,
        B: int,
        Hq: int,
        Hkv: int,
        kv_seqlen: int,
        D: int,
        dtype: torch.dtype,
    ) -> tuple:
        """Build the decode inputs (Q, K, V, attn_bias) for one shape.

        Canonical BMGHK decode layout (matches ``_test_decoder`` in the test suite):
        ``G = Hkv`` groups, ``Hq // Hkv`` query heads per group. K/V are stored folded
        (one head per group) and broadcast to the query-head count with ``.expand()``,
        i.e. stride-0 — the realistic GQA cache. Q/K/V are returned as plain tensors so
        the shared ``do_bench`` rotating buffer can size and rotate them.
        """
        dev = "cuda"
        Hpg = Hq // Hkv
        shape = (1, B * kv_seqlen, Hkv, Hpg, D)
        q = torch.randn(1, B, Hkv, Hpg, D, dtype=dtype, device=dev)
        k = torch.randn(1, B * kv_seqlen, Hkv, 1, D, dtype=dtype, device=dev).expand(shape)
        v = torch.randn(1, B * kv_seqlen, Hkv, 1, D, dtype=dtype, device=dev).expand(shape)
        attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
            q_seqlen=[1] * B,
            kv_seqlen=[kv_seqlen] * B,
            kv_padding=kv_seqlen,
        )
        attn_bias.k_seqinfo.to(dev)
        attn_bias.q_seqinfo.to(dev)
        return q, k, v, attn_bias

    def compute(self, q, k, v, attn_bias) -> torch.Tensor:
        inp = Inputs(q, k, v, attn_bias=attn_bias, scale=float(q.shape[-1] ** -0.5))
        out, _ = self.OP.apply(inp, False)
        return out

    def benchmark(self, *args, opts: BenchOptions) -> float:
        """Benchmark runtime (ms) of this op."""
        return do_bench(lambda *a: self.compute(*a), args, opts)


class _FlyDSLDecodeOp(DecodeOpBase):
    """FlyDSL ops additionally require the flydsl package + a supported arch."""

    @property
    def supported(self) -> bool:
        return (
            FLYDSL_ENABLED
            and is_flydsl_available()
            and self.OP is not None
            and self.OP.is_available()
        )


@register_decode_op
class FlyDSLDecode(_FlyDSLDecodeOp):
    OP = flydsl_decoder.FwOp


@register_decode_op
class FlyDSLSplitK(_FlyDSLDecodeOp):
    OP = flydsl_splitk.FwOp


@register_decode_op
class CKDecode(DecodeOpBase):
    OP = ck_decoder.FwOp


@register_decode_op
class CKSplitK(DecodeOpBase):
    OP = ck_splitk.FwOp


@register_decode_op
class TritonSplitK(DecodeOpBase):
    OP = triton_splitk.FwOp

    def setup(self, B, Hq, Hkv, kv_seqlen, D, dtype) -> tuple:
        # Triton split-K does not broadcast KV heads itself; materialize the expanded
        # (stride-0) K/V into contiguous tensors.
        q, k, v, attn_bias = super().setup(B, Hq, Hkv, kv_seqlen, D, dtype)
        return q, k.contiguous(), v.contiguous(), attn_bias


# --------------------------------------------------------------------------- #
# fp8 KV-cache ops (--dtype fp8): the KV cache is quantized once in setup, and
# only these ops run under fp8, so the dense and fp8 FlyDSL kernels never share a
# process.
# --------------------------------------------------------------------------- #


def _quant_pack_triton_fp8(x: torch.Tensor):
    """Quantize dense KV to Triton's int32-packed asymmetric fp8 format.
    Returns (packed_int32, scale_shift_int32) as ``InputsFp8`` expects."""
    fp8_dtype = get_fp8_constants()[0]
    fmax = torch.finfo(fp8_dtype).max
    Bx, M, G, H, Dx = x.shape
    xr = x.reshape(-1, Dx).float()
    shift = xr.mean(-1)
    xc = xr - shift[..., None]
    s = torch.nan_to_num(xc.abs().max(-1)[0] / fmax, posinf=1)
    xq = (xc / s[..., None]).to(fp8_dtype)
    packed = xq.view(torch.uint8).reshape(Bx, M, G, H, Dx).view(torch.int32)
    ss = (
        torch.concat(
            [s.reshape(Bx, M, G, H, 1).half(), shift.reshape(Bx, M, G, H, 1).half()],
            dim=-1,
        )
        .flatten(-2)
        .view(torch.int32)
    )
    return packed, ss


@register_decode_op
class FlyDSLFp8(DecodeOpBase):
    """FlyDSL native-fp8 (e4m3fn, per-token) paged decode over a precomputed fp8 cache.

    Runs the kernel directly (there is no dense fmha op for native fp8). Scratch is
    preallocated in setup so the launch is CUDA-graph capturable.
    """

    supported_dtypes = ("fp8",)

    @property
    def supported(self) -> bool:
        return FLYDSL_ENABLED and is_fp8_paged_decode_available()

    def setup(self, B, Hq, Hkv, kv_seqlen, D, dtype) -> tuple:
        dev = "cuda"
        # Native fp8 path uses G=1 with the KV heads in the H slot (the layout
        # dense_kv_to_fp8_paged expects); query dtype is bf16/f16.
        q = torch.randn(B, 1, 1, Hq, D, dtype=dtype, device=dev)
        k = torch.randn(B, kv_seqlen, 1, Hkv, D, dtype=dtype, device=dev)
        v = torch.randn(B, kv_seqlen, 1, Hkv, D, dtype=dtype, device=dev)
        key_cache, value_cache, key_scale, value_scale, block_tables = (
            dense_kv_to_fp8_paged(k, v, block_size=16)
        )
        out = torch.zeros(B, Hq, D, dtype=q.dtype, device=dev)
        q_flat = q.reshape(B, Hq, D).contiguous()
        context_lengths = torch.full((B,), kv_seqlen, dtype=torch.int32, device=dev)
        num_kv_heads = key_cache.shape[1]
        eqgs = Hq // num_kv_heads  # query_length == 1 for decode
        block_size = key_cache.shape[-2]
        mcpn = get_recommended_splits(
            B, num_kv_heads, split_kv_blocks=KV_COMPUTE_BLOCK // block_size
        )
        exp_sums = torch.zeros(
            B, num_kv_heads, mcpn, eqgs, device=dev, dtype=torch.float32
        )
        max_logits = torch.full(
            (B, num_kv_heads, mcpn, eqgs), float("-inf"), device=dev, dtype=torch.float32
        )
        tmp_out = torch.zeros(
            B, num_kv_heads, mcpn, eqgs, D, device=dev, dtype=torch.bfloat16
        )
        scale = float(D**-0.5)
        return (
            out,
            q_flat,
            key_cache,
            value_cache,
            context_lengths,
            key_scale,
            value_scale,
            block_tables,
            exp_sums,
            max_logits,
            tmp_out,
            mcpn,
            scale,
        )

    def compute(
        self,
        out,
        q_flat,
        key_cache,
        value_cache,
        context_lengths,
        key_scale,
        value_scale,
        block_tables,
        exp_sums,
        max_logits,
        tmp_out,
        mcpn,
        scale,
    ) -> torch.Tensor:
        pa_decode_ps_launch(
            out,
            q_flat,
            key_cache,
            value_cache,
            context_lengths,
            scale,
            key_scale=key_scale,
            value_scale=value_scale,
            block_tables=block_tables,
            max_context_partition_num=mcpn,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=tmp_out,
        )
        return out


@register_decode_op
class TritonFp8(DecodeOpBase):
    """Triton split-K decode over int32-packed asymmetric fp8 KV (``InputsFp8``)."""

    OP = triton_splitk.FwOp
    supported_dtypes = ("fp8",)

    def setup(self, B, Hq, Hkv, kv_seqlen, D, dtype) -> tuple:
        dev = "cuda"
        Hpg = Hq // Hkv
        shape = (1, B * kv_seqlen, Hkv, Hpg, D)
        q = torch.randn(1, B, Hkv, Hpg, D, dtype=dtype, device=dev)
        k = (
            torch.randn(1, B * kv_seqlen, Hkv, 1, D, dtype=dtype, device=dev)
            .expand(shape)
            .contiguous()
        )
        v = (
            torch.randn(1, B * kv_seqlen, Hkv, 1, D, dtype=dtype, device=dev)
            .expand(shape)
            .contiguous()
        )
        attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
            q_seqlen=[1] * B,
            kv_seqlen=[kv_seqlen] * B,
            kv_padding=kv_seqlen,
        )
        attn_bias.k_seqinfo.to(dev)
        attn_bias.q_seqinfo.to(dev)
        ki, ks = _quant_pack_triton_fp8(k)
        vi, vs = _quant_pack_triton_fp8(v)
        return q, ki, vi, ks, vs, attn_bias, float(D**-0.5)

    def compute(self, q, ki, vi, ks, vs, attn_bias, scale) -> torch.Tensor:
        inp = InputsFp8(
            q,
            ki,
            vi,
            attn_bias=attn_bias,
            scale=scale,
            k_fp8_scale_shift=ks,
            v_fp8_scale_shift=vs,
        )
        out, _ = self.OP.apply(inp, False)
        return out
