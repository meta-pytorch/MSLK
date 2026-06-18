# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import os
import unittest
from typing import Optional, Union

import mslk.gemm  # noqa: F401
import mslk.quantize  # noqa: F401
import torch
import triton  # noqa: F401
from mslk.quantize.triton.fp4_quantize import (
    _to_blocked,
    calculate_group_max,
    get_nvfp4_global_scales_naive,
    nvfp4_quantize_stacked,
    nvfp4_quantize_stacked_with_token_scale,
    quantize_nvfp4_naive,
    triton_quantize_mx4_unpack,
)
from mslk.utils.device import (
    compute_capability_in,
    gfx_arch_in,
    is_cuda,
    is_gfx942,
    is_gfx950,
    is_rocm,
    supports_float8_fnuz,
)

if torch.cuda.is_available():
    from mslk.gemm.triton.fp8_gemm import matmul_fp8_block, matmul_fp8_row
    from mslk.quantize.shuffle import quantize_int4_preshuffle
    from mslk.quantize.triton.fp8_quantize import quantize_fp8_block, quantize_fp8_row

from parameterized import parameterized

# Marlin is currently only supported internally at Meta.
MARLIN_ENABLED = False
try:
    if not torch.version.hip:
        from marlin.quantize import marlin_quantize

        torch.ops.load_library("//ai_codesign/gen_ai/marlin:marlin_ops")
        MARLIN_ENABLED = True
except ImportError:
    pass

running_on_github: bool = os.getenv("GITHUB_ENV") is not None


# Feature-support matrix for the GEMM kernels exercised in this file. These map
# the device's arch/compute-capability (via mslk.utils.device primitives) to
# whether a given GEMM dtype path is supported, and are specific to these tests.
def supports_bf16():
    if is_rocm():
        return is_gfx942()
    return compute_capability_in(9)


def supports_fp8():
    if is_rocm():
        return is_gfx942() and supports_float8_fnuz(throw_on_hip_incompatibility=False)
    return compute_capability_in(9, 10)


def supports_mxfp8():
    if is_rocm():
        return False
    return compute_capability_in(10)


def supports_bf16_int4():
    return is_cuda() and compute_capability_in(9, 9)


def supports_fp8_int4():
    return is_cuda() and compute_capability_in(9, 9)


def supports_nvfp4():
    return is_cuda() and compute_capability_in(10)


def supports_nvfp4_ultra():
    if not is_cuda() or torch.version.cuda is None:
        return False
    cuda_major = int(torch.version.cuda.split(".")[0])
    major, minor = torch.cuda.get_device_capability()
    return cuda_major >= 13 and (major, minor) >= (10, 3)


def supports_mxfp4():
    if is_rocm():
        # TODO add AMD here later
        return False
    return compute_capability_in(10)


def supports_int8():
    """True on CUDA SM80+ or ROCm CDNA3 (gfx942) / CDNA4 (gfx950)."""
    if is_rocm():
        return gfx_arch_in(["gfx942", "gfx950"])
    return compute_capability_in(8)


SUPPORTS_BF16 = supports_bf16()
SUPPORTS_FP8 = supports_fp8()
SUPPORTS_MXFP8 = supports_mxfp8()
SUPPORTS_FP8_INT4 = supports_fp8_int4()
SUPPORTS_BF16_INT4 = supports_bf16_int4()
SUPPORTS_NVFP4 = supports_nvfp4()
SUPPORTS_NVFP4_ULTRA = supports_nvfp4_ultra()
SUPPORTS_MXFP4 = supports_mxfp4()

if torch.cuda.is_available() and supports_float8_fnuz(
    throw_on_hip_incompatibility=(not running_on_github)
):
    # Supported FP8 format is different on NV and AMD.
    fp8_e4m3: torch.dtype = torch.float8_e4m3fnuz
    fp8_e5m2: torch.dtype = torch.float8_e5m2fnuz
else:
    fp8_e4m3: torch.dtype = torch.float8_e4m3fn
    fp8_e5m2: torch.dtype = torch.float8_e5m2


E4M3_MAX_POS: float = torch.finfo(fp8_e4m3).max
EPS: float = 1e-12
FP16_MAX_POS: float = torch.finfo(torch.float16).max

# pyre-fixme[16]: Module `mslk` has no attribute `open_source`.
open_source: bool = getattr(mslk, "open_source", False)


def int4_row_quantize(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_bit = 4  # Number of target bits.
    to_quant = x.reshape(-1, group_size).to(torch.float)

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int

    zeros = min_val + scales * (2 ** (n_bit - 1))

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)

    # Recenter output and move to int8.
    out = (out - 2 ** (n_bit - 1)).to(dtype=torch.int8).reshape(x.shape)

    # Cutlass expects column major layout for scale and zero point,
    # so we transpose here and make them contiguous.
    scales = scales.view(x.shape[0], -1).t().contiguous()
    zeros = zeros.view(x.shape[0], -1).t().contiguous()

    return out, scales, zeros


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    # Given int8 x, pack adjacent int4 values into a single int8.
    low_x = x[:, ::2]
    high_x = x[:, 1::2]

    # High bits need to left shift, this also masks off extra bits.
    high_x = torch.bitwise_left_shift(high_x, 4)
    # Low bits need to have sign bits removed.
    low_x = torch.bitwise_and(low_x, 0xF)

    # Recombine into a single value with bitwise or.
    return torch.bitwise_or(low_x, high_x).contiguous()


# Source: https://github.com/pytorch/ao/blob/568c1932a16ae9f30d48da214a88dc0013e98ed8/torchao/prototype/moe_training/utils.py#L310
def generate_jagged_offs(E, M, multiple_of=16, dtype=torch.int32, device="cuda"):
    """
    Utility function for tests and benchmarks.

    Generates a tensor of length E, containing random values divisible by `multiple_of`,
    from 0 to M, in sorted order, and where the final value in the tensor is always M.
    Args:
        E (int): The length of the tensor.
        M (int): The maximum value in the tensor.
    Returns:
        torch.Tensor: A tensor of length E with the specified properties.
    """
    import random

    # Ensure M is divisible by 16
    if M % multiple_of != 0:
        raise ValueError(f"M must be divisible by {multiple_of}")

    # Generate a list of possible values
    possible_values = list(range(multiple_of, M + 1, multiple_of))

    # If E is larger than the number of possible values, raise an error
    if E > len(possible_values):
        raise ValueError("E cannot be larger than the number of possible values")

    # Randomly select E - 1 values from the possible values (excluding M)
    selected_values = torch.tensor(random.sample(possible_values[:-1], E - 1))

    # Append M to the selected values
    selected_values = torch.cat((selected_values, torch.tensor([M])))

    # Sort the selected values
    selected_values, _ = torch.sort(selected_values)

    return selected_values.to(dtype).to(device)


def _fp8_gemm_cases() -> list[tuple]:
    modes = ["rowwise"] + (
        # Blockwise fp8 GEMM is numerically broken on AMD/MI300 (~99% relative
        # RMS error at all K); the loose tolerance only masked it while K was
        # small. Keep it NVIDIA-only until the CK kernel
        # (mslk/csrc/gemm/ck/fp8_blockwise_gemm.hip) is fixed. See T275007617.
        ["blockwise"]
        if torch.version.cuda is not None and compute_capability_in(9, 9)
        else []
    )

    def case(
        M: int,
        K: int,
        N: int,
        *,
        bias: bool = False,
        cudagraph: bool = False,
        use_triton: bool = False,
        fast_accum: bool = True,
        multi_dim: bool = False,
    ) -> tuple:
        # (M, K, N, Bias, CudaGraph, UseTriton, UseFastAccum, InputMultiDim)
        return (
            M,
            K,
            N,
            bias,
            cudagraph,
            use_triton,
            fast_accum,
            multi_dim,
        )

    cases = [
        case(256, 128, 256),  # small
        case(2048, 2048, 4096),  # medium
        case(4096, 4096, 8192),  # large
        case(0, 256, 4096),  # empty input
        # Other features
        case(2048, 256, 4096, bias=True),
        case(2048, 256, 4096, cudagraph=True),
        case(2048, 256, 4096, multi_dim=True),
    ]
    if torch.version.cuda is not None:
        cases += [
            case(2048, 256, 4096, use_triton=True),
            case(2048, 256, 4096, fast_accum=False),  # slow accumulation
        ]
    return [(*case, mode) for mode in modes for case in cases]


def _fp8_batched_gemm_cases() -> list[tuple]:
    modes = ["default"] + (["torch_3d3d"] if torch.version.hip else [])
    cases = []
    for mode in modes:
        for use_loopover in (True, False):
            # (B, M, N, K, use_loopover, Bias, mode)
            cases.append((1, 256, 128, 256, use_loopover, False, mode))  # small
            cases.append((4, 4096, 256, 512, use_loopover, False, mode))  # large
            # Fused bias is only supported on Nvidia for batched GEMM.
            if torch.version.cuda is not None:
                cases.append((4, 2048, 256, 512, use_loopover, True, mode))  # bias
    return cases


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Operators are only available on CUDA enabled machines",
)
@unittest.skipIf(open_source, "Temporarily disabled in OSS.")
@unittest.skipIf(
    not all((SUPPORTS_FP8, SUPPORTS_BF16_INT4, SUPPORTS_FP8_INT4)),
    "ExportCompileTests is not supported on this device.",
)
class ExportCompileTests(unittest.TestCase):
    """Test that GEMM ops can be compiled & exported."""

    @classmethod
    def setUpClass(cls):
        fp8_dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
        cls.device = torch.accelerator.current_accelerator()
        cls.M = 256
        cls.N = 256
        cls.K = 256
        cls.X = torch.randn(cls.M, cls.K, device=cls.device, dtype=torch.bfloat16)
        cls.XQ = torch.randn(cls.M, cls.K, device=cls.device).to(fp8_dtype)
        cls.WQ = torch.randn(cls.N, cls.K, device=cls.device).to(fp8_dtype)
        cls.output = torch.empty(cls.M, cls.N, device=cls.device, dtype=torch.bfloat16)
        cls.row_scale = torch.randn(cls.M, device=cls.device)
        cls.col_scale = torch.randn(cls.N, device=cls.device)
        cls.block_scale = torch.randn(cls.M // 128, cls.K // 128, device=cls.device)
        cls.tensor_scale = torch.tensor(1.0, device=cls.device)

    def test_f8f8bf16_export(self) -> None:
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
                M, K = xq.shape
                N, _ = wq.shape
                row_scale = torch.randn(M).cuda()
                col_scale = torch.randn(N).cuda()
                block_scale = torch.randn(M // 128, K // 128).cuda()
                _ = torch.ops.mslk.f8f8bf16_blockwise(xq, wq, block_scale, block_scale)
                o = torch.ops.mslk.f8f8bf16_rowwise(xq, wq, row_scale, col_scale)
                return o

        model = TestModule().cuda()
        M, N, K = 256, 256, 256
        fp8_dtype = torch.float8_e4m3fn
        if torch.version.hip:
            fp8_dtype = torch.float8_e4m3fnuz
        xq = torch.randn(M, K).to(fp8_dtype).cuda()
        wq = torch.randn(N, K).to(fp8_dtype).cuda()
        _ = torch.export.export(model, (xq, wq), strict=True)

    def test_compile_f8f8bf16_blockwise(self) -> None:
        torch.compile(torch.ops.mslk.f8f8bf16_blockwise)(
            self.XQ, self.WQ, self.block_scale, self.block_scale
        )

    def test_compile_f8f8bf16_rowwise(self) -> None:
        torch.compile(torch.ops.mslk.f8f8bf16_rowwise)(
            self.XQ,
            self.WQ,
            self.row_scale,
            self.col_scale,
        )

    def test_compile_f8f8bf16_rowwise_out(self) -> None:
        torch.compile(torch.ops.mslk.f8f8bf16_rowwise_out)(
            self.XQ, self.WQ, self.row_scale, self.col_scale, self.output
        )

    @unittest.skipIf(not is_gfx942(), "Requires MI300X")
    def test_compile_f8f8f16_rowwise(self) -> None:
        torch.compile(torch.ops.mslk.f8f8f16_rowwise)(
            self.XQ, self.WQ, self.row_scale, self.col_scale
        )

    @unittest.skipIf(not torch.version.cuda, "Requires CUDA")
    def test_compile_i8i8bf16(self) -> None:
        torch.compile(torch.ops.mslk.i8i8bf16)(
            self.XQ.view(torch.int8), self.WQ.view(torch.int8), 1.0, 1
        )

    @unittest.skipIf(not torch.version.cuda, "Requires CUDA")
    def test_compile_f8i4bf16_rowwise(self) -> None:
        torch.compile(torch.ops.mslk.f8i4bf16_rowwise)(
            self.XQ,
            self.WQ[:, ::2].view(torch.int8).contiguous(),
            self.row_scale,
            self.block_scale[0],
            self.block_scale[0],
        )

    @unittest.skipIf(not torch.version.cuda, "Requires CUDA")
    def test_compile_bf16i4bf16_rowwise(self) -> None:
        torch.compile(torch.ops.mslk.bf16i4bf16_rowwise)(
            self.X,
            self.WQ[:, ::2].view(torch.int8).contiguous(),
            self.block_scale[0].repeat(self.M).view(-1, self.M),
            self.block_scale[0].repeat(self.N).view(-1, self.N),
        )

    @unittest.skipIf(not torch.version.cuda, "Requires CUDA")
    def test_compile_bf16i4bf16_rowwise_batched(self) -> None:
        torch.compile(torch.ops.mslk.bf16i4bf16_rowwise_batched)(
            self.X.view(1, self.M, self.K),
            self.WQ[:, ::2].view(1, self.N, self.K // 2).view(torch.int8).contiguous(),
            self.block_scale[0].repeat(self.M).view(1, -1, self.M),
            self.block_scale[0].repeat(self.N).view(1, -1, self.N),
        )


@unittest.skipIf(not SUPPORTS_FP8, "FP8Tests is not supported on this device.")
class FP8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(_fp8_gemm_cases())
    @unittest.skipIf(
        torch.version.hip is not None and running_on_github,
        "type fp8e4b8 not supported in this architecture. The supported fp8 dtypes are ('fp8e5',)",
    )
    def test_gemm(
        self,
        M: int,
        K: int,
        N: int,
        Bias: bool,
        CudaGraph: bool,
        UseTriton: bool,
        UseFastAccum: bool,
        InputMultiDim: bool,
        Mode: str,
    ) -> None:
        # Slow accumulation is only supported on Nvidia.
        if torch.version.hip:
            UseFastAccum = True
        # Setup input shapes.
        if InputMultiDim:
            x = (
                torch.randn(
                    size=(3, M, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.1
            )
        else:
            x = (
                torch.randn(
                    size=(M, K),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                * 0.1
            )
        w = (
            torch.randn(
                size=(N, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )
        bias = (
            torch.randn(
                size=(N,),
                dtype=torch.bfloat16,
                device=self.device,
            )
            if Bias
            else None
        )

        if Mode == "rowwise":

            def f(
                x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor]
            ) -> torch.Tensor:
                xq, x_scale = quantize_fp8_row(x)
                wq, w_scale = quantize_fp8_row(w)
                if UseTriton and torch.version.cuda:
                    zq = matmul_fp8_row(xq, wq, x_scale, w_scale)
                    if bias is not None:
                        zq += bias
                else:
                    zq = torch.ops.mslk.f8f8bf16_rowwise(
                        xq,
                        wq,
                        x_scale,
                        w_scale,
                        bias=bias if torch.version.cuda else None,
                        use_fast_accum=UseFastAccum,
                    )
                    # Bias fusion not yet supported on AMD.
                    if bias is not None and torch.version.hip:
                        zq += bias

                return zq

            if CudaGraph:
                # Warm-up to avoid capture issues
                f(x, w, bias)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    zq = f(x, w, bias)
                g.replay()
            else:
                zq = f(x, w, bias)
        elif Mode == "blockwise":

            def f(
                x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor]
            ) -> torch.Tensor:
                block_m = block_n = block_k = 128
                wq, w_scale = quantize_fp8_block(
                    w, block_n, block_k, output_device=torch.device(self.device)
                )
                xq, x_scale = quantize_fp8_block(x, block_m, block_k)
                if UseTriton:
                    zq = matmul_fp8_block(
                        xq,
                        wq,
                        x_scale,
                        w_scale,
                        block_m,
                        block_n,
                        block_k,
                        fp8_fast_accum=UseFastAccum,
                    )
                else:
                    zq = torch.ops.mslk.f8f8bf16_blockwise(
                        xq, wq, x_scale, w_scale, block_m, block_n, block_k
                    )
                if bias is not None:
                    zq += bias

                return zq

            if CudaGraph:
                # Warm-up to avoid capture issues
                f(x, w, bias)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    zq = f(x, w, bias)
                g.replay()
            else:
                zq = f(x, w, bias)
        else:
            raise ValueError(f"Invalid mode {Mode}")

        zq_ref = (x @ w.T).to(torch.bfloat16)
        if bias is not None:
            zq_ref += bias

        # Blockwise seems to have slightly more noisy outputs.
        # Special case correctness to avoid flakiness.
        if Mode == "blockwise":
            atol = 1.3e-1
            rtol = 1.3e-1
        else:
            atol = 9.0e-2
            rtol = 9.0e-2
        torch.testing.assert_close(zq, zq_ref, atol=atol, rtol=rtol)

    @parameterized.expand(_fp8_batched_gemm_cases())
    def test_batched_gemm(
        self,
        B: int,
        M: int,
        N: int,
        K: int,
        use_loopover: bool,
        Bias: bool,
        mode: str,
    ) -> None:
        # AMD CK FP8 batched gemm does not support N < 512 or K < 512.
        # Funny enough, grouped gemm does not have this restriction.
        if mode == "default" and torch.version.hip and (N < 512 or K < 512):
            return

        x = (
            torch.rand(
                size=(B, M, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.rand(
                size=(B, N, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )
        bias = (
            torch.randn(
                size=(B, N),
                dtype=torch.float32,
                device=self.device,
            )
            if Bias
            else None
        )

        xq, x_scale = quantize_fp8_row(x)
        x_scale = x_scale.view(B, -1)
        assert x_scale.shape == (B, M)
        wq, w_scale = quantize_fp8_row(w)
        w_scale = w_scale.view(B, -1)
        assert w_scale.shape == (B, N)

        def fp8_loopover_bmm(
            xq: torch.Tensor,
            wq: torch.Tensor,
            x_scale: torch.Tensor,
            w_scale: torch.Tensor,
            bias: Optional[torch.Tensor],
        ) -> torch.Tensor:
            B = len(xq)
            M = xq[0].shape[0]
            N = wq[0].shape[0]
            y = torch.empty((B, M, N), dtype=torch.bfloat16, device=xq[0].device)
            for i in range(B):
                y[i] = torch.ops.mslk.f8f8bf16_rowwise(
                    xq[i],
                    wq[i],
                    x_scale[i],
                    w_scale[i],
                    bias[i] if bias is not None else None,
                )
            return y

        y_ref = torch.bmm(x, w.transpose(1, 2))
        if bias is not None:
            y_ref += bias.unsqueeze(1)

        if use_loopover:
            y_fp8 = fp8_loopover_bmm(xq, wq, x_scale, w_scale, bias)
        else:
            if mode == "default":
                y_fp8 = torch.ops.mslk.f8f8bf16_rowwise_batched(
                    xq, wq, x_scale, w_scale, bias
                )
            elif mode == "torch_3d3d":
                y_fp8_ = torch.empty(
                    (B, M, N), dtype=torch.bfloat16, device=xq[0].device
                )
                y_fp8 = torch.ops.mslk.f8f8bf16_rowwise_grouped_mm(
                    xq,
                    wq,
                    x_scale,
                    w_scale,
                    None,
                    y_fp8_,
                )

        torch.testing.assert_close(y_ref, y_fp8, atol=8.0e-2, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 512, 512, 512),  # small MNK (also small G)
            (16, 2048, 1024, 2048),  # medium MNK
            (64, 3584, 6144, 3584),  # large MNK
            (16, 512, 512, 512),  # medium G
            (64, 512, 512, 512),  # large G
            (1, 0, 512, 512),  # empty (M=0)
        ]
    )
    @unittest.skipIf(
        not is_gfx942(),
        "Only MI300X supports torch 3D-2D grouped gemm API",
    )
    def test_grouped_gemm_3d_2d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        N_sizes = (
            torch.randint(
                1,
                (N // 64) + 1,
                (G,),
                dtype=torch.int,
            )
            * 64
        )
        N = torch.sum(N_sizes).item()
        N_offsets = torch.cumsum(N_sizes, dim=0).to(
            device=self.device, dtype=torch.int32
        )

        X = torch.randn((G, M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
        out = torch.empty((M, N), dtype=torch.bfloat16, device=self.device)

        xq, x_scale = quantize_fp8_row(X)
        wq, w_scale = quantize_fp8_row(W)

        y = torch.ops.mslk.f8f8bf16_rowwise_grouped_mm(
            xq, wq, x_scale, w_scale, N_offsets, out
        )

        # Compare using loopover BF16 gemm
        y_fp8 = torch.split(y, tuple(N_sizes), dim=1)
        W_split = torch.split(W, tuple(N_sizes), dim=0)
        self.bf16_loopover_validate(X, W_split, y_fp8)

    @parameterized.expand(
        [
            (1, 512, 512, 512, False),  # small MNK (also small G)
            (16, 2048, 1024, 2048, False),  # medium MNK
            (64, 3584, 6144, 3584, False),  # large MNK
            (16, 512, 512, 512, False),  # medium G
            (64, 512, 512, 512, False),  # large G
            (1, 0, 512, 512, False),  # empty (M=0)
            (16, 2048, 1024, 512, True),  # cudagraph
        ]
    )
    @unittest.skipIf(
        not is_gfx942(),
        "Only MI300X supports torch 2D-2D grouped gemm API",
    )
    def test_grouped_gemm_2d_2d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        use_cudagraph: bool,
    ) -> None:
        K_sizes = torch.ones((G,), dtype=torch.int, device=self.device) * K
        K_offsets = torch.cumsum(K_sizes, dim=0).to(
            device=self.device, dtype=torch.int32
        )

        # Each group should be quantized rowwise separately
        X_list = []
        W_list = []
        xq_list = []
        wq_list = []
        x_scale_list = []
        w_scale_list = []
        for k_size in K_sizes.tolist():
            X = torch.randn((M, k_size), dtype=torch.bfloat16, device=self.device) * 0.1
            W = (
                torch.randn((N, k_size), dtype=torch.bfloat16, device=self.device)
                * 0.01
            )
            xq, x_scale = quantize_fp8_row(X)
            wq, w_scale = quantize_fp8_row(W)

            X_list.append(X)
            W_list.append(W)
            xq_list.append(xq)
            wq_list.append(wq)
            x_scale_list.append(x_scale)
            w_scale_list.append(w_scale)

        xq = torch.cat(xq_list, dim=1)
        wq = torch.cat(wq_list, dim=1)
        x_scale = torch.cat(x_scale_list, dim=0)
        w_scale = torch.cat(w_scale_list, dim=0)

        out = torch.empty((G, M, N), dtype=torch.bfloat16, device=self.device)

        if use_cudagraph:
            # warmup
            torch.ops.mslk.f8f8bf16_rowwise_grouped_mm(
                xq, wq, x_scale, w_scale, K_offsets, out
            )
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y = torch.ops.mslk.f8f8bf16_rowwise_grouped_mm(
                    xq, wq, x_scale, w_scale, K_offsets, out
                )
            g.replay()
        else:
            y = torch.ops.mslk.f8f8bf16_rowwise_grouped_mm(
                xq, wq, x_scale, w_scale, K_offsets, out
            )

        # Compare using loopover BF16 gemm
        self.bf16_loopover_validate(X_list, W_list, y)

    def bf16_loopover_validate(
        self,
        x: Union[torch.Tensor, list[torch.Tensor]],
        w: Union[torch.Tensor, list[torch.Tensor]],
        out_fp8: Union[torch.tensor, list[torch.Tensor]],
        out_bf16: Union[torch.tensor, list[torch.Tensor], None] = None,
        atol_fp8=8.0e-2,
        rtol_fp8=8.0e-2,
        atol_bf16=8.0e-3,
        rtol_bf16=8.0e-3,
    ):
        out_ref = [torch.matmul(x[i], w[i].t()) for i in range(len(x))]

        for i in range(len(out_fp8)):
            torch.testing.assert_close(
                out_fp8[i], out_ref[i], atol=atol_fp8, rtol=rtol_fp8
            )

        if out_bf16:
            for i in range(len(out_bf16)):
                torch.testing.assert_close(
                    out_bf16[i], out_ref[i], atol=atol_bf16, rtol=rtol_bf16
                )

    @parameterized.expand(
        [
            (*case, mode)
            for mode in ["stacked"] + (["torch_2d3d"] if torch.version.hip else [])
            for case in [
                (1, 512, 512, 512, False),  # small MNK (also small G)
                (16, 2048, 1024, 2048, False),  # medium MNK
                (64, 3584, 6144, 3584, False),  # large MNK
                (16, 512, 512, 512, False),  # medium G
                (64, 512, 512, 512, False),  # large G
                (1, 0, 512, 512, False),  # empty (M=0)
                (16, 2048, 1024, 512, True),  # cudagraph
            ]
        ]
    )
    def test_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        use_cudagraph: bool,
        mode: str,
    ) -> None:
        # TODO remove this restriction.
        if (N < 512 or K < 512) and mode == "stacked":
            return

        if M > 0:
            M_sizes = (
                torch.randint(
                    1,
                    (M // 64) + 1,
                    (G,),
                    dtype=torch.int,
                )
                * 64
            )
        else:
            M_sizes = torch.zeros((G,), dtype=torch.int)

        M = torch.sum(M_sizes).item()
        X = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((G, N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        xq, x_scale = quantize_fp8_row(X)
        wq, w_scale = quantize_fp8_row(W)

        # FP8 grouped gemm kernel
        if mode == "stacked":
            fp8_op = torch.ops.mslk.f8f8bf16_rowwise_grouped_stacked
            M_sizes_gpu = M_sizes.clone().to(device=self.device, dtype=torch.int64)
            fp8_args = [xq, wq, x_scale, w_scale, M_sizes_gpu]

            bf16_op = torch.ops.mslk.bf16bf16bf16_grouped_stacked
            bf16_args = [X, W, M_sizes_gpu]
        elif mode == "torch_2d3d":
            fp8_op = torch.ops.mslk.f8f8bf16_rowwise_grouped_mm
            M_offsets = torch.cumsum(M_sizes, dim=0).to(
                device=self.device, dtype=torch.int32
            )
            out = torch.empty(M, N).to(device=self.device, dtype=torch.bfloat16)
            fp8_args = [
                xq,
                wq,
                x_scale,
                w_scale,
                M_offsets,
                out,
            ]

            bf16_op = None
            bf16_args = None

        if use_cudagraph:
            # warmup
            fp8_op(*fp8_args)
            # With cudagraph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_fp8_group = fp8_op(*fp8_args)
            g.replay()
        else:
            y_fp8_group = fp8_op(*fp8_args)

        # Massage output into proper format.
        y_fp8_group = torch.split(y_fp8_group, tuple(M_sizes.tolist()), dim=0)

        # unstack input to make it compatible with loopover.
        x_group = torch.split(X, tuple(M_sizes.tolist()), dim=0)

        y_bf16_group = None
        if bf16_op is not None:
            if use_cudagraph:
                # warmup
                bf16_op(*bf16_args)
                # With cudagraph
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    y_bf16_group = bf16_op(*bf16_args)
                g.replay()
            else:
                y_bf16_group = bf16_op(*bf16_args)

            y_bf16_group = torch.split(y_bf16_group, tuple(M_sizes.tolist()), dim=0)

        # BF16 loopover gemm reference
        self.bf16_loopover_validate(x_group, W, y_fp8_group, y_bf16_group)


@unittest.skipIf(
    not SUPPORTS_BF16_INT4, "Skip if BF16Int4Tests is not supported on this device."
)
class BF16Int4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (*case, preshuffle)
            for preshuffle in [True, False]
            for case in [
                (256, 128, 256, False),  # small
                (2048, 2048, 4096, False),  # medium
                (4096, 4096, 8192, False),  # large
                (2048, 256, 512, True),  # cudagraph
            ]
        ]
    )
    def test_gemm(
        self,
        M: int,
        K: int,
        N: int,
        CudaGraph: bool,
        preshuffle: bool,
    ) -> None:
        x = (
            torch.randn(
                size=(M, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(N, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

        if preshuffle:
            wq, (w_scale, w_zp) = quantize_int4_preshuffle(w, dtype="bf16")
        else:
            wq, w_scale, w_zp = int4_row_quantize(w, 128)
            wq = pack_int4(wq).contiguous().to(device=self.device)
            w_scale = w_scale.contiguous().to(device=self.device)
            w_zp = w_zp.contiguous().to(device=self.device)

        bf16i4_op = (
            torch.ops.mslk.bf16i4bf16_shuffled
            if preshuffle
            else torch.ops.mslk.bf16i4bf16_rowwise
        )

        if CudaGraph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                zq = bf16i4_op(x, wq, w_scale, w_zp)
            g.replay()
        else:
            zq = bf16i4_op(x, wq, w_scale, w_zp)

        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq, zq_ref, atol=1.0e-1, rtol=8.0e-2)

    @parameterized.expand(
        [
            (*case, use_loopover)
            for use_loopover in [True, False]
            for case in [
                (1, 256, 256, 256),  # small
                (4, 2048, 256, 256),  # medium
                (4, 4096, 512, 512),  # large
            ]
        ]
    )
    @unittest.skipIf(not MARLIN_ENABLED, "Skip if Marlin is not enabled.")
    def test_batched_gemm(
        self,
        B: int,
        M: int,
        N: int,
        K: int,
        use_loopover: bool,
    ) -> None:
        x = (
            torch.rand(
                size=(B, M, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.rand(
                size=(B, N, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

        wq = []
        w_scale = []
        group_size = 128

        if use_loopover:
            for i in range(B):
                _, wq_, w_scale_ = marlin_quantize(
                    w[i].cuda().t().contiguous(), group_size
                )
                wq.append(wq_)
                w_scale.append(w_scale_)
            wq = torch.stack(wq)
            w_scale = torch.stack(w_scale)

            def int4_loopover_bmm(
                x: torch.Tensor,
                wq: torch.Tensor,
                w_scale: torch.Tensor,
            ) -> torch.Tensor:
                B = x.shape[0]
                M = x.shape[1]
                N = w_scale.shape[2]
                y = torch.empty((B, M, N), dtype=torch.bfloat16, device=x[0].device)
                for i in range(B):
                    y[i] = torch.ops.marlin.marlin_gemm(x[i], wq[i], w_scale[i])
                return y

            y_int4 = int4_loopover_bmm(x, wq, w_scale)
        else:
            w_zp = []
            for i in range(B):
                wq_, w_scale_, w_zp_ = int4_row_quantize(w[i], group_size)

                wq_ = pack_int4(wq_).contiguous().to(device=self.device)
                w_scale_ = w_scale_.contiguous().to(device=self.device)
                w_zp_ = w_zp_.contiguous().to(device=self.device)
                wq.append(wq_)
                w_scale.append(w_scale_)
                w_zp.append(w_zp_)
            wq = torch.stack(wq)
            w_scale = torch.stack(w_scale).view(-1, N)
            w_zp = torch.stack(w_zp).view(-1, N)
            y_int4 = torch.ops.mslk.bf16i4bf16_rowwise_batched(x, wq, w_scale, w_zp)

        y_ref = torch.bmm(x, w.transpose(1, 2))
        torch.testing.assert_close(y_ref, y_int4, atol=1e-1, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 256, 1024, 512, False),  # small
            (16, 2048, 1024, 512, False),  # medium
            (64, 3584, 6144, 3584, False),  # large
            (1, 0, 1024, 512, False),  # empty (M=0)
            (16, 2048, 1024, 512, True),  # cudagraph
        ]
    )
    @unittest.skipIf(not torch.version.cuda, "Currently not supported on AMD.")
    def test_shuffled_grouped_gemm(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        use_cudagraph: bool,
    ) -> None:
        if M > 0:
            ms = (
                torch.randint(
                    1,
                    (M // 64) + 1,
                    (G,),
                    dtype=torch.int,
                )
                * 64
            )
        else:
            ms = torch.zeros((G,), dtype=torch.int)

        M_sizes = ms.to(device=self.device, dtype=torch.int64)
        ns = [N] * G
        ks = [K] * G

        x_group = []
        w_group = []
        xq_group = []
        x_scale_group = []
        w_bf16_group = []
        bf16_group_scales = []
        bf16_group_zeros = []

        for _, (m, n, k) in enumerate(zip(ms, ns, ks)):
            x = torch.rand(
                size=(m, k),
                dtype=torch.bfloat16,
                device=self.device,
            )
            w = torch.rand(
                size=(n, k),
                dtype=torch.bfloat16,
                device=self.device,
            )

            xq, x_scale = quantize_fp8_row(x)
            w_fp8, (fp8_group_scale, fp8_row_scale) = quantize_int4_preshuffle(w)
            w_bf16, (bf16_group_scale, bf16_group_zero) = quantize_int4_preshuffle(
                w, dtype="bf16", use_zp=False
            )
            x_group.append(x)
            w_group.append(w)
            xq_group.append(xq)
            x_scale_group.append(x_scale)
            w_bf16_group.append(w_bf16)
            bf16_group_scales.append(bf16_group_scale)
            bf16_group_zeros.append(bf16_group_zero)

        # Only stacked API currently available for preshuffled grouped gemm.
        x_group = torch.cat(x_group, dim=0).contiguous()
        w_group = torch.stack(w_group, dim=0).contiguous()
        xq_group = torch.cat(xq_group, dim=0).contiguous()
        x_scale_group = torch.cat(x_scale_group, dim=0).contiguous()
        w_bf16_group = torch.stack(w_bf16_group, dim=0).contiguous()
        bf16_group_scales = torch.stack(bf16_group_scales, dim=0).contiguous()
        bf16_group_zeros = torch.stack(bf16_group_zeros, dim=0).contiguous()

        bf16_op = torch.ops.mslk.bf16i4bf16_shuffled_grouped
        bf16_args = [
            x_group,
            w_bf16_group,
            bf16_group_scales,
            bf16_group_zeros,
            M_sizes,
        ]

        if use_cudagraph:
            # warmup
            bf16_op(*bf16_args)
            # With cudagraph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_bf16_group = bf16_op(*bf16_args)
            g.replay()
        else:
            y_bf16_group = bf16_op(*bf16_args)

        # View output as list if needed.
        y_bf16_group = torch.split(y_bf16_group, tuple(ms.tolist()), dim=0)

        # BF16 loopover gemm reference
        # unstack input to make it compatible with loopover.
        x_group = torch.split(x_group, tuple(ms.tolist()), dim=0)
        y_group_ref = []
        for i in range(len(x_group)):
            y = torch.matmul(x_group[i], w_group[i].t())
            y_group_ref.append(y)

        # Assert BF16 outputs
        for i in range(len(y_group_ref)):
            torch.testing.assert_close(
                y_bf16_group[i], y_group_ref[i], atol=8.0e-2, rtol=5.0e-2
            )


@unittest.skipIf(torch.version.hip is None, "ROCm-only: BF16xINT4 Triton rowwise GEMM")
class BF16Int4TritonROCmTests(unittest.TestCase):
    """
    Tests for the Triton BF16xINT4 rowwise GEMM running on AMD GPUs.

    Imports the Triton kernel module, which also registers the torch op
    implementations via torch.library.impl for the mslk:: namespace.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = "cuda"
        from mslk.gemm.triton.int4_gemm import (  # noqa: F401
            matmul_bf16i4_rowwise,
            matmul_bf16i4_rowwise_batched,
        )

        cls.matmul_rowwise = staticmethod(matmul_bf16i4_rowwise)
        cls.matmul_rowwise_batched = staticmethod(matmul_bf16i4_rowwise_batched)

    @parameterized.expand(
        [
            (1, 256, 256, 128),  # small
            (512, 1024, 512, 128),  # medium
            (2048, 4096, 1024, 128),  # large
        ]
    )
    def test_rowwise_accuracy(
        self,
        M: int,
        N: int,
        K: int,
        group_size: int,
    ) -> None:
        """
        Checks that the Triton kernel output is close to a float32 reference.
        The tolerance (1e-1 / 8e-2) matches the existing CUDA CUTLASS test.
        """
        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.1
        w = torch.randn(N, K, dtype=torch.bfloat16, device=self.device) * 0.01

        wq, w_scale, w_zp = int4_row_quantize(w, group_size)
        wq = pack_int4(wq).contiguous().to(device=self.device)
        w_scale = w_scale.contiguous().to(device=self.device)
        w_zp = w_zp.contiguous().to(device=self.device)

        y = self.matmul_rowwise(x, wq, w_scale, w_zp)
        y_ref = (x.float() @ w.float().T).to(torch.bfloat16)

        torch.testing.assert_close(y, y_ref, atol=1.0e-1, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 64, 256, 256, 128),  # small
            (4, 512, 512, 512, 128),  # medium
            (4, 2048, 1024, 512, 128),  # large
        ]
    )
    def test_rowwise_batched_accuracy(
        self,
        B: int,
        M: int,
        N: int,
        K: int,
        group_size: int,
    ) -> None:
        """
        Checks the batched variant (loop-over-batch prototype) against float32 bmm.
        """
        x = torch.randn(B, M, K, dtype=torch.bfloat16, device=self.device) * 0.1
        w = torch.randn(B, N, K, dtype=torch.bfloat16, device=self.device) * 0.01

        wq_list, scale_list, zp_list = [], [], []
        for b in range(B):
            wq_b, s_b, z_b = int4_row_quantize(w[b], group_size)
            wq_list.append(pack_int4(wq_b))
            scale_list.append(s_b)
            zp_list.append(z_b)

        wq = torch.stack(wq_list).contiguous().to(device=self.device)
        w_scale = torch.cat(scale_list, dim=0).contiguous().to(device=self.device)
        w_zp = torch.cat(zp_list, dim=0).contiguous().to(device=self.device)

        y = self.matmul_rowwise_batched(x, wq, w_scale, w_zp)
        y_ref = torch.bmm(x.float(), w.float().transpose(1, 2)).to(torch.bfloat16)

        torch.testing.assert_close(y, y_ref, atol=1.0e-1, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 256, 256, 128),  # small
            (128, 1024, 256, 128),  # medium
            (2048, 4096, 1024, 128),  # large
        ]
    )
    def test_torch_op_dispatch(
        self,
        M: int,
        N: int,
        K: int,
        group_size: int,
    ) -> None:
        """
        Verifies that torch.ops.mslk.bf16i4bf16_rowwise dispatches to the
        Triton kernel on ROCm (requires mslk C++ library and int4_gemm.py
        both to be loaded).
        """
        if not hasattr(torch.ops, "mslk") or not hasattr(
            torch.ops.mslk, "bf16i4bf16_rowwise"
        ):
            self.skipTest("mslk:: ops not loaded; skipping dispatch test")

        x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.1
        w = torch.randn(N, K, dtype=torch.bfloat16, device=self.device) * 0.01

        wq, w_scale, w_zp = int4_row_quantize(w, group_size)
        wq = pack_int4(wq).contiguous().to(device=self.device)
        w_scale = w_scale.contiguous().to(device=self.device)
        w_zp = w_zp.contiguous().to(device=self.device)

        y_op = torch.ops.mslk.bf16i4bf16_rowwise(x, wq, w_scale, w_zp)
        y_direct = self.matmul_rowwise(x, wq, w_scale, w_zp)
        torch.testing.assert_close(y_op, y_direct, atol=0.0, rtol=0.0)


@unittest.skipIf(
    not SUPPORTS_FP8_INT4, "Skip if FP8Int4Tests is not supported on this device."
)
class FP8Int4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (256, 128, 256, False),  # small
            (2048, 2048, 4096, False),  # medium
            (4096, 4096, 8192, False),  # large
            (0, 128, 256, False),  # empty (M=0)
            (2048, 256, 512, True),  # cudagraph
        ]
    )
    def test_gemm(
        self,
        M: int,
        K: int,
        N: int,
        CudaGraph: bool,
    ) -> None:
        x = (
            torch.randn(
                size=(M, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.1
        )
        w = (
            torch.randn(
                size=(N, K),
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.01
        )

        # Standard i4 weight format.
        wq, w_scale, w_zp = int4_row_quantize(w, 128)
        wq = pack_int4(wq).contiguous().to(device=self.device)
        w_scale = w_scale.contiguous().to(device=self.device)
        w_zp = w_zp.contiguous().to(device=self.device)

        # Preshuffled i4 weight format.
        wq_shuffled, (w_scale_group, w_scale_row) = quantize_int4_preshuffle(w, 128)

        if CudaGraph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                xq, x_scale = quantize_fp8_row(x)
                zq = torch.ops.mslk.f8i4bf16_rowwise(xq, wq, x_scale, w_scale, w_zp)
                zq_shuffled = torch.ops.mslk.f8i4bf16_shuffled(
                    xq, wq_shuffled, x_scale, w_scale_row, w_scale_group
                )
            g.replay()
        else:
            xq, x_scale = quantize_fp8_row(x)
            zq = torch.ops.mslk.f8i4bf16_rowwise(xq, wq, x_scale, w_scale, w_zp)
            zq_shuffled = torch.ops.mslk.f8i4bf16_shuffled(
                xq, wq_shuffled, x_scale, w_scale_row, w_scale_group
            )

        zq_ref = (x @ w.T).to(torch.bfloat16)
        torch.testing.assert_close(zq, zq_ref, atol=8.0e-2, rtol=8.0e-2)
        torch.testing.assert_close(zq_shuffled, zq_ref, atol=8.0e-2, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 256, 1024, 512, False),  # small
            (4, 2048, 1024, 512, False),  # medium
            (16, 3584, 6144, 3584, False),  # large
            (64, 3584, 6144, 3584, False),  # large G
            (1, 0, 1024, 512, False),  # empty (M=0)
            (4, 2048, 1024, 512, True),  # cudagraph
        ]
    )
    def test_shuffled_grouped_gemm(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        use_cudagraph: bool,
    ) -> None:
        if M > 0:
            ms = (
                torch.randint(
                    1,
                    (M // 64) + 1,
                    (G,),
                    dtype=torch.int,
                )
                * 64
            )
        else:
            ms = torch.zeros((G,), dtype=torch.int)

        M_sizes = ms.to(device=self.device, dtype=torch.int64)
        ns = [N] * G
        ks = [K] * G

        x_group = []
        w_group = []
        xq_group = []
        x_scale_group = []
        w_fp8_group = []
        fp8_group_scales = []
        fp8_row_scales = []

        for _, (m, n, k) in enumerate(zip(ms, ns, ks)):
            x = torch.rand(
                size=(m, k),
                dtype=torch.bfloat16,
                device=self.device,
            )
            w = torch.rand(
                size=(n, k),
                dtype=torch.bfloat16,
                device=self.device,
            )

            xq, x_scale = quantize_fp8_row(x)
            w_fp8, (fp8_group_scale, fp8_row_scale) = quantize_int4_preshuffle(w)

            x_group.append(x)
            w_group.append(w)
            xq_group.append(xq)
            x_scale_group.append(x_scale)
            w_fp8_group.append(w_fp8)
            fp8_group_scales.append(fp8_group_scale)
            fp8_row_scales.append(fp8_row_scale)

        # Only stacked API currently available for preshuffled grouped gemm.
        x_group = torch.cat(x_group, dim=0).contiguous()
        w_group = torch.stack(w_group, dim=0).contiguous()
        xq_group = torch.cat(xq_group, dim=0).contiguous()
        x_scale_group = torch.cat(x_scale_group, dim=0).contiguous()
        w_fp8_group = torch.stack(w_fp8_group, dim=0).contiguous()
        fp8_group_scales = torch.stack(fp8_group_scales, dim=0).contiguous()
        fp8_row_scales = torch.stack(fp8_row_scales, dim=0).contiguous()

        fp8_op = torch.ops.mslk.f8i4bf16_shuffled_grouped
        fp8_args = [
            xq_group,
            w_fp8_group,
            x_scale_group,
            fp8_row_scales,
            fp8_group_scales,
            M_sizes,
        ]

        if use_cudagraph:
            # warmup
            fp8_op(*fp8_args)
            # With cudagraph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_fp8_group = fp8_op(*fp8_args)
            g.replay()
        else:
            y_fp8_group = fp8_op(*fp8_args)

        # Massage output into proper format.
        y_fp8_group = torch.split(y_fp8_group, tuple(ms.tolist()), dim=0)

        # BF16 loopover gemm reference
        # unstack input to make it compatible with loopover.
        x_group = torch.split(x_group, tuple(ms.tolist()), dim=0)
        y_group_ref = []
        for i in range(len(x_group)):
            y = torch.matmul(x_group[i], w_group[i].t())
            y_group_ref.append(y)

        # Assert FP8 outputs
        for i in range(len(y_group_ref)):
            torch.testing.assert_close(
                y_fp8_group[i], y_group_ref[i], atol=8.0e-2, rtol=2.0e-1
            )


@unittest.skipIf(
    not SUPPORTS_MXFP8, "Skip if MXFP8Tests is not supported on this device."
)
class MXFP8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (1, 2048, 256, 256),  # small
            (4, 2048, 1024, 512),  # medium
            (16, 3584, 6144, 3584),  # large
            (64, 3584, 6144, 3584),  # large G
        ]
    )
    def test_grouped_gemm_2d_2d(
        self,
        G: int,
        K: int,
        N: int,
        M: int,
    ) -> None:
        # Simulate 2d-2d grouped gemm in backward pass `grad_weight = grad_output_t @ input`,
        # where we use "K" as the contracting dim which has "G" groups.
        from mslk.gemm.triton.fp8_gemm import to_mxfp8

        total_K = K  # Alias for clarity, communicating this consists of several groups along this dim
        input_group_end_offsets = generate_jagged_offs(
            G, total_K, multiple_of=32, device=self.device
        )
        X = torch.randn((M, total_K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((N, total_K), dtype=torch.bfloat16, device=self.device) * 0.01

        # Convert scales to blocked format.
        x_list = []
        w_list = []
        x_blocked_scale_list = []
        w_blocked_scale_list = []

        def round_up(x: int, y: int) -> int:
            return ((x + y - 1) // y) * y

        for group_idx in range(G):
            # to_mxfp8 per group
            prev_group_end_offset = (
                0 if group_idx == 0 else input_group_end_offsets[group_idx - 1]
            )
            curr_group_end_offset = input_group_end_offsets[group_idx]
            group_size = curr_group_end_offset - prev_group_end_offset
            if group_size > 0:
                x_slice = X[
                    :, prev_group_end_offset:curr_group_end_offset
                ].contiguous()  # (M, K_group)
                w_slice = W[
                    :, prev_group_end_offset:curr_group_end_offset
                ].contiguous()  # (N, K_group)
                x_scale_slice, xq_slice = to_mxfp8(
                    x_slice
                )  # scale shape -> (M, K_group // 32)
                w_scale_slice, wq_slice = to_mxfp8(
                    w_slice
                )  # scale shape -> (N, K_group // 32)
                x_list.append(xq_slice)
                w_list.append(wq_slice)

                # Convert scales to blocked format.
                x_scale_slice_blocked = _to_blocked(
                    x_scale_slice
                )  # (round_up(M, 128), round_up(K_group//32, 4))
                w_scale_slice_blocked = _to_blocked(
                    w_scale_slice
                )  # (round_up(N, 128), round_up(K_group//32, 4))
                x_blocked_scale_list.append(x_scale_slice_blocked)
                w_blocked_scale_list.append(w_scale_slice_blocked)

        # Assemble the full XQ and WQ
        xq = torch.cat(x_list, dim=1).contiguous()
        wq = torch.cat(w_list, dim=1).contiguous()

        # Combine all XQ groups blocked scales into one tensor.
        x_blocked_scales = torch.cat(x_blocked_scale_list, dim=0)
        M_rounded = round_up(M, 128)
        x_blocked_scales = x_blocked_scales.reshape(M_rounded, -1)

        # Combine all WQ groups blocked scales into one tensor.
        w_blocked_scales = torch.cat(w_blocked_scale_list, dim=0)
        N_rounded = round_up(N, 128)
        w_blocked_scales = w_blocked_scales.reshape(N_rounded, -1)

        # Compute mxfp8 grouped mm output
        out = torch.empty((G, M, N), dtype=torch.bfloat16, device=self.device)
        y_mxfp8 = torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq,  # (M, total_K)
            wq.transpose(-2, -1),  # (total_K, N)
            x_blocked_scales,  # to_blocked_per_group(M, total_K//32)
            w_blocked_scales,  # to_blocked_per_group(N, total_K//32)
            input_group_end_offsets,  # (G,)
            out,  # (G, M, N)
        )

        # bf16 reference output
        y_bf16 = torch._grouped_mm(
            X, W.t(), offs=input_group_end_offsets, out_dtype=torch.bfloat16
        )

        # Assert no NaNs
        assert not y_mxfp8.isnan().any(), "mxfp8 output contains NaN"

        # Assert outputs are close
        torch.testing.assert_close(y_mxfp8, y_bf16, atol=8.0e-2, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 256, 256, 256),  # small
            (4, 2048, 1024, 512),  # medium
            (16, 3584, 6144, 3584),  # large
            (64, 3584, 6144, 3584),  # large G
        ]
    )
    def test_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        from mslk.gemm.triton.fp8_gemm import to_mxfp8

        # Simulate 2d-3d grouped gemm `out = input @ weight.t()`
        # 2D inputs with groups along M, 3D weights.
        block_size = 32
        total_M = M  # Alias for clarity that M dim contains groups.
        X = torch.randn((total_M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((G, N, K), dtype=torch.bfloat16, device=self.device) * 0.01
        input_group_end_offsets = generate_jagged_offs(
            G, total_M, multiple_of=32, device=self.device
        )

        # For each constituent 2d subtensor in the 3d weights, quantize and convert scale to blocked format separately,
        # as they each used for independent gemm in the grouped gemm.
        wq_list = []
        w_scale_list = []
        for i in range(G):
            w_scale, wq = to_mxfp8(W[i])
            w_scale = _to_blocked(w_scale)
            wq_list.append(wq)
            w_scale_list.append(w_scale)
        wq = torch.stack(wq_list, dim=0).contiguous()
        w_scale = torch.stack(w_scale_list, dim=0).contiguous()

        # For each group along `total_M` in the 2D tensor, quantize and convert scale to blocked format separately,
        # as they each used for independent gemm in the grouped gemm.
        xq_list = []
        x_scale_list = []
        for i in range(G):
            prev_group_end = 0 if i == 0 else input_group_end_offsets[i - 1]
            curr_group_end = input_group_end_offsets[i]
            group_size = curr_group_end - prev_group_end
            if group_size > 0:
                x_slice = X[prev_group_end:curr_group_end, :]
                x_scale, xq = to_mxfp8(x_slice)
                x_scale = _to_blocked(x_scale)
                xq_list.append(xq)
                x_scale_list.append(x_scale)
        xq = torch.cat(xq_list, dim=0).contiguous()
        x_scale = torch.cat(x_scale_list, dim=0).contiguous()
        x_scale = x_scale.reshape(-1, K // block_size)
        xq = xq.view(-1, xq.shape[-1])

        # Compute mxfp8 grouped gemm.
        out = torch.empty((total_M, N), dtype=torch.bfloat16, device=self.device)
        y_mxfp8 = torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq,
            wq.transpose(-2, -1),
            x_scale,
            w_scale,
            input_group_end_offsets,
            out,
        )

        # Compute reference bf16 grouped gemm.
        y_bf16 = torch._grouped_mm(
            X,
            W.transpose(-2, -1),
            offs=input_group_end_offsets,
            out_dtype=torch.bfloat16,
        )

        # Assert outputs are close.
        torch.testing.assert_close(y_mxfp8, y_bf16, atol=8.0e-2, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 16, 2, 1024, 512),  # small G
            (4, 16, 2, 1024, 512),  # small
            (16, 32, 4, 1024, 512),  # medium
            (32, 64, 8, 2048, 3072),  # large
            (64, 64, 8, 2048, 3072),  # large G
        ]
    )
    def test_grouped_gemm_2d_3d_actual_num_tokens(
        self,
        G: int,
        M_per_group: int,
        pad_factor: int,
        N: int,
        K: int,
    ) -> None:
        """Test that actual_num_tokens hint produces correct results.

        XQ is padded to pad_factor * actual_total_M, but offsets only cover the
        actual tokens. Verifies that:
        1. Output with hint matches output without hint (same GEMM, different tile)
        2. Output matches bf16 reference
        """
        from mslk.gemm.triton.fp8_gemm import to_mxfp8

        block_size = 32
        actual_total_M = M_per_group * G
        padded_total_M = actual_total_M * pad_factor

        # Create actual data
        X = (
            torch.randn((actual_total_M, K), dtype=torch.bfloat16, device=self.device)
            * 0.1
        )
        W = torch.randn((G, N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        # Uniform offsets for actual tokens
        offsets = torch.arange(
            M_per_group,
            actual_total_M + 1,
            M_per_group,
            dtype=torch.int32,
            device=self.device,
        )

        # Quantize weights
        wq_list = []
        w_scale_list = []
        for i in range(G):
            w_scale, wq = to_mxfp8(W[i])
            w_scale = _to_blocked(w_scale)
            wq_list.append(wq)
            w_scale_list.append(w_scale)
        wq = torch.stack(wq_list, dim=0).contiguous()
        w_scale = torch.stack(w_scale_list, dim=0).contiguous()

        # Quantize input per group
        xq_list = []
        x_scale_list = []
        for i in range(G):
            x_slice = X[i * M_per_group : (i + 1) * M_per_group]
            xs, xq = to_mxfp8(x_slice)
            xs = _to_blocked(xs)
            xq_list.append(xq)
            x_scale_list.append(xs)
        xq_actual = torch.cat(xq_list, dim=0).contiguous()
        x_scale_actual = torch.cat(x_scale_list, dim=0).contiguous()
        x_scale_actual = x_scale_actual.reshape(-1, K // block_size)

        # Pad XQ and x_scale to simulate static-shape padding
        xq_padded = torch.zeros(
            (padded_total_M, K), dtype=torch.float8_e4m3fn, device=self.device
        )
        xq_padded[:actual_total_M] = xq_actual

        actual_scale_rows = x_scale_actual.shape[0]
        padded_scale_rows = max(
            ((padded_total_M + 127) // 128) * 128, actual_scale_rows
        )
        x_scale_padded = torch.zeros(
            (padded_scale_rows, x_scale_actual.shape[1]),
            dtype=x_scale_actual.dtype,
            device=self.device,
        )
        x_scale_padded[:actual_scale_rows] = x_scale_actual

        # Run without hint (actual_num_tokens=None)
        out_no_hint = torch.empty(
            (padded_total_M, N), dtype=torch.bfloat16, device=self.device
        )
        y_no_hint = torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq_padded,
            wq.transpose(-2, -1),
            x_scale_padded,
            w_scale,
            offsets,
            out_no_hint,
        )

        # Run with hint (actual_num_tokens=actual_total_M)
        out_with_hint = torch.empty(
            (padded_total_M, N), dtype=torch.bfloat16, device=self.device
        )
        y_with_hint = torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq_padded,
            wq.transpose(-2, -1),
            x_scale_padded,
            w_scale,
            offsets,
            out_with_hint,
            actual_total_M,
        )

        # Outputs should be identical (only tile selection differs)
        torch.testing.assert_close(
            y_no_hint[:actual_total_M],
            y_with_hint[:actual_total_M],
            atol=0.0,
            rtol=0.0,
        )

        # Compare with bf16 reference
        y_bf16 = torch._grouped_mm(
            X,
            W.transpose(-2, -1),
            offs=offsets,
            out_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(
            y_with_hint[:actual_total_M], y_bf16, atol=8.0e-2, rtol=8.0e-2
        )


@unittest.skipIf(
    not SUPPORTS_BF16, "Skip if BF16Tests is not supported on this device."
)
class BF16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    def generate_random_splits(G: int, M: int) -> torch.Tensor:
        if M > 0:
            m_cumsums = torch.sort(
                torch.randint(
                    0,
                    M,
                    (G + 1,),
                    dtype=torch.int32,
                    device=torch.accelerator.current_accelerator(),
                )
            ).values
            m_cumsums[0], m_cumsums[-1] = 0, M
            m_sizes = m_cumsums[1:] - m_cumsums[:-1]
            return m_sizes
        else:
            return torch.zeros(
                (G,), dtype=torch.int32, device=torch.accelerator.current_accelerator()
            )

    @parameterized.expand(
        [
            (*case, dtype)
            for dtype in [torch.bfloat16, torch.float16]
            for case in [
                (1, 257, 256, 128, False),  # small G
                (2, 257, 256, 128, False),  # small
                (16, 2049, 2048, 1024, False),  # large
                (64, 2049, 2048, 1024, False),  # large G
                (2, 0, 256, 128, False),  # empty (M=0)
                (16, 2049, 2048, 1024, True),  # output_accum
            ]
        ]
    )
    @unittest.skipIf(
        not torch.version.cuda,
        "Skip on AMD: test_grouped_gemm_wgrad not yet supported.",
    )
    def test_grouped_gemm_wgrad(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        output_accum: bool,
        dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(hash((G, M, N, K)))
        # Inputs
        dy_bf16 = torch.randn(
            (M, N), dtype=dtype, device=torch.accelerator.current_accelerator()
        )
        x_bf16 = torch.randn(
            (M, K), dtype=dtype, device=torch.accelerator.current_accelerator()
        )

        m_sizes = BF16Tests.generate_random_splits(G, M)

        # Test
        if output_accum:
            wgrad_accum = torch.randn(
                (G, N, K),
                dtype=torch.float32,
                device=torch.accelerator.current_accelerator(),
            )
        else:
            wgrad_accum = None

        sm_margin = 0
        num_sms = (
            torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
        )

        test_wgrad = torch.ops.mslk.bf16bf16bf16_grouped_wgrad(
            dy_bf16,
            x_bf16,
            m_sizes.to(torch.int64),
            output=wgrad_accum.clone() if output_accum else None,
            output_accum=output_accum,
            num_sms=num_sms,
        )

        if output_accum:
            assert test_wgrad.dtype == torch.float32

        # Reference
        dy_fp32 = dy_bf16.to(torch.float32)
        x_fp32 = x_bf16.to(torch.float32)
        ref_wgrad = torch.zeros(
            (G, N, K),
            dtype=torch.float32,
            device=torch.accelerator.current_accelerator(),
        )

        # Track which groups have non-zero size for comparison
        non_zero_groups = []
        m_start = 0
        for g, m_size in enumerate(m_sizes.tolist()):
            if m_size > 0:
                # Actual slice - compute matrix multiplication
                ref_wgrad[g, :, :] = (
                    dy_fp32[m_start : m_start + m_size, :].T
                    @ x_fp32[m_start : m_start + m_size, :]
                )
                non_zero_groups.append(g)
            m_start += m_size

        if output_accum:
            assert wgrad_accum is not None
            ref_wgrad += wgrad_accum

        ref_wgrad = ref_wgrad.to(test_wgrad.dtype)

        # Float16 is slightly less accurate.
        atol = 1e-4 if dtype == torch.bfloat16 else 1e-3
        # Compare groups with non-zero m_size
        if non_zero_groups:
            if test_wgrad.dtype == torch.float32:
                torch.testing.assert_close(
                    test_wgrad[non_zero_groups],
                    ref_wgrad[non_zero_groups],
                    atol=atol,
                    rtol=1e-3,
                )
            else:
                torch.testing.assert_close(
                    test_wgrad[non_zero_groups],
                    ref_wgrad[non_zero_groups],
                    atol=atol,
                    rtol=1e-2,
                )

    @parameterized.expand(
        [
            (*case, dtype)
            for dtype in [torch.bfloat16, torch.float16]
            for case in [
                (1, 257, 256, 128),  # small G
                (2, 257, 256, 128),  # small
                (16, 2049, 2048, 1024),  # large
                (64, 2049, 2048, 1024),  # large G
                (2, 0, 256, 128),  # empty (M=0)
            ]
        ]
    )
    @unittest.skipIf(
        not torch.version.cuda,
        "Skip on AMD: test_grouped_gemm_dgrad not yet supported.",
    )
    def test_grouped_gemm_dgrad(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(hash((G, M, N, K)))

        # Inputs
        dy_half = torch.randn(
            (M, N), dtype=dtype, device=torch.accelerator.current_accelerator()
        )
        w_half = torch.randn(
            (G, N, K),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        m_sizes = BF16Tests.generate_random_splits(G, M)

        sm_margin = 0
        num_sms = (
            torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
        )

        y_half = torch.ops.mslk.bf16bf16bf16_grouped_grad(
            dy_half,
            w_half.permute(0, 2, 1),
            m_sizes.to(torch.int64),
            num_sms=num_sms,
        )

        Y_preallocated = torch.empty(
            (M * K),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        y_half_preallocated = torch.ops.mslk.bf16bf16bf16_grouped_grad(
            dy_half,
            w_half.permute(0, 2, 1),
            m_sizes.to(torch.int64),
            Y_preallocated,
            num_sms=num_sms,
        )

        # Reference
        dy_fp32 = dy_half.to(torch.float32)
        w_fp32 = w_half.to(torch.float32)

        ref_y_fp32 = torch.empty(
            (M, K), dtype=torch.float32, device=torch.accelerator.current_accelerator()
        )
        m_start = 0
        for g, m_size in enumerate(m_sizes.tolist()):
            ref_y_fp32[m_start : m_start + m_size, :] = dy_fp32[
                m_start : m_start + m_size, :
            ] @ w_fp32[g, :, :].view(N, K)
            m_start += m_size
        ref_y_half = ref_y_fp32.to(dtype)

        torch.testing.assert_close(y_half, ref_y_half, atol=1e-3, rtol=1.6e-2)
        torch.testing.assert_close(
            y_half_preallocated, ref_y_half, atol=1e-3, rtol=1.6e-2
        )

    @parameterized.expand(
        [
            (*case, dtype)
            for dtype in [torch.bfloat16, torch.float16]
            for case in [
                (1, 257, 256, 128),  # small G
                (2, 257, 256, 128),  # small
                (16, 2049, 2048, 1024),  # large
                (64, 2049, 2048, 1024),  # large G
                (2, 0, 256, 128),  # empty (M=0)
            ]
        ]
    )
    @unittest.skipIf(
        not torch.version.cuda,
        "Skip on AMD: test_grouped_gemm_fprop not yet supported.",
    )
    def test_grouped_gemm_fprop(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(hash((G, M, N, K)))

        # Inputs
        x_half = torch.randn(
            (M, K), dtype=dtype, device=torch.accelerator.current_accelerator()
        )
        w_half = torch.randn(
            (G, N, K),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        m_sizes = BF16Tests.generate_random_splits(G, M)

        sm_margin = 0
        num_sms = (
            torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
        )

        y_half = torch.ops.mslk.bf16bf16bf16_grouped_stacked(
            x_half, w_half, m_sizes.to(torch.int64), num_sms=num_sms
        )

        Y_preallocated = torch.empty(
            (M * N),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        y_half_Y_preallocated = torch.ops.mslk.bf16bf16bf16_grouped_stacked(
            x_half, w_half, m_sizes.to(torch.int64), Y_preallocated, num_sms=num_sms
        )

        # Reference
        x_fp32 = x_half.to(torch.float32)
        w_fp32 = w_half.to(torch.float32)

        ref_y_fp32 = torch.empty(
            (M, N), dtype=torch.float32, device=torch.accelerator.current_accelerator()
        )
        m_start = 0
        for g, m_size in enumerate(m_sizes.tolist()):
            ref_y_fp32[m_start : m_start + m_size, :] = (
                x_fp32[m_start : m_start + m_size, :] @ w_fp32[g, :, :].view(N, K).T
            )
            m_start += m_size
        ref_y_half = ref_y_fp32.to(dtype)

        torch.testing.assert_close(y_half, ref_y_half, atol=1e-3, rtol=1.6e-2)
        torch.testing.assert_close(
            y_half_Y_preallocated, ref_y_half, atol=1e-3, rtol=1.6e-2
        )

    @parameterized.expand(
        [
            (*case, dtype)
            for dtype in [torch.bfloat16, torch.float16]
            for case in [
                (256, 128, False, True),  # small, all experts zero
                (2048, 1024, False, False),  # large, some experts zero
                (2048, 1024, True, False),  # output_accum
            ]
        ]
    )
    @unittest.skipIf(
        not torch.version.cuda,
        "Skip on AMD: test not yet supported.",
    )
    def test_grouped_gemm_wgrad_zero_token_experts(
        self,
        N: int,
        K: int,
        output_accum: bool,
        all_zero: bool,
        dtype: torch.dtype,
    ) -> None:
        """Test wgrad produces zeros (not NaN) for experts with zero tokens.

        Covers two cases:
        - all_zero=True:  total_M == 0, all experts receive zero tokens.
        - all_zero=False: total_M > 0, some experts receive tokens, others don't.
        """
        G = 8
        if all_zero:
            M = 0
            m_sizes = torch.zeros(
                G, dtype=torch.int32, device=torch.accelerator.current_accelerator()
            )
            zero_experts = list(range(G))
        else:
            M = 100
            m_sizes = torch.zeros(
                G, dtype=torch.int32, device=torch.accelerator.current_accelerator()
            )
            # Experts 0, 3, 5 get tokens; others get zero
            m_sizes[0] = 40
            m_sizes[3] = 30
            m_sizes[5] = 30
            zero_experts = [1, 2, 4, 6, 7]

        torch.manual_seed(hash((G, M, N, K)))
        dy_bf16 = torch.randn(
            (M, N), dtype=dtype, device=torch.accelerator.current_accelerator()
        )
        x_bf16 = torch.randn(
            (M, K), dtype=dtype, device=torch.accelerator.current_accelerator()
        )

        if output_accum:
            wgrad_accum = torch.randn(
                (G, N, K),
                dtype=torch.float32,
                device=torch.accelerator.current_accelerator(),
            )
        else:
            wgrad_accum = None

        sm_margin = 0
        num_sms = (
            torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
        )

        # Poison the CUDA caching allocator by running the same kernel with all
        # experts having tokens, producing a non-zero output buffer of exactly
        # G*N*K elements. When freed, the caching allocator will reuse this
        # dirty block for the next at::empty(G*N*K, ...) call inside the kernel.
        # Zero-token experts then retain stale non-zero values.
        warmup_M = G * 10
        warmup_dy = torch.randn(
            (warmup_M, N),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        warmup_x = torch.randn(
            (warmup_M, K),
            dtype=dtype,
            device=torch.accelerator.current_accelerator(),
        )
        warmup_m_sizes = torch.full(
            (G,), 10, dtype=torch.int64, device=torch.accelerator.current_accelerator()
        )
        warmup_result = torch.ops.mslk.bf16bf16bf16_grouped_wgrad(
            warmup_dy,
            warmup_x,
            warmup_m_sizes,
            output_accum=False,
            num_sms=num_sms,
        )
        assert warmup_result.abs().sum() > 0, "Warmup should produce non-zero output"
        del warmup_result, warmup_dy, warmup_x, warmup_m_sizes

        test_wgrad = torch.ops.mslk.bf16bf16bf16_grouped_wgrad(
            dy_bf16,
            x_bf16,
            m_sizes.to(torch.int64),
            output=wgrad_accum.clone() if output_accum else None,
            output_accum=output_accum,
            num_sms=num_sms,
        )

        self.assertFalse(
            torch.isnan(test_wgrad).any().item(),
            f"NaN found in wgrad (all_zero={all_zero})",
        )

        for idx in zero_experts:
            if output_accum:
                expected = wgrad_accum[idx].to(test_wgrad.dtype)
            else:
                expected = torch.zeros(
                    (N, K),
                    dtype=test_wgrad.dtype,
                    device=torch.accelerator.current_accelerator(),
                )
            torch.testing.assert_close(
                test_wgrad[idx],
                expected,
                atol=1e-4,
                rtol=1e-3,
                msg=f"Expert {idx} (zero tokens, all_zero={all_zero}) has unexpected gradient values",
            )


@unittest.skipIf(
    not SUPPORTS_NVFP4, "Skip if NVFP4Tests is not supported on this device."
)
class NVFP4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (1, 256, 2048),  # small
            (250, 512, 2048),  # medium
            (250, 1024, 3584),  # large
        ]
    )
    def test_gemm(self, M: int, N: int, K: int) -> None:
        A = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        B = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        global_scales, a_global_scales, b_global_scales = get_nvfp4_global_scales_naive(
            [A],
            [B],
        )
        aqs, a_scales = quantize_nvfp4_naive([A], a_global_scales)
        bqs, b_scales = quantize_nvfp4_naive([B], b_global_scales)

        out_nvfp4 = torch.ops.mslk.f4f4bf16(
            aqs[0], bqs[0], a_scales[0], b_scales[0], None, global_scales[0]
        )
        out_bf16 = A @ B.t()

        torch.testing.assert_close(out_nvfp4, out_bf16, atol=5.0e-2, rtol=5.0e-2)

    @parameterized.expand(
        [
            (1, 256, 256, 2048),  # small
            (4, 500, 1024, 2048),  # medium
            (16, 3500, 6144, 3584),  # large
        ]
    )
    def test_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for _ in range(G)
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]
        offsets = torch.arange(M, G * (M + 1), M, dtype=torch.int32, device=self.device)

        global_scales, x_global_scales, w_global_scales = get_nvfp4_global_scales_naive(
            XS, WS
        )
        xqs, x_scales = quantize_nvfp4_naive(XS, x_global_scales)
        wqs, w_scales = quantize_nvfp4_naive(WS, w_global_scales)

        xq = torch.cat(xqs, dim=0).view(torch.float4_e2m1fn_x2)
        wq = torch.stack(wqs, dim=0).view(torch.float4_e2m1fn_x2)
        x_scale = torch.stack(x_scales, dim=0).view(torch.float8_e4m3fn)
        w_scale = torch.stack(w_scales, dim=0).view(torch.float8_e4m3fn)
        global_scale = torch.stack(global_scales, dim=0)

        X = torch.cat(XS, dim=0)
        W = torch.stack(WS, dim=0)

        out_bf16 = torch._grouped_mm(
            X, W.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16
        )
        out_nvfp4 = torch.ops.mslk.f4f4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets, None, global_scale
        )
        self.assertTrue(out_nvfp4.isfinite().all(), "output contains non-finite values")

        torch.testing.assert_close(out_nvfp4, out_bf16, atol=5.0e-2, rtol=6.0e-2)

    @parameterized.expand(
        [
            (1, 256, 256, 2048),  # small
            (4, 500, 1024, 2048),  # medium
            (16, 3500, 6144, 3584),  # large
            (64, 3500, 6144, 3584),  # large G
        ]
    )
    def test_grouped_gemm_2d_2d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for _ in range(G)
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]
        offsets = torch.arange(K, G * (K + 1), K, dtype=torch.int32, device=self.device)

        global_scales, x_global_scales, w_global_scales = get_nvfp4_global_scales_naive(
            XS, WS
        )
        xqs, x_scales = quantize_nvfp4_naive(XS, x_global_scales)
        wqs, w_scales = quantize_nvfp4_naive(WS, w_global_scales)

        xq = torch.cat(xqs, dim=1).view(torch.float4_e2m1fn_x2)
        wq = torch.cat(wqs, dim=1).view(torch.float4_e2m1fn_x2)
        x_scale = torch.stack(x_scales, dim=0).view(torch.float8_e4m3fn)
        w_scale = torch.stack(w_scales, dim=0).view(torch.float8_e4m3fn)
        global_scale = torch.stack(global_scales, dim=0)

        X = torch.cat(XS, dim=1)
        W = torch.cat(WS, dim=1)

        out_bf16 = torch._grouped_mm(
            X, W.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16
        )
        out_nvfp4 = torch.ops.mslk.f4f4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets, None, global_scale
        )
        self.assertTrue(out_nvfp4.isfinite().all(), "output contains non-finite values")

        torch.testing.assert_close(out_nvfp4, out_bf16, atol=5.0e-2, rtol=6.0e-2)

    @unittest.skipIf(
        not SUPPORTS_NVFP4_ULTRA,
        "Skip if NVFP4 ultra grouped GEMM is not supported on this device.",
    )
    def test_ultra_grouped_gemm_2d_3d(self) -> None:
        G = 2
        N = 512
        K = 1024
        m_sizes_list = [128, 256]
        m_sizes = torch.tensor(m_sizes_list, dtype=torch.int64, device=self.device)
        offsets = torch.cumsum(m_sizes, dim=0).to(torch.int32)

        M = sum(m_sizes_list)
        X = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        W = torch.randn((G, N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        w_cat = W.reshape(G * N, K).contiguous()
        w_m_sizes = torch.full((G,), N, dtype=torch.int64, device=self.device)
        w_global_scale, _ = calculate_group_max(w_cat, w_m_sizes)
        wq, w_scale_2d = nvfp4_quantize_stacked(
            w_m_sizes,
            w_cat,
            w_global_scale,
        )
        wq = wq.view(G, N, K // 2)
        padded_N = ((N + 127) // 128) * 128
        w_scale = w_scale_2d[: G * padded_N].view(G, padded_N, -1)
        w_global_scale_inv = torch.reciprocal(w_global_scale)

        xq, x_scale, x_token_scale_inv = nvfp4_quantize_stacked_with_token_scale(
            m_sizes,
            X,
        )

        out_nvfp4 = torch.ops.mslk.f4f4bf16_ultra_grouped_mm(
            xq,
            wq.transpose(-2, -1),
            x_scale,
            w_scale,
            offsets,
            x_token_scale_inv,
            w_global_scale_inv,
        )
        self.assertTrue(out_nvfp4.isfinite().all(), "output contains non-finite values")

        refs = []
        start = 0
        for group_idx, m_size in enumerate(m_sizes_list):
            end = start + m_size
            refs.append(X[start:end] @ W[group_idx].t())
            start = end
        out_bf16 = torch.cat(refs, dim=0).to(torch.bfloat16)

        torch.testing.assert_close(out_nvfp4, out_bf16, atol=5.0e-2, rtol=6.0e-2)

    @unittest.skipIf(
        not SUPPORTS_NVFP4_ULTRA,
        "Skip if NVFP4 ultra grouped GEMM is not supported on this device.",
    )
    def test_ultra_grouped_gemm_meta(self) -> None:
        G = 2
        M = 384
        N = 512
        K = 1024
        xq = torch.empty((M, K // 2), dtype=torch.float4_e2m1fn_x2, device="meta")
        wq = torch.empty((G, K // 2, N), dtype=torch.float4_e2m1fn_x2, device="meta")
        x_scale = torch.empty(
            (M + G * 127, K // 16), dtype=torch.float8_e4m3fn, device="meta"
        )
        w_scale = torch.empty((G, N, K // 16), dtype=torch.float8_e4m3fn, device="meta")
        offsets = torch.empty((G,), dtype=torch.int32, device="meta")
        x_global_scale = torch.empty((M,), dtype=torch.float32, device="meta")
        w_global_scale = torch.empty((G,), dtype=torch.float32, device="meta")

        out = torch.ops.mslk.f4f4bf16_ultra_grouped_mm(
            xq,
            wq,
            x_scale,
            w_scale,
            offsets,
            x_global_scale,
            w_global_scale,
        )

        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.device.type, "meta")


@unittest.skipIf(not SUPPORTS_MXFP4, "Skip if MXFP4 is not supported")
class MXFP4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (1, 256, 2048),  # small
            (250, 512, 2048),  # medium
            (250, 1024, 3584),  # large
        ]
    )
    def test_gemm(self, M: int, N: int, K: int) -> None:
        A = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        B = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        aq, a_scale = triton_quantize_mx4_unpack(A)
        bq, b_scale = triton_quantize_mx4_unpack(B)

        out_mxfp4 = torch.ops.mslk.f4f4bf16(aq, bq, a_scale, b_scale)
        out_bf16 = A @ B.t()

        torch.testing.assert_close(out_mxfp4, out_bf16, atol=8.0e-2, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 256, 256, 2048),  # small
            (4, 500, 1024, 2048),  # medium
            (16, 3500, 6144, 3584),  # large
            (64, 3500, 6144, 3584),  # large G
        ]
    )
    def test_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for _ in range(G)
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]
        offsets = torch.arange(M, G * (M + 1), M, dtype=torch.int32, device=self.device)

        wqs = []
        xqs = []
        w_scales = []
        x_scales = []

        for x, w in zip(XS, WS):
            xq, x_scale = triton_quantize_mx4_unpack(x)
            wq, w_scale = triton_quantize_mx4_unpack(w)

            xqs.append(xq)
            wqs.append(wq)
            x_scales.append(x_scale)
            w_scales.append(w_scale)

        xq = torch.cat(xqs, dim=0).view(torch.float4_e2m1fn_x2)
        wq = torch.stack(wqs, dim=0).view(torch.float4_e2m1fn_x2)
        x_scale = torch.stack(x_scales, dim=0).view(torch.float8_e8m0fnu)
        w_scale = torch.stack(w_scales, dim=0).view(torch.float8_e8m0fnu)

        X = torch.cat(XS, dim=0)
        W = torch.stack(WS, dim=0)

        out_bf16 = torch._grouped_mm(
            X, W.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16
        )
        out_mxfp4 = torch.ops.mslk.f4f4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets
        )
        self.assertTrue(out_mxfp4.isfinite().all(), "output contains non-finite values")

        torch.testing.assert_close(out_mxfp4, out_bf16, atol=8.0e-2, rtol=8.0e-2)

    @parameterized.expand(
        [
            (1, 256, 256, 2048),  # small
            (4, 500, 1024, 2048),  # medium
            (16, 3500, 6144, 3584),  # large
        ]
    )
    def test_grouped_gemm_2d_2d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for _ in range(G)
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]
        offsets = torch.arange(K, G * (K + 1), K, dtype=torch.int32, device=self.device)

        wqs = []
        xqs = []
        w_scales = []
        x_scales = []

        for x, w in zip(XS, WS):
            xq, x_scale = triton_quantize_mx4_unpack(x)
            wq, w_scale = triton_quantize_mx4_unpack(w)

            xqs.append(xq)
            wqs.append(wq)
            x_scales.append(x_scale)
            w_scales.append(w_scale)

        xq = torch.cat(xqs, dim=1).view(torch.float4_e2m1fn_x2)
        wq = torch.cat(wqs, dim=1).view(torch.float4_e2m1fn_x2)
        x_scale = torch.stack(x_scales, dim=0).view(torch.float8_e8m0fnu)
        w_scale = torch.stack(w_scales, dim=0).view(torch.float8_e8m0fnu)

        X = torch.cat(XS, dim=1)
        W = torch.cat(WS, dim=1)

        out_bf16 = torch._grouped_mm(
            X, W.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16
        )
        out_mxfp4 = torch.ops.mslk.f4f4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets
        )
        self.assertTrue(out_mxfp4.isfinite().all(), "output contains non-finite values")

        torch.testing.assert_close(out_mxfp4, out_bf16, atol=8.0e-2, rtol=8.0e-2)


@unittest.skipIf(not SUPPORTS_MXFP4, "Skip if MXFP4 is not supported")
class MXFP4BlockSize16Tests(unittest.TestCase):
    """
    Tests for MXFP4_16 format: MXFP4 with 1x16 block size
    (16 elements per scale factor).
    This is a hybrid format using MXFP4's E8M0 scale factors with NVFP4's block size.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (1, 256, 2048),  # small
            (250, 512, 2048),  # medium
            (250, 1024, 3584),  # large
        ]
    )
    def test_gemm(self, M: int, N: int, K: int) -> None:
        A = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        B = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        # Use group_size=16 for MXFP4_16 (1x16 block size)
        aq, a_scale = triton_quantize_mx4_unpack(A, group_size=16)
        bq, b_scale = triton_quantize_mx4_unpack(B, group_size=16)

        # Call f4f4bf16 with mxfp4_block_size=16 for MXFP4_16 format
        out_mxfp4_16 = torch.ops.mslk.f4f4bf16(aq, bq, a_scale, b_scale, None, None, 16)
        out_bf16 = A @ B.t()

        torch.testing.assert_close(out_mxfp4_16, out_bf16, atol=8.0e-2, rtol=8.0e-2)

    def test_mxfp4_16_vs_mxfp4_32_block_size(self) -> None:
        """
        Test that MXFP4_16 has better accuracy than MXFP4_32 for data with
        fine-grained scale variations within 32-element blocks.

        This test uses values perfectly representable by MXFP4 format:
        - FP4 (E2M1) values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
        - E8M0 scales: powers of 2

        Perfectly representable values = FP4_value × E8M0_scale:
        - High value: 6 × 1024 = 6144
        - Low value: 6 × 128 = 768

        With 16-element blocks (MXFP4_16):
        - Each group can use its own scale, so both values are exact.

        With 32-element blocks (MXFP4_32):
        - A single scale must cover both values, causing quantization error.
        """
        M, N, K = 1, 256, 512

        # Values perfectly representable by MXFP4: FP4_max(6) × E8M0_scale
        high_val = 6.0 * 1024  # 6144, needs scale=1024
        low_val = 6.0 * 128  # 768, needs scale=128

        # Create A matrix with alternating high/low magnitude 16-element groups
        A = torch.zeros((M, K), dtype=torch.bfloat16, device=self.device)
        for i in range(K // 32):
            A[:, i * 32 : i * 32 + 16] = high_val
            A[:, i * 32 + 16 : i * 32 + 32] = low_val

        # Create B matrix with inverse pattern
        B = torch.zeros((N, K), dtype=torch.bfloat16, device=self.device)
        for i in range(K // 32):
            B[:, i * 32 : i * 32 + 16] = low_val
            B[:, i * 32 + 16 : i * 32 + 32] = high_val

        # Compute bf16 reference
        out_bf16 = A @ B.t()

        # --- MXFP4_32 (block_size=32) ---
        # Single scale for 32 elements cannot represent both 6144 and 768 exactly
        aq_32, a_scale_32 = triton_quantize_mx4_unpack(A, group_size=32)
        bq_32, b_scale_32 = triton_quantize_mx4_unpack(B, group_size=32)
        out_mxfp4_32 = torch.ops.mslk.f4f4bf16(
            aq_32, bq_32, a_scale_32, b_scale_32, None, None, 32
        )

        # --- MXFP4_16 (block_size=16) ---
        # Each 16-element group uses its own scale, so values are exact
        aq_16, a_scale_16 = triton_quantize_mx4_unpack(A, group_size=16)
        bq_16, b_scale_16 = triton_quantize_mx4_unpack(B, group_size=16)
        out_mxfp4_16 = torch.ops.mslk.f4f4bf16(
            aq_16, bq_16, a_scale_16, b_scale_16, None, None, 16
        )

        # Calculate errors
        error_mxfp4_32 = (out_mxfp4_32 - out_bf16).abs().mean().item()
        error_mxfp4_16 = (out_mxfp4_16 - out_bf16).abs().mean().item()

        # MXFP4_16 should produce EXACT results (values are perfectly representable)
        self.assertTrue(
            torch.equal(out_mxfp4_16, out_bf16),
            "MXFP4_16 output should be exactly equal to bf16 baseline "
            "when using perfectly representable values",
        )

        # MXFP4_16 should have better accuracy (lower error) than MXFP4_32
        # MXFP4_32 should have significant error due to block size mismatch
        self.assertLess(
            error_mxfp4_16,
            error_mxfp4_32,
            f"MXFP4_16 error ({error_mxfp4_16:.6f}) should be less than "
            f"MXFP4_32 error ({error_mxfp4_32:.6f}) for data with "
            f"fine-grained scale variations within 32-element blocks",
        )

        # Verify outputs are finite
        self.assertTrue(
            out_mxfp4_32.isfinite().all(), "MXFP4_32 output contains non-finite values"
        )
        self.assertTrue(
            out_mxfp4_16.isfinite().all(), "MXFP4_16 output contains non-finite values"
        )

    def test_mxfp4_16_vs_nvfp4_scale_range(self) -> None:
        """
        Test that MXFP4_16 has better accuracy than NVFP4 (with global_scale=1.0)
        for data requiring scales beyond E4M3 range.

        This test uses values perfectly representable by MXFP4:
        - Value: 6 × 2048 = 12288 (FP4_max × E8M0_scale)

        NVFP4 max with global_scale=1.0:
        - Max = FP4_max × E4M3_max = 6 × 448 = 2688
        - Value 12288 > 2688, so NVFP4 will clip and lose precision.

        MXFP4_16 can represent 12288 exactly (scale=2048, FP4=6).
        """
        from mslk.quantize.triton.fp4_quantize import triton_quantize_nvfp4

        M, N, K = 1, 256, 512

        # Value perfectly representable by MXFP4: 6 × 2048 = 12288
        # This exceeds NVFP4 max of 6 × 448 = 2688
        val = 6.0 * 2048  # 12288

        A = torch.full((M, K), val, dtype=torch.bfloat16, device=self.device)
        B = torch.full((N, K), val, dtype=torch.bfloat16, device=self.device)

        # Compute bf16 reference
        out_bf16 = A @ B.t()

        # --- NVFP4 with global_scale=1.0 ---
        # Max representable = 6 × 448 × 1.0 = 2688
        # Value 12288 exceeds this, causing clipping/precision loss
        global_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        aq_nvfp4, a_scale_nvfp4 = triton_quantize_nvfp4(A, global_scale)
        bq_nvfp4, b_scale_nvfp4 = triton_quantize_nvfp4(B, global_scale)
        out_nvfp4 = torch.ops.mslk.f4f4bf16(
            aq_nvfp4, bq_nvfp4, a_scale_nvfp4, b_scale_nvfp4, None, global_scale
        )

        # --- MXFP4_16 (block_size=16) ---
        # E8M0 scale=2048, FP4=6 → value=12288 exactly
        aq_16, a_scale_16 = triton_quantize_mx4_unpack(A, group_size=16)
        bq_16, b_scale_16 = triton_quantize_mx4_unpack(B, group_size=16)
        out_mxfp4_16 = torch.ops.mslk.f4f4bf16(
            aq_16, bq_16, a_scale_16, b_scale_16, None, None, 16
        )

        # Calculate errors
        error_nvfp4 = (out_nvfp4 - out_bf16).abs().mean().item()
        error_mxfp4_16 = (out_mxfp4_16 - out_bf16).abs().mean().item()

        # MXFP4_16 should produce EXACT results (values are perfectly representable)
        self.assertTrue(
            torch.equal(out_mxfp4_16, out_bf16),
            "MXFP4_16 output should be exactly equal to bf16 baseline "
            "when using perfectly representable values",
        )

        # MXFP4_16 should have better accuracy (lower error) than NVFP4
        # NVFP4 should have significant error due to clipping at 2688
        self.assertLess(
            error_mxfp4_16,
            error_nvfp4,
            f"MXFP4_16 error ({error_mxfp4_16:.6f}) should be less than "
            f"NVFP4 error ({error_nvfp4:.6f}) for value 12288 exceeding "
            f"NVFP4 max (6 × 448 = 2688) with global_scale=1.0",
        )

        # Verify outputs are finite
        self.assertTrue(
            out_nvfp4.isfinite().all(), "NVFP4 output contains non-finite values"
        )
        self.assertTrue(
            out_mxfp4_16.isfinite().all(), "MXFP4_16 output contains non-finite values"
        )


@unittest.skipIf(
    not SUPPORTS_MXFP4 and not is_gfx950(),
    "Skip if MXFP4 is not supported (ROCm requires gfx950+)",
)
class MX8MX4Tests(unittest.TestCase):
    """Tests for the mixed MX8 x MX4 GEMM kernel (mx8mx4bf16)."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (1, 256, 2048),  # small
            (64, 512, 2048),  # medium
            (256, 1024, 4096),  # large
        ]
    )
    def test_gemm(self, M: int, N: int, K: int) -> None:
        from mslk.gemm.triton.fp8_gemm import to_mxfp8

        A = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        B = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        # Quantize A to MX8 with blocked scale layout
        (a_scale_raw, aq) = to_mxfp8(A)
        a_scale = _to_blocked(a_scale_raw.view(torch.int8).reshape(M, -1)).view(
            torch.uint8
        )

        # Quantize B to MX4
        bq, b_scale = triton_quantize_mx4_unpack(B)
        bq = bq.view(torch.float4_e2m1fn_x2)

        out_mx8mx4 = torch.ops.mslk.mx8mx4bf16(aq, bq, a_scale, b_scale)
        out_bf16 = A @ B.t()

        # No NaN or Inf
        self.assertFalse(out_mx8mx4.isnan().any().item(), "Output contains NaN")
        self.assertFalse(out_mx8mx4.isinf().any().item(), "Output contains Inf")

        # Mixed MX8xMX4 has higher tolerance than MX4xMX4 due to mixed precision
        torch.testing.assert_close(out_mx8mx4, out_bf16, atol=1.0e-1, rtol=1.0e-1)

    @parameterized.expand(
        [
            (1, 256, 256, 2048),  # small
            (4, 500, 1024, 2048),  # medium
            (16, 3500, 6144, 3584),  # large
            (64, 3500, 6144, 3584),  # large G
        ]
    )
    def test_mx8mx4_grouped_gemm_2d_3d(
        self,
        G: int,
        M: int,
        N: int,
        K: int,
    ) -> None:
        from mslk.gemm.triton.fp8_gemm import to_mxfp8

        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for _ in range(G)
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]
        offsets = torch.arange(
            M,
            G * M + 1,
            M,
            dtype=torch.int32,
            device=self.device,
        )

        xqs = []
        x_scales = []
        wqs = []
        w_scales = []
        for x, w in zip(XS, WS):
            x_scale, xq = to_mxfp8(x)
            x_scale = _to_blocked(
                x_scale.view(torch.int8).reshape(x.shape[0], -1)
            ).view(torch.uint8)
            wq, w_scale = triton_quantize_mx4_unpack(w)

            xqs.append(xq)
            x_scales.append(x_scale)
            wqs.append(wq.view(torch.float4_e2m1fn_x2))
            w_scales.append(w_scale)

        xq = torch.cat(xqs, dim=0).contiguous()
        x_scale = torch.cat(x_scales, dim=0).contiguous().reshape(-1, K // 32)
        wq = torch.stack(wqs, dim=0).contiguous()
        w_scale = torch.stack(w_scales, dim=0).contiguous()

        X = torch.cat(XS, dim=0)
        W = torch.stack(WS, dim=0)

        out_bf16 = torch._grouped_mm(
            X, W.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16
        )
        out_mx8mx4 = torch.ops.mslk.mx8mx4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets
        )
        self.assertTrue(out_mx8mx4.isfinite().all(), "output has non-finite values")

        torch.testing.assert_close(out_mx8mx4, out_bf16, atol=6.0e-2, rtol=6.0e-2)

    def test_mx8mx4_grouped_gemm_2d_3d_empty_groups(self) -> None:
        from mslk.gemm.triton.fp8_gemm import to_mxfp8

        G = 8
        N = 1024
        K = 2048
        m_sizes_list = [0, 1, 4, 0, 32, 0, 8, 16]
        offsets = torch.tensor(
            m_sizes_list, dtype=torch.int32, device=self.device
        ).cumsum(0, dtype=torch.int32)

        XS = [
            torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
            for M in m_sizes_list
        ]
        WS = [
            torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01
            for _ in range(G)
        ]

        xqs = []
        x_scales = []
        wqs = []
        w_scales = []
        refs = []
        for x, w in zip(XS, WS):
            if x.numel() > 0:
                x_scale, xq = to_mxfp8(x)
                x_scale = _to_blocked(
                    x_scale.view(torch.int8).reshape(x.shape[0], -1)
                ).view(torch.uint8)
                xqs.append(xq)
                x_scales.append(x_scale)
                refs.append(x @ w.t())
            wq, w_scale = triton_quantize_mx4_unpack(w)
            wqs.append(wq.view(torch.float4_e2m1fn_x2))
            w_scales.append(w_scale)

        xq = torch.cat(xqs, dim=0).contiguous()
        x_scale = torch.cat(x_scales, dim=0).contiguous().reshape(-1, K // 32)
        wq = torch.stack(wqs, dim=0).contiguous()
        w_scale = torch.stack(w_scales, dim=0).contiguous()

        out_bf16 = torch.cat(refs, dim=0).to(torch.bfloat16)
        out_mx8mx4 = torch.ops.mslk.mx8mx4bf16_grouped_mm(
            xq, wq.transpose(-2, -1), x_scale, w_scale, offsets
        )
        self.assertTrue(out_mx8mx4.isfinite().all(), "output has non-finite values")

        torch.testing.assert_close(out_mx8mx4, out_bf16, atol=6.0e-2, rtol=6.0e-2)


@unittest.skipIf(not SUPPORTS_MXFP4, "Skip if MXFP4 is not supported")
class MX8MX6Tests(unittest.TestCase):
    """Tests for the mixed MX8 x MX6 CUTLASS GEMM kernel (mx8mx6bf16)."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (1, 256, 2048),  # small
            (64, 512, 2048),  # medium
            (256, 1024, 4096),  # large
        ]
    )
    def test_gemm(self, M: int, N: int, K: int) -> None:
        from mslk.gemm.triton.fp8_gemm import to_mxfp8

        A = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        B = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        # Quantize A to MX8 with blocked scale layout
        (a_scale_raw, aq) = to_mxfp8(A)
        a_scale = _to_blocked(a_scale_raw.view(torch.int8).reshape(M, -1)).view(
            torch.uint8
        )

        # Quantize B: use MX8 quantization for scale factors (E8M0 BS32),
        # then create E2M3 weight data from the MX8 data.
        # E2M3 is stored as 1 byte per element (6-bit value in uint8).
        # We use the FP8 data masked to 6 bits as a proxy for E2M3.
        (b_scale_raw, bq_fp8) = to_mxfp8(B)
        b_scale = _to_blocked(b_scale_raw.view(torch.int8).reshape(N, -1)).view(
            torch.uint8
        )
        bq = bq_fp8.view(torch.uint8) & 0x3F  # mask to 6 bits for E2M3 range

        out_mx8mx6 = torch.ops.mslk.mx8mx6bf16(aq, bq, a_scale, b_scale)

        # Smoke test: no NaN or Inf
        self.assertFalse(out_mx8mx6.isnan().any().item(), "Output contains NaN")
        self.assertFalse(out_mx8mx6.isinf().any().item(), "Output contains Inf")
        # Output shape check
        self.assertEqual(out_mx8mx6.shape, (M, N))


@unittest.skipIf(not SUPPORTS_MXFP4, "Skip if block-scaled GEMM is not supported")
class MX6MX6Tests(unittest.TestCase):
    """Tests for the symmetric MX6 x MX6 CUTLASS GEMM kernel (mx6mx6bf16)."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.accelerator.current_accelerator()

    @parameterized.expand(
        [
            (1, 256, 2048),  # small
            (64, 512, 2048),  # medium
            (256, 1024, 4096),  # large
        ]
    )
    def test_gemm(self, M: int, N: int, K: int) -> None:
        # Both A and B are random uint8 bytes packed at 6 bits/element. The
        # mx6mx6bf16 kernel derives K = XQ.size(1) * 4 / 3 from packed bytes,
        # so passing unpacked (M, K)-shape uint8 (e.g. via to_mxfp8(...) & 0x3F
        # like the asymmetric mx8mx6 test) would silently corrupt K. No Python
        # MX6 quantizer is wired up, so this test only validates dispatch +
        # kernel execution (no NaN/Inf, correct shape/dtype).
        A_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device=self.device) * 0.1
        B_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device=self.device) * 0.01

        # MX6 packed weights for A and B: K elements at 6 bits each = K*6/8 bytes.
        aq_6 = torch.randint(
            0, 256, (M, K * 6 // 8), dtype=torch.uint8, device=self.device
        )
        bq_6 = torch.randint(
            0, 256, (N, K * 6 // 8), dtype=torch.uint8, device=self.device
        )
        # Reuse MX4 block-scale layout (E8M0, block size 32) for both operands.
        _, a_scale = triton_quantize_mx4_unpack(A_bf16)
        _, b_scale = triton_quantize_mx4_unpack(B_bf16)

        out_mx6mx6 = torch.ops.mslk.mx6mx6bf16(aq_6, bq_6, a_scale, b_scale)

        self.assertFalse(out_mx6mx6.isnan().any().item(), "Output contains NaN")
        self.assertFalse(out_mx6mx6.isinf().any().item(), "Output contains Inf")
        self.assertEqual(out_mx6mx6.shape, torch.Size([M, N]))
        self.assertEqual(out_mx6mx6.dtype, torch.bfloat16)


@unittest.skipIf(not supports_int8(), "Requires CUDA SM80+ or ROCm")
class RocmInt8GemmTests(unittest.TestCase):
    """Correctness tests for the Triton INT8 GEMM kernel.

    On CUDA the tests verify parity with the existing CUTLASS i8i8bf16 path.
    On ROCm they serve as the primary correctness gate since there is no
    CUTLASS reference — output is compared against a float32 reference computed
    with torch.mm.
    """

    device: torch.device

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda")
        # Import Triton kernels only when the test class is actually instantiated.
        from mslk.gemm.triton.int8_gemm import (  # noqa: F401
            i8i8bf16_dynamic_triton,
            i8i8bf16_triton,
        )

        cls.i8i8bf16_triton = staticmethod(i8i8bf16_triton)
        cls.i8i8bf16_dynamic_triton = staticmethod(i8i8bf16_dynamic_triton)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_int8_pair(
        M: int, N: int, K: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        XQ = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=device)
        WQ = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=device)
        return XQ, WQ

    @staticmethod
    def _reference_bf16(
        XQ: torch.Tensor, WQ: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """Float32 reference: cast to float, matmul, scale, cast to bf16."""
        return (XQ.float() @ WQ.float().T * scale).bfloat16()

    # ------------------------------------------------------------------
    # Static-scale variant (i8i8bf16)
    # ------------------------------------------------------------------

    def _run_static(self, M: int, N: int, K: int, scale: float = 0.01) -> None:
        XQ, WQ = self._make_int8_pair(M, N, K, self.device)
        ref = self._reference_bf16(XQ, WQ, scale)
        out = self.i8i8bf16_triton(XQ, WQ, scale)
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=1.0, rtol=1e-2)

    def test_static_scale_small(self) -> None:
        self._run_static(M=1, N=1024, K=1024)

    def test_static_scale_decode(self) -> None:
        self._run_static(M=128, N=4096, K=1024)

    def test_static_scale_prefill(self) -> None:
        self._run_static(M=256, N=4096, K=8192)

    def test_static_scale_large(self) -> None:
        self._run_static(M=256, N=8192, K=8192)

    def test_static_scale_non_power_of_two_K(self) -> None:
        # K not a multiple of BLOCK_K → boundary masking must be correct.
        self._run_static(M=64, N=512, K=384)

    # ------------------------------------------------------------------
    # Dynamic-scale variant (i8i8bf16_dynamic)
    # ------------------------------------------------------------------

    def _run_dynamic(self, M: int, N: int, K: int, scale: float = 0.01) -> None:
        XQ, WQ = self._make_int8_pair(M, N, K, self.device)
        scale_tensor = torch.tensor(scale, dtype=torch.float32, device=self.device)
        ref = self._reference_bf16(XQ, WQ, scale)
        out = self.i8i8bf16_dynamic_triton(XQ, WQ, scale_tensor)
        self.assertEqual(out.shape, (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=1.0, rtol=1e-2)

    def test_dynamic_scale_small(self) -> None:
        self._run_dynamic(M=1, N=1024, K=1024)

    def test_dynamic_scale_decode(self) -> None:
        self._run_dynamic(M=128, N=4096, K=1024)

    def test_dynamic_scale_prefill(self) -> None:
        self._run_dynamic(M=256, N=4096, K=8192)

    # ------------------------------------------------------------------
    # Parity with CUDA CUTLASS path (CUDA only)
    # ------------------------------------------------------------------

    @unittest.skipIf(not torch.version.cuda, "CUTLASS parity only tested on CUDA")
    def test_parity_with_cutlass_static(self) -> None:
        M, N, K = 128, 2048, 4096
        scale = 0.01
        XQ, WQ = self._make_int8_pair(M, N, K, self.device)
        cutlass_out = torch.ops.mslk.i8i8bf16(XQ, WQ, scale, 1)
        triton_out = self.i8i8bf16_triton(XQ, WQ, scale)
        torch.testing.assert_close(cutlass_out, triton_out, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(not torch.version.cuda, "CUTLASS parity only tested on CUDA")
    def test_parity_with_cutlass_dynamic(self) -> None:
        M, N, K = 128, 2048, 4096
        scale = 0.01
        XQ, WQ = self._make_int8_pair(M, N, K, self.device)
        scale_tensor = torch.tensor(scale, dtype=torch.float32, device=self.device)
        cutlass_out = torch.ops.mslk.i8i8bf16_dynamic(XQ, WQ, scale_tensor, 1)
        triton_out = self.i8i8bf16_dynamic_triton(XQ, WQ, scale_tensor)
        torch.testing.assert_close(cutlass_out, triton_out, atol=1e-2, rtol=1e-2)

    # ------------------------------------------------------------------
    # torch.ops.mslk dispatch on ROCm
    # ------------------------------------------------------------------

    @unittest.skipIf(not torch.version.hip, "Op dispatch only tested on ROCm")
    def test_ops_dispatch_static(self) -> None:
        M, N, K = 128, 1024, 1024
        scale = 0.01
        XQ, WQ = self._make_int8_pair(M, N, K, self.device)
        out = torch.ops.mslk.i8i8bf16(XQ, WQ, scale, 1)
        ref = self._reference_bf16(XQ, WQ, scale)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=1.0, rtol=1e-2)

    @unittest.skipIf(not torch.version.hip, "Op dispatch only tested on ROCm")
    def test_ops_dispatch_dynamic(self) -> None:
        M, N, K = 128, 1024, 1024
        scale = 0.01
        XQ, WQ = self._make_int8_pair(M, N, K, self.device)
        scale_tensor = torch.tensor(scale, dtype=torch.float32, device=self.device)
        out = torch.ops.mslk.i8i8bf16_dynamic(XQ, WQ, scale_tensor, 1)
        ref = self._reference_bf16(XQ, WQ, scale)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=1.0, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
