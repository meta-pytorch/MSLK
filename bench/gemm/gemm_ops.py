# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Keep a registry of all quantize operators.
import abc
import functools
from enum import auto, Enum

import numpy as np
import torch
from mslk.bench.common.utils import BenchOptions, do_bench
from mslk.gemm.triton.fp8_gemm import matmul_fp8_block, matmul_fp8_row, to_mxfp8
from mslk.gemm.triton.grouped_gemm import grouped_gemm, grouped_gemm_fp8_rowwise
from mslk.quantize.shuffle import ck_preshuffle, quantize_int4_preshuffle
from mslk.quantize.triton.fp4_quantize import (
    _to_blocked,
    calculate_group_max,
    mega_fp4_pack,
    mega_fp4_quantize_kernel,
    mega_fp4_unpack,
    triton_quantize_mx4_unpack,
    triton_quantize_nvfp4,
    triton_scale_nvfp4_quant_rms,
    triton_scale_nvfp4_quant_silu,
)
from mslk.quantize.triton.fp8_quantize import (
    quantize_fp8_block,
    quantize_fp8_group,
    quantize_fp8_row,
    scale_fp8_row,
    triton_quantize_fp8_row,
)
from mslk.utils.triton.fp8_utils import get_fp8_constants

try:
    from gen_ai.llm_inference.fb.llm.kernel.rms_norm import rms_norm
    from gen_ai.llm_inference.fb.llm.kernel.silu_mul import silu_mul
except ImportError:
    # Above is used for some experiments, but the quantize is not relying on them. Okay to just skip.
    pass

try:
    from tinygemm.utils import group_quantize_tensor

    if torch.cuda.is_available() and torch.version.cuda:
        torch.ops.load_library("//tinygemm:tinygemm")
    TINYGEMM_ENABLED = True
except ImportError:
    TINYGEMM_ENABLED = False

# Marlin currently only is supported only internally at Meta.
try:
    from marlin.quantize import marlin_quantize

    torch.ops.load_library("//ai_codesign/gen_ai/marlin:marlin_ops")
    MARLIN_ENABLED = True
except ImportError:
    MARLIN_ENABLED = False

try:
    from deep_gemm import (
        gemm_fp8_fp8_bf16_nt,
        get_col_major_tma_aligned_tensor,
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
        m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    )

    DEEPGEMM_ENABLED = True
except ImportError:
    DEEPGEMM_ENABLED = False


# Machete is also only supported internally at Meta for now.
try:
    from machete.machete import machete_gemm
    from machete.quantize import machete_quantize_and_pack

    MACHETE_ENABLED = True
except ImportError:
    MACHETE_ENABLED = False


# CuteDSL mixed-input GEMM for Blackwell.
try:
    import cutlass
    from mslk.gemm.blackwell_mixed_input_gemm import int4bf16bf16_gemm

    CUTEDSL_MIXED_INPUT_ENABLED = True
except ImportError:
    CUTEDSL_MIXED_INPUT_ENABLED = False


class Accelerator(Enum):
    NVIDIA_SM90 = auto()
    NVIDIA_SM100 = auto()
    NVIDIA_SM103 = auto()
    AMD_MI300X = auto()


class GemmType(Enum):
    REGULAR = auto()
    GROUPED = auto()


class ComputeDtype(Enum):
    FP32 = auto()
    TF32 = auto()
    BF16 = auto()
    FP8 = auto()
    FP4 = auto()


@functools.cache
def get_current_accelerator() -> Accelerator | None:
    if not torch.cuda.is_available():
        raise Exception("Cannot run gemm_bench without accelerator.")

    if torch.version.hip is not None:
        device_name = torch.cuda.get_device_name()
        if "MI300X" in device_name.upper():
            return Accelerator.AMD_MI300X
    elif torch.version.cuda is not None:
        major, minor = torch.cuda.get_device_capability()
        if major == 9 and minor == 0:
            return Accelerator.NVIDIA_SM90
        elif major == 10 and minor == 0:
            return Accelerator.NVIDIA_SM100
        elif major == 10 and minor == 3:
            return Accelerator.NVIDIA_SM103

    raise Exception("Cannot detect hardware that is supported by gemm_bench.")


gemm_op_registry = []


class GemmOpBase(metaclass=abc.ABCMeta):
    """Helper abstract class to define expected methods of quantize ops."""

    @abc.abstractmethod
    def quantize(self, *args):
        """Function which quantizes inputs."""
        pass

    @abc.abstractmethod
    def compute(self, *args):
        """Function which performs main compute operation."""
        pass

    @abc.abstractmethod
    def quantize_and_compute(self, *args):
        """Function which quantizes inputs and performs main compute operation."""
        pass

    def preprocess(self, *args):
        """Preprocess inputs before benchmarking. These outputs will be passed to quantize."""
        return args

    def benchmark(
        self,
        *args,
        opts: BenchOptions,
        bench_quantize: bool,
    ) -> float:
        """Benchmark runtime of this operator."""
        t = do_bench(
            lambda *a: self.quantize_and_compute(*a)
            if bench_quantize
            else self.compute(*a),
            args,
            opts,
        )
        return t

    @abc.abstractproperty
    def name(self) -> str:
        """Name of the operator."""
        pass

    @abc.abstractproperty
    def supported_accelerators(self) -> set[Accelerator]:
        pass

    @property
    def supported(self) -> bool:
        """Whether this op will run on the current device."""
        accelerator = get_current_accelerator()
        return accelerator in self.supported_accelerators

    @abc.abstractproperty
    def supported_gemm_types(self) -> set[GemmType]:
        pass

    @abc.abstractproperty
    def compute_dtype(self) -> ComputeDtype:
        """The dtype used by tensor cores for the compute."""
        pass


def register_gemm_op(op):
    """Decorator function for assembling all quantize ops."""
    gemm_op_registry.append(op())
    return op


def get_gemm_ops() -> list[GemmOpBase]:
    """Get all registered quantize ops."""
    return gemm_op_registry


@register_gemm_op
class FP32Baseline(GemmOpBase):
    """
    FP32 matmul baseline.
    """

    def quantize(self, x, w):
        if isinstance(x, list):
            x = [i.float() for i in x]
            w = [torch.transpose(i, -2, -1).float() for i in w]
        else:
            x = x.float()
            w = torch.transpose(w, -2, -1).float()
        return x, w

    def compute(self, x, w):
        # Handle both grouped and standard gemm.
        if isinstance(x, list):
            output = []
            for i in range(len(x)):
                output.append(torch.matmul(x[i], w[i]))
            return output
        return torch.matmul(x, w)

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "fp32_baseline"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR, GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP32


@register_gemm_op
class TF32Baseline(GemmOpBase):
    """
    TF32 matmul baseline.
    """

    def quantize(self, x, w):
        if isinstance(x, list):
            x = [i.float() for i in x]
            w = [torch.transpose(i, -2, -1).float() for i in w]
        else:
            x = x.float()
            w = torch.transpose(w, -2, -1).float()
        return x, w

    def compute(self, x, w):
        # Handle both grouped and standard gemm.
        original_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        if isinstance(x, list):
            output = []
            for i in range(len(x)):
                output.append(torch.matmul(x[i], w[i]))
            return output
        out = torch.matmul(x, w)
        torch.set_float32_matmul_precision(original_precision)
        return out

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "tf32_baseline"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR, GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.TF32


@register_gemm_op
class BF16Baseline(GemmOpBase):
    """
    Baseline BF16 matmul.
    """

    def quantize(self, x, w):
        if isinstance(x, list):
            x = [i.bfloat16() for i in x]
            w = [torch.transpose(i, -2, -1).bfloat16() for i in w]
        else:
            x = x.bfloat16()
            w = torch.transpose(w, -2, -1).bfloat16()
        return x, w

    def compute(self, x, w):
        # Handle both grouped and standard gemm.
        if isinstance(x, list):
            output = []
            for i in range(len(x)):
                output.append(torch.matmul(x[i], w[i]))
            return output
        return torch.matmul(x, w)

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "bf16_baseline"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR, GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class ScaledMMBaseline(GemmOpBase):
    """
    Reference FP8 matmul implemented in native torch with cublas or hipblas.
    """

    def __init__(self):
        self.fp8_dtype, _, _, _ = get_fp8_constants()
        self.E4M3_MAX_POS: float = torch.finfo(self.fp8_dtype).max
        self.E5M2_MAX_POS: float = torch.finfo(torch.float8_e5m2).max
        self.FP16_MAX_POS: float = torch.finfo(torch.float16).max
        self.EPS: float = 1e-12
        self.fast_accum = True

    def _amax_to_scale(
        self, amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
    ) -> torch.Tensor:
        # To make scale dtype to be fp32 for accuracy
        amax = amax.float()
        if float8_dtype == self.fp8_dtype:
            # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
            res = self.E4M3_MAX_POS / torch.clamp(amax, min=self.EPS)
        else:  # e5m2
            # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
            res = self.E5M2_MAX_POS / torch.clamp(amax, min=self.EPS)

        # pyre-fixme[7]: Expected `Tensor` but got `Union[float, Tensor]`.
        return res

    def _to_fp8_saturated(
        self, x: torch.Tensor, float8_dtype: torch.dtype
    ) -> torch.Tensor:
        if float8_dtype == torch.float8_e4m3fn:
            x = x.clamp(min=-1 * self.E4M3_MAX_POS, max=self.E4M3_MAX_POS)
        else:
            x = x.clamp(min=-1 * self.E5M2_MAX_POS, max=self.E5M2_MAX_POS)
        return x.to(float8_dtype)

    def _quantize_tensor(self, x):
        x_amax = torch.max(torch.abs(x))
        scale = self._amax_to_scale(x_amax, self.fp8_dtype, x.dtype)
        scaled_x = self._to_fp8_saturated(x * scale, self.fp8_dtype)
        x_inverse_scale = scale.reciprocal()
        return scaled_x, x_inverse_scale

    def quantize(self, x, w):
        xq, x_scale = self._quantize_tensor(x)
        wq, w_scale = self._quantize_tensor(w.t())
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        output = torch._scaled_mm(
            xq,
            wq,
            bias=None,
            out_dtype=torch.bfloat16,
            scale_a=x_scale,
            scale_b=w_scale,
            scale_result=None,
            use_fast_accum=self.fast_accum,
        )
        return output

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "scaled_mm"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class BF16X9Baseline(GemmOpBase):
    """
    FP32 matmul implemented with BF16X9 emulation.
    """

    def quantize(self, x, w):
        if isinstance(x, list):
            x = [i.float() for i in x]
            w = [i.float() for i in w]
        else:
            x = x.float()
            w = w.float()
        return x, w

    def compute(self, x, w):
        # Handle both grouped and standard gemm.
        if isinstance(x, list):
            output = []
            for i in range(len(x)):
                output.append(torch.ops.mslk.bf16x9_gemm(x[i], w[i]))
            return output
        return torch.ops.mslk.bf16x9_gemm(x, w)

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "bf16x9_gemm"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR, GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class ScaledMMRowwise(GemmOpBase):
    def __init__(self):
        self.fast_accum = True
        self.torch_compile = False

    def quantize(self, x, w):
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        return xq, wq.t(), x_scale.unsqueeze(1), w_scale.unsqueeze(0)

    def compute(self, xq, wq, x_scale, w_scale):
        if self.torch_compile:
            f = torch.compile(
                torch._scaled_mm,
                options={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON,CK,CUTLASS,ATEN",
                },
            )
        else:
            f = torch._scaled_mm

        return f(
            xq,
            wq,
            bias=None,
            out_dtype=torch.bfloat16,
            scale_a=x_scale,
            scale_b=w_scale,
            scale_result=None,
            use_fast_accum=self.fast_accum,
        )

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "scaled_mm_rowwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class ScaledMMMXFP8(GemmOpBase):
    def __init__(self):
        self.torch_compile = False

    def quantize(self, x, w):
        x_scale, xq = to_mxfp8(x)
        x_scale = _to_blocked(x_scale)
        w_scale, wq = to_mxfp8(w)
        w_scale = _to_blocked(w_scale)
        return xq, wq.t(), x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        if self.torch_compile:
            f = torch.compile(
                torch._scaled_mm,
                options={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON,CK,CUTLASS,ATEN",
                },
            )
        else:
            f = torch._scaled_mm

        return f(
            xq,
            wq,
            bias=None,
            out_dtype=torch.bfloat16,
            scale_a=x_scale,
            scale_b=w_scale,
        )

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "scaled_mm_mxfp8"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class ScaledMMNVFP4(GemmOpBase):
    def __init__(self):
        self.torch_compile = False

    def quantize(self, x, w):
        x_global_scale = torch.tensor([1.0], device=x.device, dtype=torch.float32)
        w_global_scale = torch.tensor([1.0], device=w.device, dtype=torch.float32)

        xq, x_scale = triton_quantize_nvfp4(x, x_global_scale)
        wq, w_scale = triton_quantize_nvfp4(w, w_global_scale)

        return (
            xq.view(torch.float4_e2m1fn_x2),
            wq.view(torch.float4_e2m1fn_x2),
            x_scale.view(torch.float8_e4m3fn),
            w_scale.view(torch.float8_e4m3fn),
        )

    def compute(self, xq, wq, x_scale, w_scale):
        if self.torch_compile:
            f = torch.compile(
                torch._scaled_mm,
                options={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON,CK,CUTLASS,ATEN",
                },
            )
        else:
            f = torch._scaled_mm

        return f(
            xq,
            wq.t(),
            bias=None,
            out_dtype=torch.bfloat16,
            scale_a=x_scale,
            scale_b=w_scale,
        )

    def quantize_and_compute(self, x, w):
        return self.compute(*self.quantize(x, w))

    @property
    def name(self) -> str:
        return "scaled_mm_nvfp4"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP4


@register_gemm_op
class FP8TensorwiseGemm(GemmOpBase):
    """
    FP8 matmul with tensorwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = torch.ops.mslk.quantize_fp8_per_tensor(x)
        wq, w_scale = torch.ops.mslk.quantize_fp8_per_tensor(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.mslk.f8f8bf16(xq, wq, x_scale * w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cutlass_tensorwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8CublasRowwiseGemm(GemmOpBase):
    """
    FP8 cublas matmul with rowwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = torch.ops.mslk.quantize_fp8_per_row(x)
        wq, w_scale = torch.ops.mslk.quantize_fp8_per_row(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        out = torch.ops.mslk.f8f8bf16_cublas(xq, wq)
        scaled_out = scale_fp8_row(out, x_scale, w_scale)
        return scaled_out

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cublas_rowwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8CublasTensorwiseGemm(GemmOpBase):
    """
    FP8 cublas matmul with tensorwise scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = torch.ops.mslk.quantize_fp8_per_tensor(x)
        wq, w_scale = torch.ops.mslk.quantize_fp8_per_tensor(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.mslk.f8f8bf16_cublas(xq, wq, x_scale * w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale * w_scale)

    @property
    def name(self) -> str:
        return "cublas_tensorwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8RowwiseGemm(GemmOpBase):
    """
    FP8 matmul with rowwise scaling.
    """

    def __init__(self):
        self.fast_accum = True
        self.gemm_op = torch.ops.mslk.f8f8bf16_rowwise

    def preprocess(self, x, w):
        # Prequantize weights.
        if isinstance(w, (list, tuple)):
            wq, w_scale = zip(*[quantize_fp8_row(i) for i in w])
        else:
            wq, w_scale = quantize_fp8_row(w)
            if wq.dim() == 3:
                w_scale = w_scale.view(wq.size(0), -1)
        return x, wq, w_scale

    def quantize(self, x, wq, w_scale):
        # Quantize both input tensors.
        # Handle both grouped and standard gemm.
        if isinstance(x, (list, tuple)):
            xq, x_scale = zip(*[quantize_fp8_row(i) for i in x])
        else:
            xq, x_scale = quantize_fp8_row(x)
            # Set proper batch dimension shapes.
            if xq.dim() == 3:
                x_scale = x_scale.view(xq.size(0), -1)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        # Handle group gemm if inputs are grouped.
        if isinstance(xq, (list, tuple)):
            output = []
            for i in range(len(xq)):
                output.append(
                    self.gemm_op(
                        xq[i],
                        wq[i],
                        x_scale[i],
                        w_scale[i],
                        use_fast_accum=self.fast_accum,
                    )
                )
            return output
        # Unroll batched gemm if needed.
        elif xq.dim() == 3 and wq.dim() == 3:
            B, M, _ = xq.shape
            _, N, _ = wq.shape
            y = torch.empty((B, M, N), device=xq.device, dtype=torch.bfloat16)
            for i in range(B):
                y[i] = self.gemm_op(
                    xq[i], wq[i], x_scale[i], w_scale[i], use_fast_accum=self.fast_accum
                )
            return y
        # Otherwise return normal gemm result.
        return self.gemm_op(xq, wq, x_scale, w_scale, use_fast_accum=self.fast_accum)

    def quantize_and_compute(self, x, wq, w_scale):
        xq, wq, x_scale, w_scale = self.quantize(x, wq, w_scale)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_rowwise"
        else:
            return "ck_rowwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
            Accelerator.AMD_MI300X,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8RowwisePreshuffleGemm(FP8RowwiseGemm):
    """
    FP8 matmul with rowwise scaling and preshuffling of input B.
    """

    def __init__(self):
        self.fast_accum = True
        if self.supported:
            self.gemm_op = torch.ops.mslk.f8f8bf16_rowwise_preshuffle

    def preprocess(self, x, w):
        x, wq, w_scale = super().preprocess(x, w)
        return x, ck_preshuffle(wq, 16), w_scale

    @property
    def name(self) -> str:
        return "ck_rowwise_preshuffle"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.AMD_MI300X}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8RowwiseGroupedGemm(GemmOpBase):
    """
    FP8 grouped matmul with rowwise scaling.
    """

    def preprocess(self, x, w):
        # Apply sparsity to inputs if appropriate.
        # First check if N and K are fixed.
        m_values = [i.shape[0] for i in x]
        n_values = [i.shape[0] for i in w]
        k_values = [i.shape[1] for i in w]
        # If so, do specialized version of initialization.
        if len(np.unique(n_values)) == 1 and len(np.unique(k_values)) == 1:
            m_values = [i.shape[0] for i in x]
            # Inputs for fixed nk mode must be contiguous, however in the benchmark
            # script they typically are not. Do a little special processing to make them
            # work. In practice this wont be needed.
            # Start by padding along m dimension with zeros.
            max_m = max(m_values)
            x = [
                torch.nn.functional.pad(i, (0, 0, 0, max_m - i.shape[0]), value=0)
                for i in x
            ]
            # Stack inputs into groups.
            x = torch.stack(x).contiguous()
            w = torch.stack(w).contiguous()

            # Preapply weight quantization.
            wq, w_scale = quantize_fp8_row(w)
            # Return processed tensors.
            return (
                x,
                wq,
                w_scale,
                torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device),
            )
        # Otherwise run without sparsity.
        wq, w_scale = zip(*[quantize_fp8_row(i) for i in w])
        return x, wq, w_scale, None

    def quantize(self, x, wq, w_scale, m_values=None):
        # Handle case where inputs are explicitly grouped and non-sparse.
        if isinstance(x, (tuple, list)):
            xq, x_scale = zip(*[triton_quantize_fp8_row(i) for i in x])
            return xq, wq, x_scale, w_scale, m_values
        # Otherwise inputs are unified tensors and sparse.
        else:
            B = x.shape[0]
            xq, x_scale = triton_quantize_fp8_row(x, zero_start_index_M=m_values)
            x_scale = x_scale.view(B, -1)
            return xq, wq, x_scale, w_scale, m_values

    def compute(self, xq, wq, x_scale, w_scale, m_values):
        if m_values is None:
            return torch.ops.mslk.f8f8bf16_rowwise_grouped(
                xq,
                wq,
                x_scale,
                w_scale,
            )
        else:
            # Break tensor into groups, simulates what is done e2e.
            return torch.ops.mslk.f8f8bf16_rowwise_grouped_dynamic(
                xq,
                wq,
                x_scale,
                w_scale,
                zero_start_index_M=m_values,
            )

    def quantize_and_compute(self, x, wq, w_scale, m_values=None):
        xq, wq, x_scale, w_scale, m_values = self.quantize(x, wq, w_scale, m_values)
        return self.compute(xq, wq, x_scale, w_scale, m_values)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_rowwise_grouped"
        else:
            return "ck_rowwise_grouped"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
            Accelerator.AMD_MI300X,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class BF16TritonStackedGroupedGemm(GemmOpBase):
    """
    BF16 grouped matmul with stacked inputs implemented with triton.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)
        w = torch.concat(w, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, w, m_sizes

    def quantize(self, x, w, m_sizes):
        return x, w, m_sizes

    def compute(self, x, w, m_sizes):
        return grouped_gemm(x, w, m_sizes, _use_warp_specialization=True)

    def quantize_and_compute(self, x, w, m_sizes):
        x, w, m_sizes = self.quantize(x, w, m_sizes)
        return self.compute(x, w, m_sizes)

    @property
    def name(self) -> str:
        return "triton_bf16_grouped_stacked"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class BF16TritonStackedGroupedGemmFuseScatterAdd(BF16TritonStackedGroupedGemm):
    """
    BF16 grouped matmul with stacked inputs implemented with triton. Fused with ScatterAdd.
    """

    def preprocess(self, x, w):
        x, w, m_sizes = super().preprocess(x, w)
        M = x.shape[0]
        N = w.shape[0] // m_sizes.shape[0]
        output = torch.zeros(M, N, dtype=torch.bfloat16, device=x.device)
        indices = torch.randperm(M, dtype=torch.int32, device=x.device)
        return x, w, m_sizes, output, indices

    def quantize(self, x, w, m_sizes, *args):
        return *super().quantize(x, w, m_sizes), *args

    def compute(self, x, w, m_sizes, output, indices):
        return grouped_gemm(
            x,
            w,
            m_sizes,
            _use_warp_specialization=True,
            _output_tensor=output,
            _scatter_add_indices=indices,
        )

    def quantize_and_compute(self, x, w, m_sizes, *args):
        x, w, m_sizes, *ret = self.quantize(x, w, m_sizes, *args)
        return self.compute(x, w, m_sizes, *ret)

    @property
    def name(self) -> str:
        return "triton_bf16_grouped_stacked_fuse_scatter_add"


@register_gemm_op
class FP8TritonStackedGroupedGemm(GemmOpBase):
    """
    FP8 grouped matmul with rowwise scaling and stacked inputs implemented with triton.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)
        # Quantize weights.
        wq, w_scale = zip(*[quantize_fp8_row(i) for i in w])
        # Group weights as single tensor.
        wq = torch.concat(wq, dim=0).contiguous()
        w_scale = torch.concat(w_scale, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, w_scale, m_sizes

    def quantize(self, x, wq, w_scale, m_sizes):
        B = x.shape[0]
        xq, x_scale = triton_quantize_fp8_row(x)
        x_scale = x_scale.view(B, -1)
        return xq, wq, x_scale, w_scale, m_sizes

    def compute(self, xq, wq, x_scale, w_scale, m_sizes):
        return grouped_gemm_fp8_rowwise(
            xq, wq, m_sizes, x_scale, w_scale, _use_warp_specialization=True
        )

    def quantize_and_compute(self, x, wq, w_scale, m_sizes):
        xq, wq, x_scale, w_scale, m_sizes = self.quantize(x, wq, w_scale, m_sizes)
        return self.compute(xq, wq, x_scale, w_scale, m_sizes)

    @property
    def name(self) -> str:
        return "triton_grouped_stacked"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8TritonStackedGroupedGemmFuseScatterAdd(FP8TritonStackedGroupedGemm):
    """
    FP8 grouped matmul with stacked inputs implemented with triton. Fused with ScatterAdd.
    """

    def preprocess(self, x, w):
        x, wq, w_scale, m_sizes = super().preprocess(x, w)
        M = x.shape[0]
        N = wq.shape[0] // m_sizes.shape[0]
        output = torch.zeros(M, N, dtype=torch.bfloat16, device=x.device)
        indices = torch.randperm(M, dtype=torch.int32, device=x.device)
        return x, wq, w_scale, m_sizes, output, indices

    def quantize(self, x, wq, w_scale, m_sizes, *args):
        return *super().quantize(x, wq, w_scale, m_sizes), *args

    def compute(self, xq, wq, x_scale, w_scale, m_sizes, output, indices):
        return grouped_gemm_fp8_rowwise(
            xq,
            wq,
            m_sizes,
            x_scale,
            w_scale,
            _use_warp_specialization=True,
            _output_tensor=output,
            _scatter_add_indices=indices,
        )

    def quantize_and_compute(self, x, wq, w_scale, m_sizes, *args):
        xq, wq, x_scale, w_scale, m_sizes, *ret = self.quantize(
            x, wq, w_scale, m_sizes, *args
        )
        return self.compute(xq, wq, x_scale, w_scale, m_sizes, *ret)

    @property
    def name(self) -> str:
        return "triton_grouped_stacked_fuse_scatter_add"


@register_gemm_op
class DeepGemmStacked(GemmOpBase):
    """
    FP8 grouped matmul with blockwise scaling implemented with DeepGemm.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        indices = torch.arange(len(m_values))
        m_indices = indices.repeat_interleave(torch.tensor(m_values)).to(
            device=x[0].device, dtype=torch.int
        )
        # Quantize weights.
        wq, w_scale = zip(*[quantize_fp8_block(i, block_k=128, block_m=128) for i in w])
        # Group weights as single tensor.
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, w_scale, m_indices

    def quantize(self, x, wq, w_scale, m_indices):
        xq, x_scale = quantize_fp8_block(x, block_m=1, block_k=128)
        # Pretranspose scales to deepgemm format.
        x_scale = get_col_major_tma_aligned_tensor(x_scale)
        return xq, wq, x_scale, w_scale, m_indices

    def compute(self, xq, wq, x_scale, w_scale, m_indices):
        # Preallocate output.
        out = torch.empty(
            [xq.shape[0], wq.shape[1]], device=xq.device, dtype=torch.bfloat16
        )
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (xq, x_scale), (wq, w_scale), out, m_indices
        )
        return out

    def quantize_and_compute(self, x, wq, w_scale, m_indices):
        xq, wq, x_scale, w_scale, m_indices = self.quantize(x, wq, w_scale, m_indices)
        return self.compute(xq, wq, x_scale, w_scale, m_indices)

    @property
    def name(self) -> str:
        return "deepgemm_stacked"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        if DEEPGEMM_ENABLED:
            return {Accelerator.NVIDIA_SM90}
        return set()

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class DeepGemmMaskedStacked(DeepGemmStacked):
    def preprocess(self, x, w):
        # Quantize weights.
        wq, w_scale = zip(*[quantize_fp8_block(i, block_k=128, block_m=128) for i in w])
        # Group weights as single tensor.
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()

        # Also view input as flattened.
        m_values = [i.shape[0] for i in x]
        expected_m = max(m_values)
        padded_m_max = ((max(m_values) + 127) // 128) * 128
        masked_m = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)

        num_groups = len(m_values)
        k = x[0].shape[1]
        x_padded = torch.zeros(
            [num_groups, padded_m_max, k], device=x[0].device, dtype=x[0].dtype
        )
        for g in range(num_groups):
            x_padded[g, : m_values[g], :] = x[g]

        # Return processed tensors.
        return x_padded, wq, w_scale, masked_m, expected_m, m_values

    def quantize(self, x, wq, w_scale, masked_m, expected_m, m_values):
        g, m_max, k = x.shape
        xq, x_scale = quantize_fp8_block(x.view(-1, k), block_m=1, block_k=128)
        # Pretranspose scales to deepgemm format.
        x_scale = get_col_major_tma_aligned_tensor(x_scale)
        return (
            xq.view(g, m_max, -1),
            wq,
            x_scale.view(g, m_max, -1),
            w_scale,
            masked_m,
            expected_m,
            m_values,
        )

    def compute(self, xq, wq, x_scale, w_scale, masked_m, expected_m, m_values):
        # Preallocate output.
        out = torch.empty(
            [xq.shape[0], xq.shape[1], wq.shape[1]],
            device=xq.device,
            dtype=torch.bfloat16,
        )
        m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            (xq, x_scale), (wq, w_scale), out, masked_m, expected_m
        )
        num_groups = xq.shape[0]
        out_list = [out[g, : m_values[g], :] for g in range(num_groups)]
        return out_list

    def quantize_and_compute(self, x, wq, w_scale, masked_m, expected_m, m_values):
        xq, wq, x_scale, w_scale, masked_m, expected_m = self.quantize(
            x, wq, w_scale, masked_m, expected_m, m_values
        )
        return self.compute(xq, wq, x_scale, w_scale, masked_m, expected_m, m_values)

    @property
    def name(self) -> str:
        return "deepgemm_masked_stacked"


@register_gemm_op
class DeepGemmBlockwise(GemmOpBase):
    """
    FP8 matmul with blockwise scaling implemented with DeepGemm.
    """

    def preprocess(self, x, w):
        # Quantize weights.
        wq, w_scale = quantize_fp8_block(w, block_m=128, block_k=128)
        # allocate output.
        out = torch.empty(
            x.shape[0], wq.shape[0], device=x.device, dtype=torch.bfloat16
        )
        # Return processed tensors.
        return x, wq, w_scale, out

    def quantize(self, x, wq, w_scale, out):
        xq, x_scale = quantize_fp8_group(x, group_size=128)
        return xq, wq, x_scale, w_scale, out

    def compute(self, xq, wq, x_scale, w_scale, out):
        gemm_fp8_fp8_bf16_nt((xq, x_scale), (wq, w_scale), out)
        return out

    def quantize_and_compute(self, x, wq, w_scale, out):
        xq, wq, x_scale, w_scale, out = self.quantize(x, wq, w_scale, out)
        return self.compute(xq, wq, x_scale, w_scale, out)

    @property
    def name(self) -> str:
        return "deepgemm_blockwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        if DEEPGEMM_ENABLED:
            return {Accelerator.NVIDIA_SM90}
        return set()

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class DeepGemmRowwise(GemmOpBase):
    """
    FP8 matmul with rowwise scaling implemented with DeepGemm.
    """

    def preprocess(self, x, w):
        # Quantize weights.
        wq, w_scale = quantize_fp8_row(w)
        # allocate output.
        out = torch.empty(
            x.shape[0], wq.shape[0], device=x.device, dtype=torch.bfloat16
        )
        # Return processed tensors.
        return x, wq, w_scale, out

    def quantize(self, x, wq, w_scale, out):
        xq, x_scale = quantize_fp8_row(x)
        # Pretranspose scales to deepgemm format.
        x_scale = get_col_major_tma_aligned_tensor(x_scale, rowwise_scaling=True)
        return xq, wq, x_scale, w_scale, out

    def compute(self, xq, wq, x_scale, w_scale, out):
        gemm_fp8_fp8_bf16_nt((xq, x_scale), (wq, w_scale), out)
        return out

    def quantize_and_compute(self, x, wq, w_scale, out):
        xq, wq, x_scale, w_scale, out = self.quantize(x, wq, w_scale, out)
        return self.compute(xq, wq, x_scale, w_scale, out)

    @property
    def name(self) -> str:
        return "deepgemm_rowwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        if DEEPGEMM_ENABLED:
            return {Accelerator.NVIDIA_SM90}
        return set()

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8StackedGroupedGemm(GemmOpBase):
    """
    FP8 grouped matmul with rowwise scaling and stacked inputs.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        # Quantize weights.
        wq, w_scale = zip(*[quantize_fp8_row(i) for i in w])
        # Group weights as single tensor.
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, w_scale, m_sizes

    def quantize(self, x, wq, w_scale, m_sizes):
        B = x.shape[0]
        xq, x_scale = triton_quantize_fp8_row(x)
        x_scale = x_scale.view(B, -1)
        return xq, wq, x_scale, w_scale, m_sizes

    def compute(self, xq, wq, x_scale, w_scale, m_sizes):
        return torch.ops.mslk.f8f8bf16_rowwise_grouped_stacked(
            xq, wq, x_scale, w_scale, m_sizes
        )

    def quantize_and_compute(self, x, wq, w_scale, m_sizes):
        xq, wq, x_scale, w_scale, m_sizes = self.quantize(x, wq, w_scale, m_sizes)
        return self.compute(xq, wq, x_scale, w_scale, m_sizes)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_grouped_stacked"
        else:
            return "ck_grouped_stacked"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
            Accelerator.AMD_MI300X,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8StackedGroupedGemmTorch(FP8StackedGroupedGemm):
    def quantize(self, x, wq, w_scale, m_sizes):
        xq, wq, x_scale, w_scale, m_sizes = super().quantize(x, wq, w_scale, m_sizes)
        offsets = torch.cumsum(m_sizes, dim=0, dtype=torch.int32)
        out = torch.empty(
            (xq.shape[0], wq.shape[1]), dtype=torch.bfloat16, device=xq.device
        )
        x_scale = x_scale.view(x_scale.shape[0])
        return xq, wq, x_scale, w_scale, offsets, out

    def compute(self, xq, wq, x_scale, w_scale, offsets, out):
        return torch.ops.mslk.f8f8bf16_rowwise_grouped_mm(
            xq, wq, x_scale, w_scale, offsets, out
        )

    def quantize_and_compute(self, x, wq, w_scale, m_sizes):
        xq, wq, x_scale, w_scale, offsets, out = self.quantize(x, wq, w_scale, m_sizes)
        return self.compute(xq, wq, x_scale, w_scale, offsets, out)

    @property
    def name(self) -> str:
        return "ck_grouped_stacked_torch_2d3d"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.AMD_MI300X}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class ScaledGroupedMMRowwise(FP8StackedGroupedGemmTorch):
    def __init__(self):
        self.fast_accum = True
        self.torch_compile = False

    def compute(self, xq, wq, x_scale, w_scale, offsets, _):
        if self.torch_compile:
            f = torch.compile(
                torch._scaled_grouped_mm,
                options={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON,CK,CUTLASS,ATEN",
                },
            )
        else:
            f = torch._scaled_grouped_mm

        return f(
            xq,
            wq.transpose(-2, -1),
            offs=offsets,
            out_dtype=torch.bfloat16,
            scale_a=x_scale,
            scale_b=w_scale,
            scale_result=None,
            use_fast_accum=self.fast_accum,
        )

    @property
    def name(self) -> str:
        return "scaled_grouped_mm_rowwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.AMD_MI300X,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8StackedGroupwiseGroupedGemm(GemmOpBase):
    """
    FP8 grouped matmul with groupwise scaling and stacked inputs.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        # Quantize weights.
        wq, w_scale = zip(
            *[quantize_fp8_block(i, block_m=128, block_k=128, k_major=False) for i in w]
        )
        # Group weights as single tensor.
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, w_scale, m_sizes

    def quantize(self, x, wq, w_scale, m_sizes):
        xq, x_scale = quantize_fp8_group(x, m_sizes=m_sizes)
        return xq, wq, x_scale, w_scale, m_sizes

    def compute(self, xq, wq, x_scale, w_scale, m_sizes):
        return torch.ops.mslk.f8f8bf16_groupwise_grouped(
            xq, wq, x_scale, w_scale, m_sizes
        )

    def quantize_and_compute(self, x, wq, w_scale, m_sizes):
        xq, wq, x_scale, w_scale, m_sizes = self.quantize(x, wq, w_scale, m_sizes)
        return self.compute(xq, wq, x_scale, w_scale, m_sizes)

    @property
    def name(self) -> str:
        return "cutlass_groupwise_grouped"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class BF16GroupedGemm(GemmOpBase):
    """
    BF16 grouped matmul implemented with CK or Cutlass.
    """

    def preprocess(self, x, w):
        # Apply sparsity to inputs if appropriate.
        # First check if N and K are fixed.
        m_values = [i.shape[0] for i in x]
        n_values = [i.shape[0] for i in w]
        k_values = [i.shape[1] for i in w]
        # If so, do specialized version of initialization.
        if len(np.unique(n_values)) == 1 and len(np.unique(k_values)) == 1:
            m_values = [i.shape[0] for i in x]
            # Inputs for fixed nk mode must be contiguous, however in the benchmark
            # script they typically are not. Do a little special processing to make them
            # work. In practice this wont be needed.
            # Start by padding along m dimension with zeros.
            max_m = max(m_values)
            x = [
                torch.nn.functional.pad(i, (0, 0, 0, max_m - i.shape[0]), value=0)
                for i in x
            ]
            # Stack inputs into groups.
            x = torch.stack(x).contiguous()
            w = torch.stack(w).contiguous()
            return (
                x,
                w,
                torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device),
            )
        return x, w, None

    def quantize(self, x, w, m_values=None):
        # No action required.
        return x, w, m_values

    def compute(self, x, w, m_values):
        if m_values is None:
            return torch.ops.mslk.bf16bf16bf16_grouped(x, w)
        else:
            return torch.ops.mslk.bf16bf16bf16_grouped_dynamic(x, w, m_values)

    def quantize_and_compute(self, x, w, m_values):
        return self.compute(x, w, m_values)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_bf16_grouped"
        else:
            return "ck_bf16_grouped"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
            Accelerator.AMD_MI300X,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class FP8RowwiseBatchedGemm(GemmOpBase):
    """
    FP8 batched matmul with rowwise scaling.
    """

    def quantize(self, x, w):
        assert isinstance(x, list) and isinstance(w, list)
        x = torch.stack(x, dim=0)
        w = torch.stack(w, dim=0)
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.mslk.f8f8bf16_rowwise_batched(xq, wq, x_scale, w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_rowwise_batched"
        else:
            return "ck_rowwise_batched"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
            Accelerator.AMD_MI300X,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


# This kernel is broken and causes GPU to lock up, needs some investigation
# @register_gemm_op
class TritonFP8RowwiseGemm(GemmOpBase):
    """
    FP8 matmul with rowwise scaling implemented with Triton.
    """

    def __init__(self):
        self.fast_accum = True

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale = quantize_fp8_row(w)
        bias = torch.randn(w.shape[0], device=x.device, dtype=torch.float32)
        return xq, wq, x_scale, w_scale, bias

    def compute(self, xq, wq, x_scale, w_scale, bias):
        return matmul_fp8_row(
            xq,
            wq,
            x_scale,
            w_scale,
            bias=bias,
            fp8_fast_accum=self.fast_accum,
            use_warp_specialization=True,
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "triton_rowwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8TritonBlockwiseGemm(GemmOpBase):
    """
    FP8 matmul with block scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_block(x, 128, 128)
        wq, w_scale = quantize_fp8_block(w, 128, 128)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return matmul_fp8_block(xq, wq, x_scale, w_scale, 128, 128, 128)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "triton_blockwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8CutlassBlockwiseGemm(GemmOpBase):
    """
    FP8 matmul with block scaling.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_block(x, 128, 128)
        wq, w_scale = quantize_fp8_block(w, 128, 128)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.mslk.f8f8bf16_blockwise(
            xq, wq, x_scale, w_scale, 128, 128, 128
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_blockwise"
        else:
            return "ck_blockwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.AMD_MI300X,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class FP8CutlassGroupwiseGemm(GemmOpBase):
    """
    FP8 matmul with group / block scaling.
    """

    def preprocess(self, x, w):
        # Quantize weights.
        # Scale is expected to be in [K, N] layout (N Major).
        wq, w_scale = quantize_fp8_block(w, block_m=128, block_k=128, k_major=False)
        # Return processed tensors.
        return x, wq, w_scale

    def quantize(self, x, wq, w_scale):
        # Scale is expected to be in [K, M] layout (M Major).
        xq, x_scale = quantize_fp8_group(x, k_major=False)
        # Pretranspose scales to deepgemm format.
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.mslk.f8f8bf16_groupwise(xq, wq, x_scale, w_scale)

    def quantize_and_compute(self, x, wq, w_scale):
        xq, wq, x_scale, w_scale = self.quantize(x, wq, w_scale)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cutlass_groupwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class F8I4RowwiseGemm(GemmOpBase):
    """
    Mixed Precision FP8 Activations with Int4 Weights.
    """

    def _int4_row_quantize(
        self,
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

    def _pack_int4(self, x: torch.Tensor) -> torch.Tensor:
        # Given int8 x, pack adjacent int4 values into a single int8.
        low_x = x[:, ::2]
        high_x = x[:, 1::2]

        # High bits need to left shift, this also masks off extra bits.
        high_x = torch.bitwise_left_shift(high_x, 4)
        # Low bits need to have sign bits removed.
        low_x = torch.bitwise_and(low_x, 0xF)

        # Recombine into a single value with bitwise or.
        return torch.bitwise_or(low_x, high_x).contiguous()

    def quantize(self, x, w):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        wq, w_scale, w_zp = self._int4_row_quantize(w)
        # Pack int4 values together.
        wq = self._pack_int4(wq)
        return xq, wq, x_scale, w_scale, w_zp

    def compute(self, xq, wq, x_scale, w_scale, w_zp):
        return torch.ops.mslk.f8i4bf16_rowwise(xq, wq, x_scale, w_scale, w_zp)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale, w_zp = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale, w_zp)

    @property
    def name(self) -> str:
        return "cutlass_f8i4_rowwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class F8I4ShuffledGemm(GemmOpBase):
    def preprocess(self, x, w):
        # Prequantize and pack weights.
        wq, (group_scale, row_scale) = quantize_int4_preshuffle(w)
        return x, wq, row_scale, group_scale

    def quantize(self, x, wq, row_scale, group_scale):
        # Quantize both input tensors.
        xq, x_scale = quantize_fp8_row(x)
        return xq, wq, x_scale, row_scale, group_scale

    def compute(self, xq, wq, x_scale, row_scale, group_scale):
        # Handle batched cases by looping over each batch.
        if xq.dim() == 3:
            B, M, _ = xq.shape
            _, N, _ = wq.shape
            y = torch.empty((B, M, N), device=xq.device, dtype=torch.bfloat16)
            for i in range(B):
                y[i] = torch.ops.mslk.f8i4bf16_shuffled(
                    xq[i], wq[i], x_scale[i], row_scale[i], group_scale[i]
                )
            return y
        # Otherwise run gemm normally.
        return torch.ops.mslk.f8i4bf16_shuffled(xq, wq, x_scale, row_scale, group_scale)

    def quantize_and_compute(self, x, wq, row_scale, group_scale):
        xq, wq, x_scale, row_scale, group_scale = self.quantize(
            x, wq, row_scale, group_scale
        )
        return self.compute(xq, wq, x_scale, row_scale, group_scale)

    @property
    def name(self) -> str:
        return "cutlass_f8i4_preshuffle"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class BF16I4ShuffledGemm(GemmOpBase):
    def preprocess(self, x, w):
        # Prequantize and pack weights.
        wq, (group_scale, group_zero) = quantize_int4_preshuffle(w, dtype="bf16")
        return x, wq, group_scale, group_zero

    def quantize(self, x, wq, group_scale, group_zero):
        # No extra action required.
        return x, wq, group_scale, group_zero

    def compute(self, x, wq, group_scale, group_zero):
        # Handle batched cases by looping over each batch.
        if x.dim() == 3:
            B, M, _ = x.shape
            _, N, _ = wq.shape
            y = torch.empty((B, M, N), device=x.device, dtype=torch.bfloat16)
            for i in range(B):
                y[i] = torch.ops.mslk.bf16i4bf16_shuffled(
                    x[i], wq[i], group_scale[i], group_zero[i]
                )
            return y
        # Otherwise run gemm normally.
        return torch.ops.mslk.bf16i4bf16_shuffled(x, wq, group_scale, group_zero)

    def quantize_and_compute(self, x, wq, group_scale, group_zero):
        x, wq, group_scale, group_zero = self.quantize(x, wq, group_scale, group_zero)
        return self.compute(x, wq, group_scale, group_zero)

    @property
    def name(self) -> str:
        return "cutlass_bf16i4_preshuffle"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class BF16I4ShuffledBatchedGemm(GemmOpBase):
    """
    BF16 x INT4 mixed dtype batched gemm with preshuffling.
    """

    def preprocess(self, x, w):
        assert isinstance(x, list) and isinstance(w, list)
        x = torch.stack(x, dim=0)
        w = torch.stack(w, dim=0)
        # Prequantize and pack weights.
        wq, (group_scale, group_zero) = quantize_int4_preshuffle(w, dtype="bf16")
        return x, wq, group_scale, group_zero

    def quantize(self, x, wq, group_scale, group_zero):
        # No extra action required.
        return x, wq, group_scale, group_zero

    def compute(self, x, wq, group_scale, group_zero):
        return torch.ops.mslk.bf16i4bf16_shuffled_batched(
            x, wq, group_scale, group_zero
        )

    def quantize_and_compute(self, x, wq, group_scale, group_zero):
        x, wq, group_scale, group_zero = self.quantize(x, wq, group_scale, group_zero)
        return self.compute(x, wq, group_scale, group_zero)

    @property
    def name(self) -> str:
        return "cutlass_bf16i4_preshuffle_batched"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class F8I4ShuffledGroupedGemm(GemmOpBase):
    """
    FP8 x Int4 mixed dtype grouped gemm with preshuffling.
    """

    def preprocess(self, x, w):
        assert isinstance(x, list) and isinstance(w, list), (
            "Only supported for grouped inputs."
        )
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)
        # Quantize weights.
        wq, scales = zip(*[quantize_int4_preshuffle(i) for i in w])
        group_scale, row_scale = zip(*scales)
        # Group weights as single tensor.
        wq = torch.stack(wq, dim=0).contiguous()
        row_scale = torch.stack(row_scale, dim=0).contiguous()
        group_scale = torch.stack(group_scale, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, row_scale, group_scale, m_sizes

    def quantize(self, x, wq, row_scale, group_scale, m_sizes):
        B = x.shape[0]
        xq, x_scale = triton_quantize_fp8_row(x)
        x_scale = x_scale.view(B, -1)
        return xq, wq, x_scale, row_scale, group_scale, m_sizes

    def compute(self, xq, wq, x_scale, row_scale, group_scale, m_sizes):
        out = torch.ops.mslk.f8i4bf16_shuffled_grouped(
            xq, wq, x_scale, row_scale, group_scale, m_sizes
        )
        return out

    def quantize_and_compute(self, x, wq, row_scale, group_scale, m_sizes):
        xq, wq, x_scale, row_scale, group_scale, m_sizes = self.quantize(
            x, wq, row_scale, group_scale, m_sizes
        )
        return self.compute(xq, wq, x_scale, row_scale, group_scale, m_sizes)

    @property
    def name(self) -> str:
        if torch.version.cuda:
            return "cutlass_f8i4_grouped_preshuffle"
        else:
            return "ck_f8i4_grouped_preshuffle"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class BF16I4ShuffledGroupedGemm(GemmOpBase):
    """
    BF16 x Int4 mixed dtype grouped gemm with preshuffling.
    """

    def preprocess(self, x, w):
        assert isinstance(x, list) and isinstance(w, list), (
            "Only supported for grouped inputs."
        )
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int32, device=x[0].device)
        # Quantize weights.
        wq, scales = zip(
            *[quantize_int4_preshuffle(i, dtype="bf16", use_zp=False) for i in w]
        )
        # Group weights as single tensor.
        group_scale, group_zero = zip(*scales)
        wq = torch.stack(wq, dim=0).contiguous()
        group_scale = torch.stack(group_scale, dim=0).contiguous()
        group_zero = torch.stack(group_zero, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, wq, group_scale, group_zero, m_sizes

    def quantize(self, x, wq, group_scale, group_zero, m_sizes):
        return x, wq, group_scale, group_zero, m_sizes

    def compute(self, x, wq, group_scale, group_zero, m_sizes):
        # TODO Zero points arent currently supported in grouped gemm.
        # We leave them as inputs for future compatibility but they are ignored.
        return torch.ops.mslk.bf16i4bf16_shuffled_grouped(
            x, wq, group_scale, group_zero, m_sizes
        )

    def quantize_and_compute(self, x, wq, group_scale, group_zero, m_sizes):
        x, wq, group_scale, group_zero, m_sizes = self.quantize(
            x, wq, group_scale, group_zero, m_sizes
        )
        return self.compute(x, wq, group_scale, group_zero, m_sizes)

    @property
    def name(self) -> str:
        return "cutlass_bf16i4_grouped_preshuffle"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class BF16GroupedGrad(GemmOpBase):
    """
    BF16 grouped matmul with dgrad inputs in pretraining backed by cutlass
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        # Group weights as single tensor.
        w = torch.stack(w, dim=0).contiguous()
        # Prepare online dgrad during pretraining backward.
        w_perm = w.permute(0, 2, 1).contiguous()
        # w.contiguous() is very expensive so handling it inside the gmm kernel for free
        w = w_perm.permute(0, 2, 1)

        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, w, m_sizes

    def quantize(self, x, w, m_sizes):
        return x, w, m_sizes

    def compute(self, x, w, m_sizes):
        return torch.ops.mslk.bf16bf16bf16_grouped_grad(x, w, m_sizes)

    def quantize_and_compute(self, x, w, m_sizes):
        x, w, m_sizes = self.quantize(x, w, m_sizes)
        return self.compute(x, w, m_sizes)

    @property
    def name(self) -> str:
        return "bf16_grouped_grad"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class BF16GroupedWGrad(GemmOpBase):
    """
    BF16 grouped matmul with wgrad inputs in pretraining backed by cutlass
    """

    def preprocess(self, x, w):
        # Get K values for each group
        k_values = [xi.shape[1] for xi in x]  # K dimension for each group

        # Convert k_values into sizes tensor
        k_sizes = torch.tensor(k_values).to(dtype=torch.int64, device=x[0].device)

        x = torch.concat(x, dim=1).contiguous()  # shape: (M, G*K)
        w = torch.concat(w, dim=1).contiguous()  # shape: (N, G*K)

        # Transpose the follows to simulate wgrad shapes
        x = x.t().contiguous()  # shape: (G*K, M)
        w = w.t().contiguous()  # shape: (G*K, N)

        # Return processed tensors
        return x, w, k_sizes

    def quantize(self, x, w, k_sizes):
        return x, w, k_sizes

    def compute(self, x, w, k_sizes):
        return torch.ops.mslk.bf16bf16bf16_grouped_wgrad(x, w, k_sizes)

    def quantize_and_compute(self, x, w, k_sizes):
        x, w, k_sizes = self.quantize(x, w, k_sizes)
        return self.compute(x, w, k_sizes)

    @property
    def name(self) -> str:
        return "bf16_grouped_wgrad"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class BF16GroupedStacked(GemmOpBase):
    """
    BF16 grouped matmul with stacked inputs backed by cutlass or ck.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        # Convert m_values into offsets into grouped tensor.
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        # Group weights as single tensor.
        w = torch.stack(w, dim=0).contiguous()
        # Also view input as flattened.
        x = torch.concat(x, dim=0).contiguous()
        # Return processed tensors.
        return x, w, m_sizes

    def quantize(self, x, w, m_sizes):
        return x, w, m_sizes

    def compute(self, x, w, m_sizes):
        return torch.ops.mslk.bf16bf16bf16_grouped_stacked(x, w, m_sizes)

    def quantize_and_compute(self, x, w, m_sizes):
        x, w, m_sizes = self.quantize(x, w, m_sizes)
        return self.compute(x, w, m_sizes)

    @property
    def name(self) -> str:
        return "bf16_grouped_stacked"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {
            Accelerator.NVIDIA_SM90,
            Accelerator.NVIDIA_SM100,
            Accelerator.NVIDIA_SM103,
            Accelerator.AMD_MI300X,
        }

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class BF16I4RowwiseGemm(F8I4RowwiseGemm):
    """
    Mixed Precision BF16 Activations with Int4 Weights.
    """

    def quantize(self, x, w):
        # Quantize both input tensors.
        wq, w_scale, w_zp = self._int4_row_quantize(w)
        # Pack int4 values together.
        wq = self._pack_int4(wq)
        return (
            x.to(torch.bfloat16),
            wq,
            w_scale,
            w_zp,
        )

    def compute(self, x, wq, w_scale, w_zp):
        return torch.ops.mslk.bf16i4bf16_rowwise(x, wq, w_scale, w_zp)

    def quantize_and_compute(self, x, w):
        x, wq, w_scale, w_zp = self.quantize(x, w)
        return self.compute(x, wq, w_scale, w_zp)

    @property
    def name(self) -> str:
        return "cutlass_bf16i4_rowwise"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM90}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class TinyGemmBF16I4(GemmOpBase):
    """
    Mixed Precision BF16 Activations with Int4 Weights using tinygemm.
    """

    def quantize(self, x, w):
        # Quantize and pack weights to int4 using tinygemm utils.
        w_int32, w_scales_and_zeros = group_quantize_tensor(
            w, n_bit=4, q_group_size=128
        )
        wq = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, 4)
        return x, wq, w_scales_and_zeros

    def compute(self, x, wq, scale):
        return torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_int4TC(
            wq, x, 128, scale, False
        )

    def quantize_and_compute(self, x, w):
        x, wq, scale = self.quantize(x, w)
        return self.compute(x, wq, scale)

    @property
    def name(self) -> str:
        return "tinygemm_bf16i4"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        if TINYGEMM_ENABLED:
            return {Accelerator.NVIDIA_SM90}
        return set()

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class MarlinBF16I4(GemmOpBase):
    """
    Mixed Precision BF16 Activations with Int4 Weights using Marlin.
    """

    def quantize(self, x, w):
        # Marlin quantize expects weights in [K, N] layout.
        _, wq, scale = marlin_quantize(w.t().contiguous(), 128)
        return x, wq, scale

    def compute(self, x, wq, scale):
        return torch.ops.marlin.marlin_gemm(x, wq, scale)

    def quantize_and_compute(self, x, w):
        x, wq, scale = self.quantize(x, w)
        return self.compute(x, wq, scale)

    @property
    def name(self) -> str:
        return "marlin_bf16i4"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        if MARLIN_ENABLED:
            return {
                Accelerator.NVIDIA_SM90,
                Accelerator.NVIDIA_SM100,
                Accelerator.NVIDIA_SM103,
            }
        return set()

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class MacheteBF16I4(GemmOpBase):
    """
    Mixed Precision BF16 Activations with Int4 Weights using Machete.
    """

    def quantize(self, x, w):
        # Marlin quantize expects weights in [K, N] layout.
        _, wq, scale, _ = machete_quantize_and_pack(
            w.t().contiguous(), bits=4, groupsize=128
        )
        return x, wq, scale

    def compute(self, x, wq, scale):
        return machete_gemm(x, wq, bits=4, groupsize=128, scales=scale)

    def quantize_and_compute(self, x, w):
        x, wq, scale = self.quantize(x, w)
        return self.compute(x, wq, scale)

    @property
    def name(self) -> str:
        return "machete_bf16i4"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        if MACHETE_ENABLED:
            return {Accelerator.NVIDIA_SM90}
        return set()

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class NVFP4Gemm(GemmOpBase):
    """
    NVFP4 matmul with block-wise scaling.
    """

    def quantize(self, x, w):
        x_global_scale = (448.0 * 6.0) / torch.amax(torch.abs(x.flatten()), dim=-1).to(
            torch.float32
        )
        w_global_scale = (448.0 * 6.0) / torch.amax(torch.abs(w.flatten()), dim=-1).to(
            torch.float32
        )
        global_scale = 1 / (x_global_scale * w_global_scale)

        xq, x_scale = triton_quantize_nvfp4(x, x_global_scale)
        wq, w_scale = triton_quantize_nvfp4(w, w_global_scale)

        return xq, wq, x_scale, w_scale, global_scale

    def compute(self, xq, wq, x_scale, w_scale, global_scale):
        return torch.ops.mslk.f4f4bf16(
            xq, wq, x_scale, w_scale, global_scale=global_scale
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale, global_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale, global_scale=global_scale)

    @property
    def name(self) -> str:
        return "cutlass_nv_f4f4bf16"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP4


@register_gemm_op
class NVFP4Quantize(GemmOpBase):
    """
    NVFP4 quantization with block-wise scaling.
    """

    def quantize_rms(self, x, w):
        M, N = w.shape
        group_size = 16
        w = torch.randn(group_size, dtype=torch.bfloat16, device=w.device)
        x_global_scale = torch.tensor([448.0 * 6.0]).to(
            device=x.device, dtype=torch.float32
        ) / torch.amax(torch.abs(x.flatten()), dim=-1).to(torch.float32)
        xq_ref, x_scale_ref = triton_scale_nvfp4_quant_rms(
            x,
            w.repeat(M * N // group_size),
            x_global_scale,
            group_size=group_size,
            EPS=1e-5,
        )

        intermediate = rms_norm(x.reshape(-1, 16), w, eps=1e-5)
        intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
        xq, x_scale = triton_quantize_nvfp4(
            intermediate,
            x_global_scale,
            group_size=group_size,
        )

    def quantize_silu(self, x, w):
        M, N = x.shape
        group_size = 16
        x_global_scale = torch.tensor([448.0 * 6.0]).to(
            device=x.device, dtype=torch.float32
        ) / torch.amax(torch.abs(x.flatten()), dim=-1).to(torch.float32)
        xq_ref, x_scale_ref = triton_scale_nvfp4_quant_silu(
            x,
            w,
            x_global_scale,
            group_size=group_size,
        )

        intermediate = silu_mul(x.reshape(-1, 16), w.reshape(-1, 16))
        intermediate = intermediate.to(torch.bfloat16).reshape(M, N)
        xq, x_scale = triton_quantize_nvfp4(
            intermediate,
            x_global_scale,
            group_size=group_size,
        )

    def quantize(self, x, w):
        x_global_scale = (448.0 * 6.0) / torch.amax(torch.abs(x.flatten()), dim=-1).to(
            torch.float32
        )
        w_global_scale = (448.0 * 6.0) / torch.amax(torch.abs(w.flatten()), dim=-1).to(
            torch.float32
        )
        global_scale = 1 / (x_global_scale * w_global_scale)

        xq, x_scale = triton_quantize_nvfp4(x, x_global_scale)
        wq, w_scale = triton_quantize_nvfp4(w, w_global_scale)
        return xq, wq, x_scale, w_scale, global_scale

    def compute(self, xq, wq, x_scale, w_scale, global_scale):
        return torch.ops.mslk.f4f4bf16(
            xq, wq, x_scale, w_scale, global_scale=global_scale
        )

    def quantize_and_compute(self, x, w):
        return self.quantize(x, w)

    @property
    def name(self) -> str:
        return "nvfp4_quantize"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP4


@register_gemm_op
class MXFP4Gemm(GemmOpBase):
    """
    MXFP4 matmul with block-wise scaling.
    """

    def quantize(self, x, w):
        xq, x_scale = triton_quantize_mx4_unpack(x)
        wq, w_scale = triton_quantize_mx4_unpack(w)
        return xq, wq, x_scale, w_scale

    def compute(self, xq, wq, x_scale, w_scale):
        return torch.ops.mslk.f4f4bf16(xq, wq, x_scale, w_scale)

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale = self.quantize(x, w)
        return self.compute(xq, wq, x_scale, w_scale)

    @property
    def name(self) -> str:
        return "cutlass_f4f4bf16"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP4


@register_gemm_op
class MXFP4StackedGroupedGemm(GemmOpBase):
    """
    MXFP4 grouped matmul with blockwise scaling and stacked inputs.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        wq, w_scale = zip(*[triton_quantize_mx4_unpack(i) for i in w])
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()
        return x, wq, w_scale, m_sizes

    def quantize(self, x, wq, w_scale, m_sizes):
        starting_row_after_padding_list = [0]
        xq_list = []
        x_scale_list = []
        for i in range(m_sizes.shape[0]):
            scale_slice = x[i]
            if m_sizes[i].item() != 0:
                xq, x_scale = triton_quantize_mx4_unpack(scale_slice)
                xq_list.append(xq)
                x_scale_list.append(x_scale)
                starting_row_after_padding_list.append(
                    starting_row_after_padding_list[i]
                    + x_scale.numel() // (x[0].shape[1] // 32)
                )
            else:
                starting_row_after_padding_list.append(
                    starting_row_after_padding_list[i]
                )
        xq = torch.cat(xq_list, dim=0).contiguous()
        x_scale = torch.cat(x_scale_list, dim=0).contiguous()
        x_scale = x_scale.reshape(-1, x[0].shape[-1] // 32)
        xq = xq.view(-1, xq.shape[-1])
        return (
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            torch.tensor(starting_row_after_padding_list, device=xq.device),
        )

    def compute(self, xq, wq, x_scale, w_scale, m_sizes, starting_row_after_padding):
        return torch.ops.mslk.f4f4bf16_grouped_stacked(
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            starting_row_after_padding=starting_row_after_padding,
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale, m_sizes, starting_row_after_padding = self.quantize(
            x, w
        )
        return self.compute(
            xq, wq, x_scale, w_scale, m_sizes, starting_row_after_padding
        )

    @property
    def name(self) -> str:
        return "cutlass_f4f4bf16_grouped_stacked"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP4


@register_gemm_op
class NVFP4StackedGroupedGemm(GemmOpBase):
    """
    NVFP4 grouped matmul with blockwise scaling and stacked inputs.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        x = torch.concat(x, dim=0).contiguous()

        def get_global_scale(x, w, m_sizes):
            G = len(w)
            w_global_scale = []
            global_scale = []

            cumulative_sum = torch.zeros(
                m_sizes.shape[0] + 1, dtype=torch.int64, device=m_sizes.device
            )
            cumulative_sum[1:] = torch.cumsum(m_sizes, dim=0)

            x_global_scale, tensor_idx = calculate_group_max(x, m_sizes=m_sizes)

            for i in range(G):
                w_global_scale_ = (448.0 * 6.0) / torch.amax(
                    torch.abs(w[i].flatten()), dim=-1
                ).to(torch.float32)

                global_scale_ = 1 / (x_global_scale[i] * w_global_scale_)

                w_global_scale.append(w_global_scale_)
                global_scale.append(global_scale_)

            return x_global_scale, w_global_scale, global_scale, tensor_idx

        # Compute global scale for each group
        G = m_sizes.numel()
        x_global_scale, w_global_scale, global_scale, tensor_idx = get_global_scale(
            x, w, m_sizes
        )
        global_scale = torch.stack(global_scale, dim=0).contiguous()

        wq, w_scale = zip(
            *[triton_quantize_nvfp4(w[i], w_global_scale[i]) for i in range(G)]
        )
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()

        return x, wq, w_scale, x_global_scale, global_scale, m_sizes, tensor_idx

    def quantize(
        self, x, wq, w_scale, x_global_scale, global_scale, m_sizes, tensor_idx
    ):
        # alternative methods, may be useful in some scenarios
        """
        starting_row_after_padding, belong_indices, row_within_tensor = (
            nvfp4_fused_padding_cumsum_and_segmented_arange(m_sizes, x.shape[0])
            # fused_single_block_cumsum_and_segmented_arange(m_sizes, x.shape[0])
        )

        xq, x_scale = triton_nvfp4_quant_stacked(
            x,
            x_global_scale[0],
            belong_indices,
            starting_row_after_padding,
            row_within_tensor,
        )
        """

        # we can optionally set optional_tensor_idx to None to run the alternative method
        xq, x_scale, starting_row_after_padding = mega_fp4_quantize_kernel(
            m_sizes, x, x_global_scale, optional_tensor_idx=tensor_idx
        )

        x_scale = x_scale.reshape(-1, x.shape[1] // 16)
        return (
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            global_scale,
            starting_row_after_padding,
        )

    def compute(
        self,
        xq,
        wq,
        x_scale,
        w_scale,
        m_sizes,
        global_scale,
        starting_row_after_padding,
    ):
        gemm_result = torch.ops.mslk.f4f4bf16_grouped_stacked(
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            global_scale,
            starting_row_after_padding,
            use_mx=False,
        )
        return gemm_result

    def quantize_and_compute(
        self, x, wq, w_scale, x_global_scale, global_scale, m_sizes, tensor_idx
    ):
        (
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            global_scale,
            starting_row_after_padding,
        ) = self.quantize(
            x, wq, w_scale, x_global_scale, global_scale, m_sizes, tensor_idx
        )
        return self.compute(
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            global_scale,
            starting_row_after_padding,
        )

    @property
    def name(self) -> str:
        return "cutlass_nv_f4f4bf16_grouped_stacked"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP4


# Broken with cuda graph
# @register_gemm_op
class NVFP4StackedGroupedGemmPackUnpack(GemmOpBase):
    """
    NVFP4 grouped matmul with blockwise scaling and stacked inputs.
    """

    def preprocess(self, x, w):
        m_values = [i.shape[0] for i in x]
        m_sizes = torch.tensor(m_values).to(dtype=torch.int64, device=x[0].device)
        x = torch.concat(x, dim=0).contiguous()

        def get_global_scale(x, w):
            G = len(w)
            x_global_scale = []
            w_global_scale = []
            global_scale = []

            x_global_scale_ = (448.0 * 6.0) / torch.amax(
                torch.abs(x.flatten()), dim=-1
            ).to(torch.float32)

            for i in range(G):
                w_global_scale_ = (448.0 * 6.0) / torch.amax(
                    torch.abs(w[i].flatten()), dim=-1
                ).to(torch.float32)

                global_scale_ = 1 / (x_global_scale_ * w_global_scale_)

                x_global_scale.append(x_global_scale_)
                w_global_scale.append(w_global_scale_)
                global_scale.append(global_scale_)

            return x_global_scale, w_global_scale, global_scale

        # Compute global scale for each group
        G = m_sizes.numel()
        x_global_scale, w_global_scale, global_scale = get_global_scale(x, w)

        global_scale = torch.stack(global_scale, dim=0).contiguous()

        wq, w_scale = zip(
            *[triton_quantize_nvfp4(w[i], w_global_scale[i]) for i in range(G)]
        )
        wq = torch.stack(wq, dim=0).contiguous()
        w_scale = torch.stack(w_scale, dim=0).contiguous()
        x_global_scale = torch.tensor(x_global_scale, device=m_sizes.device)
        return (
            x,
            wq,
            w_scale,
            x_global_scale,
            global_scale,
            m_sizes,
        )

    def quantize(self, x, wq, w_scale, x_global_scale, global_scale, m_sizes):
        # alternative packing methods that only uses the overall global scale rather than per tensor
        """
        packed = mega_fp4_pack(x, x_global_scale[0])
        """
        packed = mega_fp4_pack(
            x,
            x_global_scale,
            per_tensor=True,
            m_sizes=m_sizes,
        )
        xq, x_scale, starting_row_after_padding = mega_fp4_unpack(m_sizes, packed)
        xq_other, x_scale_other, starting_row_after_padding_other = (
            mega_fp4_quantize_kernel(
                m_sizes,
                x,
                x_global_scale,
            )
        )

        x_scale = x_scale.reshape(-1, x.shape[1] // 16)
        x_scale_other = x_scale_other.reshape(-1, x.shape[1] // 16)
        return (
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            global_scale,
            starting_row_after_padding,
            xq_other,
            x_scale_other,
            starting_row_after_padding_other,
        )

    def compute(
        self,
        xq,
        wq,
        x_scale,
        w_scale,
        m_sizes,
        global_scale,
        starting_row_after_padding,
        xq_other,
        x_scale_other,
        starting_row_after_padding_other,
    ):
        ref_solution = torch.ops.mslk.f4f4bf16_grouped_stacked(
            xq_other,
            wq,
            x_scale_other,
            w_scale,
            m_sizes,
            global_scale,
            starting_row_after_padding_other,
            use_mx=False,
        )
        gemm_result = torch.ops.mslk.f4f4bf16_grouped_stacked(
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            global_scale,
            starting_row_after_padding,
            use_mx=False,
        )
        assert torch.allclose(ref_solution, gemm_result)

        return gemm_result

    def quantize_and_compute(
        self, x, wq, w_scale, x_global_scale, global_scale, m_sizes
    ):
        (
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            global_scale,
            starting_row_after_padding,
            xq_other,
            x_scale_other,
            starting_row_after_padding_other,
        ) = self.quantize(x, wq, w_scale, x_global_scale, global_scale, m_sizes)
        return self.compute(
            xq,
            wq,
            x_scale,
            w_scale,
            m_sizes,
            global_scale,
            starting_row_after_padding,
            xq_other,
            x_scale_other,
            starting_row_after_padding_other,
        )

    @property
    def name(self) -> str:
        return "cutlass_nv_f4f4bf16_grouped_stacked_pack_unpack"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP4


@register_gemm_op
class BF16GroupedGemm2d3d(GemmOpBase):
    """
    Torch BF16 grouped GEMM with 2D inputs and 3D weights.
    """

    def preprocess(self, x, w):
        assert isinstance(x, list)
        assert isinstance(w, list)
        offs = torch.tensor(
            [i.shape[0] for i in x], dtype=torch.int32, device=x[0].device
        )
        offs = torch.cumsum(offs, dim=0).to(torch.int32)
        x = torch.cat(x, dim=0).contiguous()  # (G * M, K)
        w = torch.stack(w, dim=0).contiguous()  # (G, N, K)
        return x, w, offs

    def quantize(self, x, w, offs):
        return x, w, offs

    def compute(self, x, w, offs):
        return torch._grouped_mm(
            x,
            w.transpose(-2, -1),
            offs=offs,
        )

    def quantize_and_compute(self, x, w, offs):
        x, w, offs = self.quantize(x, w)
        return self.compute(x, w, offs)

    @property
    def name(self) -> str:
        return "bf16_baseline_grouped_2d_3d"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return set(Accelerator)

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16


@register_gemm_op
class MXFP8GroupedGemm2d3d(GemmOpBase):
    """
    MXFP8 grouped GEMM with 2D inputs and 3D weights.
    """

    def preprocess(self, x, w):
        assert isinstance(x, list)
        assert isinstance(w, list)
        x = torch.cat(x, dim=0).contiguous()  # (G * M, K)
        w = torch.stack(w, dim=0).contiguous()  # (G, N, K)
        return x, w

    def quantize(self, x, w):
        block_size = 32
        G, N, K = w.shape
        total_M = x.shape[0]
        group_size = total_M // G
        input_group_end_offsets = torch.arange(
            group_size, total_M + 1, group_size, dtype=torch.int32, device=x.device
        )

        # For each constituent 2d subtensor in the 3d weights, quantize and convert scale to blocked format separately,
        # as they each used for independent gemm in the grouped gemm.
        wq_list = []
        w_scale_list = []
        for i in range(G):
            w_scale, wq = to_mxfp8(w[i])
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
                x_slice = x[prev_group_end:curr_group_end, :]
                x_scale, xq = to_mxfp8(x_slice)
                x_scale = _to_blocked(x_scale)
                xq_list.append(xq)
                x_scale_list.append(x_scale)
        xq = torch.cat(xq_list, dim=0).contiguous()
        x_scale = torch.cat(x_scale_list, dim=0).contiguous()
        x_scale = x_scale.reshape(-1, K // block_size)
        xq = xq.view(-1, xq.shape[-1])
        return xq, wq, x_scale, w_scale, input_group_end_offsets

    def compute(self, xq, wq, x_scale, w_scale, input_group_end_offsets):
        return torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq,
            wq.transpose(-2, -1),
            x_scale,
            w_scale,
            input_group_end_offsets,
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale, input_group_end_offsets = self.quantize(x, w)
        return self.compute(
            xq,
            wq,
            x_scale,
            w_scale,
            input_group_end_offsets,
        )

    @property
    def name(self) -> str:
        return "cutlass_mx8mx8bf16_grouped_mm_2d_3d"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class MXFP8GroupedGemm2d2d(GemmOpBase):
    """
    MXFP8 grouped GEMM with 2D inputs and 3D weights.
    """

    def preprocess(self, x, w):
        assert isinstance(x, list)
        assert isinstance(w, list)
        G = len(x)
        x = torch.cat(x, dim=1).contiguous()  # (M, total_K)
        w = torch.cat(w, dim=1).contiguous()  # (N, total_K)
        return x, w, G

    def quantize(self, x, w, G):
        # Simulate 2d-2d grouped gemm in backward pass `grad_weight = grad_output_t @ input`,
        # where we use "K" as the contracting dim which has "G" groups.
        M, total_K = x.shape
        N, _ = w.shape
        group_size = total_K // G
        input_group_end_offsets = torch.arange(
            group_size, total_K + 1, group_size, dtype=torch.int32, device=x.device
        )

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
                x_slice = x[
                    :, prev_group_end_offset:curr_group_end_offset
                ].contiguous()  # (M, K_group)
                w_slice = w[
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
        return xq, wq, x_blocked_scales, w_blocked_scales, input_group_end_offsets

    def compute(self, xq, wq, x_scale, w_scale, input_group_end_offsets):
        return torch.ops.mslk.mx8mx8bf16_grouped_mm(
            xq,
            wq.transpose(-2, -1),
            x_scale,
            w_scale,
            input_group_end_offsets,
        )

    def quantize_and_compute(self, x, w):
        xq, wq, x_scale, w_scale, input_group_end_offsets = self.quantize(x, w)
        return self.compute(
            xq,
            wq,
            x_scale,
            w_scale,
            input_group_end_offsets,
        )

    @property
    def name(self) -> str:
        return "cutlass_mx8mx8bf16_grouped_mm_2d_2d"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.GROUPED}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.FP8


@register_gemm_op
class I4BF16CuteDSLGemm(GemmOpBase):
    """
    CuteDSL Mixed-Input GEMM for Blackwell (SM100+).
    Supports Int4 (x) x BF16 (w) using convert-scale mode.

    This implementation uses the mixed_input_gemm function which takes
    torch tensors as input directly. Activation (x) is quantized to int4
    and weight (w) is in BF16.
    """

    def __init__(self):
        # Default configuration for Int4 x BF16
        self._scale_granularity_m = 1
        self._scale_granularity_k = 128
        self._acc_dtype = cutlass.Float32 if CUTEDSL_MIXED_INPUT_ENABLED else None

    def _int4_quantize(
        self,
        x: torch.Tensor,
        group_size: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to int4 with per-group scaling.

        Args:
            x: Input tensor of shape (M, K) in bf16/fp16.
            group_size: Number of elements per quantization group along K.

        Returns:
            x_quant: Quantized tensor of shape (M, K) in int8 (packed int4).
            scales: Scale tensor of shape (M, K // group_size) in bf16.
        """
        m, k = x.shape
        num_groups = k // group_size

        # Reshape for group-wise quantization: (M, K) -> (M, num_groups, group_size)
        x_reshaped = x.reshape(m, num_groups, group_size).float()

        # Compute scales per group (M, num_groups, 1)
        max_val = x_reshaped.abs().amax(dim=2, keepdim=True)
        max_int4 = 7  # Max value for signed 4-bit integer (-8 to 7)
        scales = max_val.clamp(min=1e-6) / max_int4

        # Quantize: scale -> round -> clamp
        x_quant = (x_reshaped / scales).round().clamp(-max_int4, max_int4)
        x_quant = x_quant.to(torch.int8).reshape(m, k)

        # Make scale m-major, and reshape scales: (M, num_groups, 1) -> (M, num_groups)
        scales = scales.permute(2, 1, 0).contiguous().permute(2, 1, 0)
        scales = scales.squeeze(-1).to(x.dtype)

        return x_quant, scales

    def preprocess(self, x, w):
        """Preprocess inputs - just pass through for quantization step."""
        return x, w

    def quantize(self, x, w):
        """Quantize activation (x) to Int4 format with scales.

        Args:
            x: Activation tensor of shape (M, K) in bf16.
            w: Weight tensor of shape (N, K) in bf16.

        Returns:
            Tuple of (x_quant, x_scale, w, output_tensor, shape).
        """
        m, k = x.shape
        n, _ = w.shape

        # Quantize x to int4 with group-wise scales
        x_quant, x_scale = self._int4_quantize(x, self._scale_granularity_k)

        # Allocate output tensor
        output = torch.empty(m, n, dtype=w.dtype, device=w.device)

        return x_quant, x_scale, w, output, (m, n, k)

    def compute(self, x_quant, x_scale, w, output, shape):
        """Execute the mixed-input GEMM kernel.

        Args:
            x_quant: Quantized activation tensor (M, K) in int8.
            x_scale: Scale tensor (M, K // group_size) in bf16.
            w: Weight tensor (N, K) in bf16.
            output: Output tensor (M, N) in bf16.
            shape: Tuple of (M, N, K).

        Returns:
            Output tensor of shape (M, N) in bf16.
        """
        return int4bf16bf16_gemm(
            A=x_quant,
            B=w,
            A_scale=x_scale,
            C=output,
            scale_granularity_m=self._scale_granularity_m,
            scale_granularity_k=self._scale_granularity_k,
            acc_dtype=self._acc_dtype,
        )

    def quantize_and_compute(self, x, w):
        preprocessed = self.preprocess(x, w)
        quantized = self.quantize(*preprocessed)
        return self.compute(*quantized)

    @property
    def name(self) -> str:
        return "int4bf16_cutedsl_gemm"

    @property
    def supported_accelerators(self) -> set[Accelerator]:
        return {Accelerator.NVIDIA_SM100, Accelerator.NVIDIA_SM103}

    @property
    def supported_gemm_types(self) -> set[GemmType]:
        return {GemmType.REGULAR}

    @property
    def compute_dtype(self) -> ComputeDtype:
        return ComputeDtype.BF16

    @property
    def supported(self) -> bool:
        """Check if the op is supported on current hardware and dependencies are available."""
        if not CUTEDSL_MIXED_INPUT_ENABLED:
            return False
        accelerator = get_current_accelerator()
        if accelerator not in self.supported_accelerators:
            return False
        return True


@register_gemm_op
class BF16I4CuteDSLGemm(I4BF16CuteDSLGemm):
    """
    CuteDSL Mixed-Input GEMM for Blackwell (SM100+).
    Supports BF16 (x) x Int4 (w) using convert-scale mode.

    This is the transpose variant of I4BF16CuteDSLGemm. Here the weight
    (w) is quantized to Int4 and x (activation) is in BF16.
    """

    def preprocess(self, x, w):
        """Preprocess inputs - keep original order for activation quantization."""
        return w, x

    def compute(self, w_quant, w_scale, x, output, shape):
        """Execute the mixed-input GEMM kernel.

        Args:
            w_quant: Quantized activation tensor (N, K) in int8.
            w_scale: Scale tensor (N, K // group_size) in bf16.
            x: Weight tensor (M, K) in bf16.
            output: Output tensor (M, N) in bf16.
            shape: Tuple of (M, N, K).

        Returns:
            Output tensor of shape (M, N) in bf16.
        """
        return int4bf16bf16_gemm(
            A=w_quant,
            B=x,
            A_scale=w_scale,
            C=output,
            scale_granularity_m=self._scale_granularity_m,
            scale_granularity_k=self._scale_granularity_k,
            acc_dtype=self._acc_dtype,
        ).T

    @property
    def name(self) -> str:
        return "bf16int4_cutedsl_gemm"
