# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Keep a registry of all convolution operators.
import abc

import mslk.conv  # noqa: F401
import torch
from mslk.bench.common.utils import BenchOptions, do_bench
from mslk.quantize.triton.fp8_quantize import quantize_fp8_tensor


conv_op_registry = []


class ConvOpBase(metaclass=abc.ABCMeta):
    """Helper abstract class to define expected methods of conv ops."""

    @abc.abstractmethod
    def quantize(self, *args):
        """Function which quantizes inputs."""
        pass

    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        """Function which performs main compute operation."""
        pass

    @abc.abstractmethod
    def quantize_and_compute(self, *args, **kwargs):
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
        return do_bench(
            lambda *a: self.quantize_and_compute(*a)
            if bench_quantize
            else self.compute(*a),
            args,
            opts,
        )

    @abc.abstractproperty
    def name(self) -> str:
        """Name of the operator."""
        pass

    @abc.abstractproperty
    def hip(self) -> bool:
        """Whether this operator supports AMD or not."""
        pass

    @abc.abstractproperty
    def cuda(self) -> bool:
        """Whether this operator supports Nvidia or not."""
        pass

    @property
    def supported(self) -> bool:
        """Whether this op will run on the current device."""
        if torch.version.hip is not None:
            return self.hip
        elif torch.version.cuda is not None:
            return self.cuda
        else:
            return False


def register_conv_op(op):
    """Decorator function for assembling all conv ops."""
    conv_op_registry.append(op())
    return op


def get_conv_ops() -> list[ConvOpBase]:
    """Get all registered conv ops."""
    return conv_op_registry


@register_conv_op
class TorchBaseline(ConvOpBase):
    """
    PyTorch baseline convolution.
    """

    def __init__(self):
        self.torch_compile = False

    def quantize(self, activation, filter, padding, stride, dilation):
        return (
            activation.to(torch.bfloat16),
            filter.to(torch.bfloat16),
            padding,
            stride,
            dilation,
        )

    def compute(self, activation, filter, padding, stride, dilation):
        if self.torch_compile:
            f = torch.compile(
                torch.nn.functional.conv3d,
                options={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON,CK,CUTLASS,ATEN",
                },
            )
        else:
            f = torch.nn.functional.conv3d

        return f(
            activation,
            filter,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def quantize_and_compute(self, activation, filter, padding, stride, dilation):
        return self.compute(
            *self.quantize(activation, filter, padding, stride, dilation)
        )

    @property
    def name(self) -> str:
        return "torch_baseline"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_conv_op
class F8F8BF16Conv(ConvOpBase):
    """
    FP8 convolution with rowwise scaling.
    """

    def preprocess(self, activation, filter, padding, stride, dilation):
        # Inputs and filters are provided in channels first layout.
        # Cutlass kernels support this but require the underlying memory
        # to be channels last. Torch enables this through the memory format
        # transformation which we assume has been applied ahead of time.
        activation = activation.to(memory_format=torch.channels_last_3d)
        filter = filter.to(memory_format=torch.channels_last_3d)
        return activation, filter, padding, stride, dilation

    def _quantize_tensor(self, x):
        """Quantize tensor to FP8 with rowwise scaling."""
        xq, x_scale = quantize_fp8_tensor(x)
        return xq, x_scale

    def quantize(self, activation, filter, padding, stride, dilation):
        # Quantize both input tensors
        activation_q, activation_scale = self._quantize_tensor(activation)
        filter_q, filter_scale = self._quantize_tensor(filter)

        # Compute combined scale for output
        # For conv, we need a single scale value
        scale = torch.tensor(
            [activation_scale * filter_scale],
            device=activation.device,
            dtype=torch.float32,
        )

        return activation_q, filter_q, scale, padding, stride, dilation

    def compute(self, activation_q, filter_q, scale, padding, stride, dilation):
        output = torch.ops.mslk.f8f8bf16_conv(
            activation_q,
            filter_q,
            scale,
            padding,
            stride,
            dilation,
        )
        return output

    def quantize_and_compute(self, activation, filter, padding, stride, dilation):
        activation_q, filter_q, scale, padding, stride, dilation = self.quantize(
            activation, filter, padding, stride, dilation
        )
        return self.compute(activation_q, filter_q, scale, padding, stride, dilation)

    @property
    def name(self) -> str:
        return "f8f8bf16_conv"

    @property
    def hip(self) -> bool:
        # Currently only supported on CUDA
        return False

    @property
    def cuda(self) -> bool:
        return True
