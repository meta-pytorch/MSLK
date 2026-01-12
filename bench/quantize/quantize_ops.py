# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from typing import Any, TypeVar

import torch
from mslk.bench.common.utils import BenchOptions, do_bench
from mslk.quantize.triton.fp4_quantize import triton_quantize_nvfp4
from mslk.quantize.triton.fp8_quantize import (
    dequantize_fp8_block,
    dequantize_fp8_row,
    triton_quantize_fp8_block,
    triton_quantize_fp8_group,
    triton_quantize_fp8_row,
    triton_quantize_fp8_tensor,
)
from mslk.test.quantize.triton.fp4_quantize_test import (
    dequantize_nvfp4,
    global_scale_nvfp4,
)


class QuantizeOpBase(metaclass=abc.ABCMeta):
    """Helper abstract class to define expected methods of quantize ops."""

    @abc.abstractmethod
    def quantize(self, input: torch.Tensor) -> Any:
        """Function which quantizes inputs."""
        pass

    @abc.abstractmethod
    def dequantize(self, *args: Any) -> torch.Tensor:
        """Function which dequantizes inputs. Used for sanity checking."""
        pass

    @abc.abstractproperty
    def hip(self) -> bool:
        """Whether this operator supports AMD or not."""
        pass

    @abc.abstractproperty
    def cuda(self) -> bool:
        """Whether this operator supports Nvidia or not."""
        pass

    def preprocess(self, input: torch.Tensor) -> Any:
        """This is used for ops that require additional preprocessing. This method will not be counted in benchmarking."""
        return ()

    @property
    def name(self) -> str:
        """Name of this operator."""
        return self.__class__.__name__

    @property
    def supported(self) -> bool:
        """Whether this op will run on the current device."""
        if torch.version.hip is not None:
            return self.hip
        elif torch.version.cuda is not None:
            return self.cuda
        else:
            return False

    def benchmark(
        self,
        input: torch.Tensor,
        args: Any,
        opts: BenchOptions,
    ) -> float:
        """Benchmark runtime of this operator using do_bench from common."""
        return do_bench(
            lambda inp, *a: self.quantize(inp, *a),
            (input, *args),
            opts,
        )


op_registry: dict[str, QuantizeOpBase] = {}

T = TypeVar("T", bound=QuantizeOpBase)


def register_op(op_class: type[T]) -> type[T]:
    """Decorator function for assembling all quantize ops."""
    op_registry[op_class.__name__] = op_class()
    return op_class


def get_ops() -> list[QuantizeOpBase]:
    """Get all registered quantize ops."""
    return list(op_registry.values())


@register_op
class TritonFP8Rowwise(QuantizeOpBase):
    def quantize(self, input: torch.Tensor) -> Any:
        return triton_quantize_fp8_row(input)

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args
        return dequantize_fp8_row(input_quantized, scale)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonFP8Blockwise(QuantizeOpBase):
    def __init__(self) -> None:
        super().__init__()
        self.block_m = 128
        self.block_k = 128

    def quantize(self, input: torch.Tensor) -> Any:
        return triton_quantize_fp8_block(
            input, block_m=self.block_m, block_k=self.block_k
        )

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args
        return dequantize_fp8_block(input_quantized, scale, self.block_m, self.block_k)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonFP8Groupwise(QuantizeOpBase):
    def __init__(self) -> None:
        super().__init__()
        self.group_size = 128

    def quantize(self, input: torch.Tensor) -> Any:
        return triton_quantize_fp8_group(input, group_size=self.group_size)

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args

        input_quantized = input_quantized.to(torch.float)
        dequantized = input_quantized.view(
            -1, input_quantized.shape[1] // self.group_size, self.group_size
        ) * scale.unsqueeze(-1)
        return dequantized.view(input_quantized.shape)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonNVFP4(QuantizeOpBase):
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, input: torch.Tensor) -> Any:
        global_scale = global_scale_nvfp4(input)
        return (global_scale,)

    def quantize(self, input: torch.Tensor, *args: Any) -> Any:
        global_scale: torch.Tensor
        global_scale = args[0]
        input_quantized, scales = triton_quantize_nvfp4(input, global_scale)
        return input_quantized.view(torch.uint8), scales, global_scale

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        global_scale: torch.Tensor
        input_quantized, scale, global_scale = args

        return dequantize_nvfp4(input_quantized, scale, global_scale)

    @property
    def hip(self) -> bool:
        return False

    @property
    def cuda(self) -> bool:
        return True


@register_op
class CudaFP8Rowwise(QuantizeOpBase):
    def quantize(self, input: torch.Tensor) -> Any:
        return torch.ops.mslk.quantize_fp8_per_row(input)

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args
        return dequantize_fp8_row(input_quantized, scale)

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class CudaFP8Tensorwise(QuantizeOpBase):
    def quantize(self, input: torch.Tensor) -> Any:
        return torch.ops.mslk.quantize_fp8_per_tensor(input)

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args
        return input_quantized.to(torch.float32) * scale

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True


@register_op
class TritonFP8Tensorwise(QuantizeOpBase):
    def quantize(self, input: torch.Tensor) -> Any:
        return triton_quantize_fp8_tensor(input)

    def dequantize(self, *args: Any) -> torch.Tensor:
        input_quantized: torch.Tensor
        scale: torch.Tensor
        input_quantized, scale = args
        return input_quantized.to(torch.float32) * scale

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True
