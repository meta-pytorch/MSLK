# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import abc
from typing import Any, TypeVar

import torch
import triton  # @manual=//triton:triton
from mslk.quantize.triton.fp8_quantize import (
    dequantize_fp8_block,
    dequantize_fp8_row,
    triton_quantize_fp8_block,
    triton_quantize_fp8_group,
    triton_quantize_fp8_row,
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

    def benchmark(
        self,
        A: torch.Tensor,
        use_cuda_graph: bool = True,
    ) -> float:
        if use_cuda_graph:
            with torch.cuda.stream(torch.cuda.Stream()):
                return triton.testing.do_bench_cudagraph(
                    lambda: self.quantize(A), rep=200
                )
        else:
            return triton.testing.do_bench(lambda: self.quantize(A), rep=200)


op_registry: list[QuantizeOpBase] = []

T = TypeVar("T", bound=QuantizeOpBase)


def register_op(op_class: type[T]) -> type[T]:
    """Decorator function for assembling all quantize ops."""
    op_registry.append(op_class())
    return op_class


def get_ops() -> list[QuantizeOpBase]:
    """Get all registered quantize ops."""
    return op_registry


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
    def name(self) -> str:
        return "TritonFP8Rowwise"

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
    def name(self) -> str:
        return "TritonFP8Blockwise"

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
    def name(self) -> str:
        return "TritonFP8Groupwise"

    @property
    def hip(self) -> bool:
        return True

    @property
    def cuda(self) -> bool:
        return True
