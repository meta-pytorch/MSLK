# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

import unittest

import mslk.conv  # noqa: F401
import torch
from mslk.quantize.triton.fp8_quantize import quantize_fp8_tensor


def evaluate_cuda_compute_capability(
    major_min: int, major_max: int | None = None
) -> bool:
    major, _ = torch.cuda.get_device_capability()
    return major >= major_min and (major_max is None or major <= major_max)


def supports_f8f8bf16_conv() -> bool:
    """Check if f8f8bf16_conv is supported on this device."""
    if torch.cuda.is_available():
        # Currently only supported on CUDA (not HIP) and requires SM100+
        if torch.version.cuda:
            return evaluate_cuda_compute_capability(10)
    return False


@unittest.skipIf(
    not torch.cuda.is_available(),
    "Operators are only available on CUDA enabled machines",
)
@unittest.skipIf(
    not supports_f8f8bf16_conv(),
    "f8f8bf16_conv is not supported on this device.",
)
class F8F8BF16ConvTest(unittest.TestCase):
    """Test f8f8bf16_conv operator for correctness, compile, and export."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.accelerator.current_accelerator()

    def _prepare_inputs(
        self,
        n: int,
        c: int,
        d: int,
        h: int,
        w: int,
        k: int,
        t: int,
        r: int,
        s: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare FP8 quantized inputs for conv operation.

        Args:
            n: batch size
            c: input channels
            d, h, w: input spatial dimensions (depth, height, width)
            k: output channels (filters)
            t, r, s: kernel spatial dimensions

        Returns:
            activation_q: quantized activation tensor (FP8)
            filter_q: quantized filter tensor (FP8)
            scale: combined scale tensor
            activation_bf16: original activation for reference
            filter_bf16: original filter for reference
        """
        # Create input tensors in NCDHW format
        activation = torch.randn(
            n, c, d, h, w, device=self.device, dtype=torch.bfloat16
        )
        # Create filter tensors in KCTRS format
        filter_tensor = torch.randn(
            k, c, t, r, s, device=self.device, dtype=torch.bfloat16
        )

        # Convert to channels_last_3d memory format (required by CUTLASS kernels)
        activation = activation.to(memory_format=torch.channels_last_3d)
        filter_tensor = filter_tensor.to(memory_format=torch.channels_last_3d)

        # Quantize to FP8 with rowwise scaling
        activation_q, activation_scale = quantize_fp8_tensor(activation)
        filter_q, filter_scale = quantize_fp8_tensor(filter_tensor)

        # Compute combined scale for output
        scale = torch.tensor(
            [activation_scale * filter_scale],
            device=self.device,
            dtype=torch.float32,
        )

        return activation_q, filter_q, scale, activation, filter_tensor

    def test_f8f8bf16_conv_correctness(self) -> None:
        """Test that f8f8bf16_conv produces outputs close to reference."""
        # Use a simple shape for correctness testing
        n, c, d, h, w = 1, 64, 8, 8, 8
        k, t, r, s = 64, 3, 3, 3
        padding = [1, 1, 1]
        stride = [1, 1, 1]
        dilation = [1, 1, 1]

        activation_q, filter_q, scale, _, _ = self._prepare_inputs(
            n, c, d, h, w, k, t, r, s
        )

        # Compute reference output using PyTorch conv3d
        out_ref = (
            torch.nn.functional.conv3d(
                activation_q.to(torch.bfloat16),
                filter_q.to(torch.bfloat16),
                bias=None,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            * scale
        )

        # Compute FP8 conv output
        output = torch.ops.mslk.f8f8bf16_conv(
            activation_q,
            filter_q,
            scale,
            padding,
            stride,
            dilation,
        )

        # Check output shape matches reference
        self.assertEqual(output.shape, out_ref.shape)

        # Check output dtype
        self.assertEqual(output.dtype, torch.bfloat16)

        # Check outputs are reasonably close (FP8 has limited precision)
        # Using mean squared error as a sanity check
        torch.testing.assert_close(output, out_ref.to(torch.bfloat16))

    def test_f8f8bf16_conv_various_shapes(self) -> None:
        """Test f8f8bf16_conv with various input shapes."""
        test_shapes = [
            # (N, C, D, H, W, K, T, R, S, pad, stride, dilation)
            (1, 64, 8, 8, 8, 64, 3, 3, 3, 1, 1, 1),
            (4, 64, 8, 8, 8, 128, 3, 3, 3, 1, 1, 1),
            (1, 128, 16, 16, 16, 128, 3, 3, 3, 1, 1, 1),
            # 1x1x1 convolution
            (1, 64, 16, 16, 16, 128, 1, 1, 1, 0, 1, 1),
        ]

        for n, c, d, h, w, k, t, r, s, pad, stride_val, dilation_val in test_shapes:
            with self.subTest(N=n, C=c, D=d, H=h, W=w, K=k, T=t, R=r, S=s):
                padding = [pad, pad, pad]
                stride = [stride_val, stride_val, stride_val]
                dilation = [dilation_val, dilation_val, dilation_val]

                activation_q, filter_q, scale, _, _ = self._prepare_inputs(
                    n, c, d, h, w, k, t, r, s
                )

                # Compute reference using dequantized inputs.

                out_ref = (
                    torch.nn.functional.conv3d(
                        activation_q.to(torch.bfloat16),
                        filter_q.to(torch.bfloat16),
                        bias=None,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                    * scale
                )

                # Compute FP8 output
                output = torch.ops.mslk.f8f8bf16_conv(
                    activation_q,
                    filter_q,
                    scale,
                    padding,
                    stride,
                    dilation,
                )

                self.assertEqual(output.shape, out_ref.shape)
                torch.testing.assert_close(output, out_ref.to(torch.bfloat16))

    def test_compile_f8f8bf16_conv(self) -> None:
        """Test that f8f8bf16_conv can be compiled with torch.compile."""
        n, c, d, h, w = 1, 64, 8, 8, 8
        k, t, r, s = 64, 3, 3, 3
        padding = [1, 1, 1]
        stride = [1, 1, 1]
        dilation = [1, 1, 1]

        activation_q, filter_q, scale, _, _ = self._prepare_inputs(
            n, c, d, h, w, k, t, r, s
        )

        # Test that the operator can be compiled
        compiled_fn = torch.compile(torch.ops.mslk.f8f8bf16_conv)
        output = compiled_fn(
            activation_q,
            filter_q,
            scale,
            padding,
            stride,
            dilation,
        )

        # Verify output is valid
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertFalse(output.isnan().any(), "Output contains NaN values")

    def test_export_f8f8bf16_conv(self) -> None:
        """Test that f8f8bf16_conv can be exported with torch.export."""

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(
                self,
                activation_q: torch.Tensor,
                filter_q: torch.Tensor,
                scale: torch.Tensor,
            ) -> torch.Tensor:
                padding = [1, 1, 1]
                stride = [1, 1, 1]
                dilation = [1, 1, 1]
                return torch.ops.mslk.f8f8bf16_conv(
                    activation_q,
                    filter_q,
                    scale,
                    padding,
                    stride,
                    dilation,
                )

        n, c, d, h, w = 1, 64, 8, 8, 8
        k, t, r, s = 64, 3, 3, 3

        activation_q, filter_q, scale, _, _ = self._prepare_inputs(
            n, c, d, h, w, k, t, r, s
        )

        model = TestModule().cuda()

        # Test that the model can be exported
        _ = torch.export.export(model, (activation_q, filter_q, scale), strict=True)


if __name__ == "__main__":
    unittest.main()
