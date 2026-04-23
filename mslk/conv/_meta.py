# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python-side Meta (shape inference) implementations for Conv ops.

These were previously registered in C++ (csrc/conv/conv_ops.cpp) but are moved
here to decouple Meta dispatch from the C++ ABI.

Platform-specific ops have their C++ schema conditionally compiled, so we
guard their Meta registrations with a `hasattr(torch.ops.mslk, ...)` check on
a representative op. If the schema was not registered for this build, the
check is False and we skip the block.
"""

import torch


# ---------------------------------------------------------------------------
# CUDA-only ops (not defined on ROCm builds)
# ---------------------------------------------------------------------------
if hasattr(torch.ops.mslk, "f8f8bf16_conv"):

    @torch.library.register_fake("mslk::f8f8bf16_conv")
    def f8f8bf16_conv_meta(
        activation: torch.Tensor,
        filter: torch.Tensor,
        scale: torch.Tensor,
        padding: list[int],
        stride: list[int],
        dilation: list[int],
    ) -> torch.Tensor:
        assert activation.dim() == 5, "Activation must be 5D tensor"
        assert filter.dim() == 5, "Filter must be 5D tensor"

        # Check if input is channels first or channels last.
        channels_last_act = activation.stride(4) == 1
        channels_last_filt = filter.stride(4) == 1

        if channels_last_act:
            n, d, h, w = (
                activation.shape[0],
                activation.shape[1],
                activation.shape[2],
                activation.shape[3],
            )
        else:
            n, d, h, w = (
                activation.shape[0],
                activation.shape[2],
                activation.shape[3],
                activation.shape[4],
            )

        if channels_last_filt:
            k, t, r, s = (
                filter.shape[0],
                filter.shape[1],
                filter.shape[2],
                filter.shape[3],
            )
        else:
            k, t, r, s = (
                filter.shape[0],
                filter.shape[2],
                filter.shape[3],
                filter.shape[4],
            )

        assert (
            len(padding) == 3
        ), "Padding must have 3 elements corresponding to [d, h, w]"
        assert (
            len(stride) == 3
        ), "Stride must have 3 elements corresponding to [d, h, w]"
        assert (
            len(dilation) == 3
        ), "Dilation must have 3 elements corresponding to [d, h, w]"

        pad_d, pad_h, pad_w = padding
        stride_d, stride_h, stride_w = stride
        dilation_d, dilation_h, dilation_w = dilation

        z = 1 + (d + 2 * pad_d - ((t - 1) * dilation_d + 1)) // stride_d
        p = 1 + (h + 2 * pad_h - ((r - 1) * dilation_h + 1)) // stride_h
        q = 1 + (w + 2 * pad_w - ((s - 1) * dilation_w + 1)) // stride_w

        if channels_last_act:
            Y = torch.empty(
                (n, z, p, q, k),
                dtype=torch.bfloat16,
                device=activation.device,
            )
        else:
            Y = torch.empty(
                (n, k, z, p, q),
                dtype=torch.bfloat16,
                device=activation.device,
                memory_format=torch.channels_last_3d,
            )
        return Y
