# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python-side Meta (shape inference) implementations for GEMM ops.

These were previously registered in C++ (csrc/gemm/gemm_ops.cpp) but are moved
here to decouple Meta dispatch from the C++ ABI.

Platform-specific ops have their C++ schema conditionally compiled, so we
guard each Meta registration with its own `hasattr(torch.ops.mslk, ...)`
check. If the schema was not registered for this build, we skip that op.
"""

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Common ops (always defined, all platforms)
# ---------------------------------------------------------------------------


@torch.library.register_fake("mslk::f8f8bf16_rowwise")
def f8f8bf16_rowwise_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    x_dims = XQ.dim()
    w_dims = WQ.dim()
    assert (x_dims == 2 or x_dims == 3) and (
        w_dims == 2
    ), "The dim of XQ must be 2 or 3, and dim of WQ must be 2"
    if x_dims == 2:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)
    else:
        B = XQ.shape[0]
        M = XQ.shape[1]
        N = WQ.shape[0]
        return torch.empty((B, M, N), dtype=torch.bfloat16, device=XQ.device)


@torch.library.register_fake("mslk::f8f8bf16_rowwise_out")
def f8f8bf16_rowwise_out_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
) -> None:
    pass


@torch.library.register_fake("mslk::f8f8bf16_rowwise_batched")
def f8f8bf16_rowwise_batched_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B = XQ.shape[0]
    M = XQ.shape[1]
    N = WQ.shape[1]
    return torch.empty((B, M, N), dtype=torch.bfloat16, device=XQ.device)


@torch.library.register_fake("mslk::f8f8bf16_blockwise")
def f8f8bf16_blockwise_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    x_dims = XQ.dim()
    w_dims = WQ.dim()
    assert (x_dims == 2 or x_dims == 3) and (
        w_dims == 2
    ), "The dim of XQ must be 2 or 3, and dim of WQ must be 2"
    if x_dims == 2:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)
    else:
        B = XQ.shape[0]
        M = XQ.shape[1]
        N = WQ.shape[0]
        return torch.empty((B, M, N), dtype=torch.bfloat16, device=XQ.device)


@torch.library.register_fake("mslk::f8f8bf16_tensorwise")
def f8f8bf16_tensorwise_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    scale: float,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    x_dims = XQ.dim()
    w_dims = WQ.dim()
    assert (x_dims == 2 or x_dims == 3) and (
        w_dims == 2
    ), "The dim of XQ must be 2 or 3, and dim of WQ must be 2"
    if x_dims == 2:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)
    else:
        B = XQ.shape[0]
        M = XQ.shape[1]
        N = WQ.shape[0]
        return torch.empty((B, M, N), dtype=torch.bfloat16, device=XQ.device)


@torch.library.register_fake("mslk::f8f8bf16_rowwise_grouped_stacked")
def f8f8bf16_rowwise_grouped_stacked_meta(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    total_M = XQ.shape[0]
    N = WQ.shape[1]
    return torch.empty((total_M, N), dtype=torch.bfloat16, device=XQ.device)


@torch.library.register_fake("mslk::bf16bf16bf16_grouped")
def bf16bf16bf16_grouped_meta(
    X: list[torch.Tensor],
    W: list[torch.Tensor],
) -> list[torch.Tensor]:
    Y = []
    for i in range(len(X)):
        M = X[i].shape[0]
        N = W[i].shape[0]
        Y.append(torch.empty((M, N), dtype=torch.bfloat16, device=X[i].device))
    return Y


@torch.library.register_fake("mslk::bf16bf16bf16_grouped_dynamic")
def bf16bf16bf16_grouped_dynamic_meta(
    X: torch.Tensor,
    W: torch.Tensor,
    zero_start_index_M: torch.Tensor,
) -> torch.Tensor:
    G = X.shape[0]
    M = X.shape[1]
    N = W.shape[1]
    return torch.empty((G, M, N), dtype=torch.bfloat16, device=X.device)


@torch.library.register_fake("mslk::bf16bf16bf16_grouped_stacked")
def bf16bf16bf16_grouped_stacked_meta(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    if out is not None:
        return out
    total_M = X.shape[0]
    N = W.shape[1]
    return torch.empty((total_M, N), dtype=torch.bfloat16, device=X.device)


@torch.library.register_fake("mslk::bf16bf16bf16_grouped_grad")
def bf16bf16bf16_grouped_grad_meta(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    if out is not None:
        return out
    total_M = X.shape[0]
    N = W.shape[1]
    return torch.empty((total_M, N), dtype=X.dtype, device=X.device)


@torch.library.register_fake("mslk::bf16bf16bf16_grouped_wgrad")
def bf16bf16bf16_grouped_wgrad_meta(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_accum: bool = False,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    G = M_sizes.shape[0]
    N = X.shape[1]
    K = W.shape[1]
    return torch.empty((G, N, K), dtype=X.dtype, device=X.device)


# ---------------------------------------------------------------------------
# CUDA-only ops (not defined on ROCm builds)
# ---------------------------------------------------------------------------

if hasattr(torch.ops.mslk, "i8i8bf16"):

    @torch.library.register_fake("mslk::i8i8bf16")
    def i8i8bf16_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        scale: float,
        split_k: int = 1,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)


if hasattr(torch.ops.mslk, "f4f4bf16"):

    @torch.library.register_fake("mslk::f4f4bf16")
    def f4f4bf16_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
        mxfp4_block_size: int = 32,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)


if hasattr(torch.ops.mslk, "mx8mx4bf16"):

    @torch.library.register_fake("mslk::mx8mx4bf16")
    def mx8mx4bf16_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)


if hasattr(torch.ops.mslk, "f8f8bf16"):

    @torch.library.register_fake("mslk::f8f8bf16")
    def f8f8bf16_meta(
        X: torch.Tensor,
        W: torch.Tensor,
        scale: torch.Tensor,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        M = X.shape[0]
        N = W.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=X.device)


if hasattr(torch.ops.mslk, "f8f8bf16_groupwise"):

    @torch.library.register_fake("mslk::f8f8bf16_groupwise")
    def f8f8bf16_groupwise_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)


if hasattr(torch.ops.mslk, "f8f8bf16_cublas"):

    @torch.library.register_fake("mslk::f8f8bf16_cublas")
    def f8f8bf16_cublas_meta(
        X: torch.Tensor,
        W: torch.Tensor,
        x_scale: Optional[torch.Tensor] = None,
        w_scale: Optional[torch.Tensor] = None,
        use_fast_accum: bool = True,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        M = X.shape[0]
        N = W.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=X.device)


if hasattr(torch.ops.mslk, "bf16x9_gemm"):

    @torch.library.register_fake("mslk::bf16x9_gemm")
    def bf16x9_gemm_meta(
        A: torch.Tensor,
        B: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        M = A.shape[0]
        N = B.shape[0]
        return torch.empty((M, N), dtype=torch.float32, device=A.device)


if hasattr(torch.ops.mslk, "f8i4bf16_rowwise"):

    @torch.library.register_fake("mslk::f8i4bf16_rowwise")
    def f8i4bf16_rowwise_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        w_zp: torch.Tensor,
    ) -> torch.Tensor:
        x_dims = XQ.dim()
        w_dims = WQ.dim()
        assert (x_dims == 2 or x_dims == 3) and (
            w_dims == 2
        ), "The dim of X must be 2 or 3, and dim of W must be 2"
        if x_dims == 2:
            M = XQ.shape[0]
            N = WQ.shape[0]
            return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)
        else:
            B = XQ.shape[0]
            M = XQ.shape[1]
            N = WQ.shape[0]
            return torch.empty(
                (B, M, N), dtype=torch.bfloat16, device=XQ.device
            )


if hasattr(torch.ops.mslk, "f8i4bf16_shuffled"):

    @torch.library.register_fake("mslk::f8i4bf16_shuffled")
    def f8i4bf16_shuffled_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        w_scale_group: torch.Tensor,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)


if hasattr(torch.ops.mslk, "bf16i4bf16_rowwise"):

    @torch.library.register_fake("mslk::bf16i4bf16_rowwise")
    def bf16i4bf16_rowwise_meta(
        X: torch.Tensor,
        W: torch.Tensor,
        w_scale_group: torch.Tensor,
        w_zero_group: torch.Tensor,
    ) -> torch.Tensor:
        x_dims = X.dim()
        w_dims = W.dim()
        assert (x_dims == 2 or x_dims == 3) and (
            w_dims == 2
        ), "The dim of XQ must be 2 or 3, and dim of WQ must be 2"
        if x_dims == 2:
            M = X.shape[0]
            N = W.shape[0]
            return torch.empty((M, N), dtype=torch.bfloat16, device=X.device)
        else:
            B = X.shape[0]
            M = X.shape[1]
            N = W.shape[0]
            return torch.empty(
                (B, M, N), dtype=torch.bfloat16, device=X.device
            )


if hasattr(torch.ops.mslk, "bf16i4bf16_shuffled"):

    @torch.library.register_fake("mslk::bf16i4bf16_shuffled")
    def bf16i4bf16_shuffled_meta(
        X: torch.Tensor,
        W: torch.Tensor,
        w_scale_group: torch.Tensor,
        w_zero_group: torch.Tensor,
    ) -> torch.Tensor:
        x_dims = X.dim()
        w_dims = W.dim()
        assert (x_dims == 2 or x_dims == 3) and (
            w_dims == 2
        ), "The dim of XQ must be 2 or 3, and dim of WQ must be 2"
        if x_dims == 2:
            M = X.shape[0]
            N = W.shape[0]
            return torch.empty((M, N), dtype=torch.bfloat16, device=X.device)
        else:
            B = X.shape[0]
            M = X.shape[1]
            N = W.shape[0]
            return torch.empty(
                (B, M, N), dtype=torch.bfloat16, device=X.device
            )


if hasattr(torch.ops.mslk, "bf16i4bf16_shuffled_batched"):

    @torch.library.register_fake("mslk::bf16i4bf16_shuffled_batched")
    def bf16i4bf16_shuffled_batched_meta(
        X: torch.Tensor,
        W: torch.Tensor,
        w_scale_group: torch.Tensor,
        w_zero_group: torch.Tensor,
    ) -> torch.Tensor:
        B = X.shape[0]
        M = X.shape[1]
        N = W.shape[1]
        return torch.empty((B, M, N), dtype=torch.bfloat16, device=X.device)


if hasattr(torch.ops.mslk, "bf16i4bf16_rowwise_batched"):

    @torch.library.register_fake("mslk::bf16i4bf16_rowwise_batched")
    def bf16i4bf16_rowwise_batched_meta(
        X: torch.Tensor,
        W: torch.Tensor,
        w_scale_group: torch.Tensor,
        w_zero_group: torch.Tensor,
    ) -> torch.Tensor:
        B = X.shape[0]
        M = X.shape[1]
        N = W.shape[1]
        return torch.empty((B, M, N), dtype=torch.bfloat16, device=X.device)


if hasattr(torch.ops.mslk, "i8i8bf16_dynamic"):

    @torch.library.register_fake("mslk::i8i8bf16_dynamic")
    def i8i8bf16_dynamic_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        scale: torch.Tensor,
        split_k: int = 1,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)


if hasattr(torch.ops.mslk, "preshuffle_i4"):

    @torch.library.register_fake("mslk::preshuffle_i4")
    def preshuffle_i4_meta(
        WQ: torch.Tensor,
        w_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        WS = torch.empty_like(w_scale)
        if w_scale.dtype != torch.bfloat16:
            WS = torch.empty(
                (w_scale.shape[0], 8, w_scale.shape[1]),
                dtype=w_scale.dtype,
                device=w_scale.device,
            )
        return torch.empty_like(WQ), WS


# ---------------------------------------------------------------------------
# ROCm-only ops (not defined on CUDA builds)
# ---------------------------------------------------------------------------

if hasattr(torch.ops.mslk, "f8f8f16_rowwise"):

    @torch.library.register_fake("mslk::f8f8f16_rowwise")
    def f8f8f16_rowwise_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.float16, device=XQ.device)


if hasattr(torch.ops.mslk, "f8f8bf16_rowwise_preshuffle"):

    @torch.library.register_fake("mslk::f8f8bf16_rowwise_preshuffle")
    def f8f8bf16_rowwise_preshuffle_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        x_dims = XQ.dim()
        if x_dims == 2:
            M = XQ.shape[0]
            N = WQ.shape[0]
            return torch.empty((M, N), dtype=torch.bfloat16, device=XQ.device)
        else:
            B = XQ.shape[0]
            M = XQ.shape[1]
            N = WQ.shape[0]
            return torch.empty(
                (B, M, N), dtype=torch.bfloat16, device=XQ.device
            )


if hasattr(torch.ops.mslk, "f8f8f16_rowwise_preshuffle"):

    @torch.library.register_fake("mslk::f8f8f16_rowwise_preshuffle")
    def f8f8f16_rowwise_preshuffle_meta(
        XQ: torch.Tensor,
        WQ: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        M = XQ.shape[0]
        N = WQ.shape[0]
        return torch.empty((M, N), dtype=torch.float16, device=XQ.device)
