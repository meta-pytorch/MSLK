#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Benchmark: symmetric MX6×MX6 CUTLASS block-scaled GEMM kernel (mx6mx6bf16)
# Compares against BF16 matmul, MX8×MX8, MX8×MX6, MX8×MX4, and native NVFP4 (f4f4bf16).
# Throughput/latency only — no SQNR (no Python MX6 quantizer for the A operand).

from __future__ import annotations

import time

import mslk.gemm  # noqa: F401
import mslk.quantize  # noqa: F401
import torch
from mslk.gemm.triton.fp8_gemm import to_mxfp8
from mslk.quantize.triton.fp4_quantize import _to_blocked, triton_quantize_mx4_unpack


def benchmark_fn(fn, warmup: int = 5, iters: int = 20) -> float:
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times.sort()
    return times[len(times) // 2]  # median


def tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """Compute TFLOPS from GEMM dimensions and time."""
    flops = 2 * M * N * K
    return flops / (time_ms * 1e-3) / 1e12


def main() -> None:
    device = torch.accelerator.current_accelerator()
    N = 16384
    K = 16384
    num_iters = 20
    M_values = [1, 16, 64, 128, 256, 512, 1024]

    print(
        f"Benchmark on {torch.cuda.get_device_name()}, N=K={N}, num_iters={num_iters}"
    )
    print()

    print(
        f"{'M':>6} | {'bf16_mm':>10} | {'mx8mx8':>8} | {'mx8mx6':>8} | {'mx6mx6':>8} | {'mx8mx4':>8} | {'f4f4':>8}"
    )
    print(
        f"{'':>6} | {'(TFLOPS)':>10} | {'(TFLOPS)':>8} | {'(TFLOPS)':>8} | {'(TFLOPS)':>8} | {'(TFLOPS)':>8} | {'(TFLOPS)':>8}"
    )
    print("-" * 84)

    for M in M_values:
        # BF16 matmul reference
        A_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
        B_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device=device)

        t_bf16 = benchmark_fn(
            lambda _a=A_bf16, _b=B_bf16: _a @ _b.t(),
            iters=num_iters,
        )
        tflops_bf16 = tflops(M, N, K, t_bf16)

        # Quantize A to MX8 (used as activation for mx8mx8/mx8mx6/mx8mx4)
        A_small = A_bf16 * 0.1
        B_small = B_bf16 * 0.01
        (a_scale_raw, aq) = to_mxfp8(A_small)
        a_scale = _to_blocked(a_scale_raw.view(torch.int8).reshape(M, -1)).view(
            torch.uint8
        )

        # MX8×MX8 via grouped_mm (single group)
        (b_scale_raw_8, bq_8) = to_mxfp8(B_small)
        b_scale_8 = _to_blocked(b_scale_raw_8.view(torch.int8).reshape(N, -1)).view(
            torch.uint8
        )
        bq_8_col = bq_8.t()
        offsets = torch.tensor([K], dtype=torch.int32, device=device)
        out_mx8 = torch.empty((1, M, N), dtype=torch.bfloat16, device=device)
        a_scale_2d = (
            a_scale.reshape(-1, a_scale.shape[-1]) if a_scale.ndim != 2 else a_scale
        )
        b_scale_8_2d = (
            b_scale_8.reshape(-1, b_scale_8.shape[-1])
            if b_scale_8.ndim != 2
            else b_scale_8
        )

        if hasattr(torch.ops.mslk, "mx8mx8bf16_grouped_mm"):
            t_mx8 = benchmark_fn(
                lambda _aq=aq,
                _bq=bq_8_col,
                _as=a_scale_2d,
                _bs=b_scale_8_2d,
                _off=offsets,
                _out=out_mx8: torch.ops.mslk.mx8mx8bf16_grouped_mm(
                    _aq, _bq, _as, _bs, _off, _out
                ),
                iters=num_iters,
            )
            tflops_mx8 = tflops(M, N, K, t_mx8)
        else:
            tflops_mx8 = float("nan")

        # MX8×MX4 (B quantized to MX4 via triton)
        bq_4, b_scale_4 = triton_quantize_mx4_unpack(B_small)
        bq_4_cast = bq_4.view(torch.float4_e2m1fn_x2)

        if hasattr(torch.ops.mslk, "mx8mx4bf16"):
            t_mx4 = benchmark_fn(
                lambda _aq=aq,
                _bq=bq_4_cast,
                _as=a_scale,
                _bs=b_scale_4: torch.ops.mslk.mx8mx4bf16(_aq, _bq, _as, _bs),
                iters=num_iters,
            )
            tflops_mx4 = tflops(M, N, K, t_mx4)
        else:
            tflops_mx4 = float("nan")

        # MX8×MX6 (raw uint8 packed-6 weights; reuse MX4-style block scales).
        # The mx8mx6bf16 op is optional — it only exists once D103697749 lands.
        wq_6 = torch.randint(0, 256, (N, K * 6 // 8), dtype=torch.uint8, device=device)

        if hasattr(torch.ops.mslk, "mx8mx6bf16"):
            t_mx8mx6 = benchmark_fn(
                lambda _aq=aq,
                _wq=wq_6,
                _as=a_scale,
                _bs=b_scale_4: torch.ops.mslk.mx8mx6bf16(_aq, _wq, _as, _bs),
                iters=num_iters,
            )
            tflops_mx8mx6 = tflops(M, N, K, t_mx8mx6)
        else:
            tflops_mx8mx6 = float("nan")

        # MX6×MX6 (both A and B raw uint8 packed-6; reuse MX4-style block scales)
        aq_6 = torch.randint(0, 256, (M, K * 6 // 8), dtype=torch.uint8, device=device)
        # Build a placeholder MX4-style A scale to match the kernel scale layout.
        _, a_scale_6 = triton_quantize_mx4_unpack(A_small)

        t_mx6mx6 = benchmark_fn(
            lambda _aq=aq_6,
            _wq=wq_6,
            _as=a_scale_6,
            _bs=b_scale_4: torch.ops.mslk.mx6mx6bf16(_aq, _wq, _as, _bs),
            iters=num_iters,
        )
        tflops_mx6mx6 = tflops(M, N, K, t_mx6mx6)

        # f4f4bf16 (native NVFP4 — both operands MX4)
        aq_4, a_scale_4 = triton_quantize_mx4_unpack(A_small)
        aq_4_cast = aq_4.view(torch.float4_e2m1fn_x2)

        if hasattr(torch.ops.mslk, "f4f4bf16"):
            t_f4f4 = benchmark_fn(
                lambda _aq=aq_4_cast,
                _bq=bq_4_cast,
                _as=a_scale_4,
                _bs=b_scale_4: torch.ops.mslk.f4f4bf16(_aq, _bq, _as, _bs),
                iters=num_iters,
            )
            tflops_f4f4 = tflops(M, N, K, t_f4f4)
        else:
            tflops_f4f4 = float("nan")

        print(
            f"{M:>6} | {tflops_bf16:>10.1f} | {tflops_mx8:>8.1f} | {tflops_mx8mx6:>8.1f} | {tflops_mx6mx6:>8.1f} | {tflops_mx4:>8.1f} | {tflops_f4f4:>8.1f}"
        )

    print()
    print(
        "Note: SQNR omitted — no Python MX6 quantizer is wired up, so MX6 weights "
        "and (for mx6mx6) MX6 activations are random uint8."
    )


if __name__ == "__main__":
    main()
