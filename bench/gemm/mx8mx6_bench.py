#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Benchmark: MX8×MX6 CUTLASS block-scaled GEMM kernel (mx8mx6bf16)
# Compares against BF16 matmul, MX8×MX4, and native NVFP4 (f4f4bf16).

import time

import mslk.gemm  # noqa: F401
import mslk.quantize  # noqa: F401
import torch
from mslk.gemm.triton.fp8_gemm import to_mxfp8
from mslk.quantize.triton.fp4_quantize import _to_blocked, triton_quantize_mx4_unpack


def benchmark_fn(fn, warmup=5, iters=20):
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


def tflops(M, N, K, time_ms):
    """Compute TFLOPS from GEMM dimensions and time."""
    flops = 2 * M * N * K
    return flops / (time_ms * 1e-3) / 1e12


def main():
    device = torch.accelerator.current_accelerator()
    N = 16384
    K = 16384
    num_iters = 20
    M_values = [1, 16, 64, 128, 256, 512, 1024]

    print(
        f"Benchmark on {torch.cuda.get_device_name()}, N=K={N}, num_iters={num_iters}"
    )
    print()

    # Header
    hdr = (
        f"{'M':>6} | {'bf16_mm':>10} | {'mx8mx8':>8}"
        f" | {'mx8mx6':>8} | {'mx8mx4':>8} | {'f4f4':>8}"
    )
    sub = (
        f"{'':>6} | {'(TFLOPS)':>10} | {'(TFLOPS)':>8}"
        f" | {'(TFLOPS)':>8} | {'(TFLOPS)':>8} | {'(TFLOPS)':>8}"
    )
    print(hdr)
    print(sub)
    print("-" * 72)

    results = []

    for M in M_values:
        # BF16 matmul reference
        A_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device=device)
        B_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device=device)

        t_bf16 = benchmark_fn(lambda _a=A_bf16, _b=B_bf16: _a @ _b.t(), iters=num_iters)
        tflops_bf16 = tflops(M, N, K, t_bf16)

        # Quantize A to MX8
        A_small = A_bf16 * 0.1
        B_small = B_bf16 * 0.01
        (a_scale_raw, aq) = to_mxfp8(A_small)
        a_scale = _to_blocked(a_scale_raw.view(torch.int8).reshape(M, -1)).view(
            torch.uint8
        )

        # MX8×MX8: use grouped_mm with single group (offsets=[K])
        (b_scale_raw_8, bq_8) = to_mxfp8(B_small)
        b_scale_8 = _to_blocked(b_scale_raw_8.view(torch.int8).reshape(N, -1)).view(
            torch.uint8
        )
        # grouped_mm expects column-major WQ
        bq_8_col = bq_8.t()
        offsets = torch.tensor([K], dtype=torch.int32, device=device)
        out_mx8 = torch.empty((1, M, N), dtype=torch.bfloat16, device=device)
        # Ensure scales are explicitly 2D for grouped_mm
        a_scale_2d = (
            a_scale.reshape(-1, a_scale.shape[-1]) if a_scale.ndim != 2 else a_scale
        )
        b_scale_8_2d = (
            b_scale_8.reshape(-1, b_scale_8.shape[-1])
            if b_scale_8.ndim != 2
            else b_scale_8
        )

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

        # MX8×MX4: quantize B to MX4
        bq_4, b_scale_4 = triton_quantize_mx4_unpack(B_small)
        bq_4_cast = bq_4.view(torch.float4_e2m1fn_x2)

        t_mx4 = benchmark_fn(
            lambda _aq=aq,
            _bq=bq_4_cast,
            _as=a_scale,
            _bs=b_scale_4: torch.ops.mslk.mx8mx4bf16(_aq, _bq, _as, _bs),
            iters=num_iters,
        )
        tflops_mx4 = tflops(M, N, K, t_mx4)

        # MX8×MX6: create MX6 weight tensor (raw uint8, 6-bit packed)
        wq_6 = torch.randint(0, 256, (N, K * 6 // 8), dtype=torch.uint8, device=device)

        t_mx6 = benchmark_fn(
            lambda _aq=aq,
            _wq=wq_6,
            _as=a_scale,
            _bs=b_scale_4: torch.ops.mslk.mx8mx6bf16(_aq, _wq, _as, _bs),
            iters=num_iters,
        )
        tflops_mx6 = tflops(M, N, K, t_mx6)

        # f4f4bf16 (native NVFP4 — both operands MX4)
        aq_4, a_scale_4 = triton_quantize_mx4_unpack(A_small)
        aq_4_cast = aq_4.view(torch.float4_e2m1fn_x2)

        t_f4f4 = benchmark_fn(
            lambda _aq=aq_4_cast,
            _bq=bq_4_cast,
            _as=a_scale_4,
            _bs=b_scale_4: torch.ops.mslk.f4f4bf16(_aq, _bq, _as, _bs),
            iters=num_iters,
        )
        tflops_f4f4 = tflops(M, N, K, t_f4f4)

        print(
            f"{M:>6} | {tflops_bf16:>10.1f} | {tflops_mx8:>8.1f}"
            f" | {tflops_mx6:>8.1f} | {tflops_mx4:>8.1f}"
            f" | {tflops_f4f4:>8.1f}"
        )
        results.append(
            (M, tflops_bf16, tflops_mx8, tflops_mx6, tflops_mx4, tflops_f4f4)
        )

    # SQNR comparison (only for MX4 since MX6 has no Python quantizer)
    print()
    print("Output SQNR vs BF16 (dB, higher is better):")
    print(f"{'M':>6} | {'mx8mx8bf16':>12} | {'mx8mx4bf16':>12} | {'f4f4bf16':>10}")
    print("-" * 50)

    for M in [16, 64, 256, 1024]:
        A_bf16 = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.1
        B_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device=device) * 0.01

        ref = A_bf16 @ B_bf16.t()

        # MX8 activation
        (a_scale_raw, aq) = to_mxfp8(A_bf16)
        a_scale = _to_blocked(a_scale_raw.view(torch.int8).reshape(M, -1)).view(
            torch.uint8
        )

        # MX8×MX8
        (b_scale_raw_8, bq_8) = to_mxfp8(B_bf16)
        b_scale_8 = _to_blocked(b_scale_raw_8.view(torch.int8).reshape(N, -1)).view(
            torch.uint8
        )
        bq_8_col = bq_8.t()
        offsets = torch.tensor([K], dtype=torch.int32, device=device)
        a_scale_2d = (
            a_scale.reshape(-1, a_scale.shape[-1]) if a_scale.ndim != 2 else a_scale
        )
        b_scale_8_2d = (
            b_scale_8.reshape(-1, b_scale_8.shape[-1])
            if b_scale_8.ndim != 2
            else b_scale_8
        )
        out_mx8_buf = torch.empty((1, M, N), dtype=torch.bfloat16, device=device)
        torch.ops.mslk.mx8mx8bf16_grouped_mm(
            aq, bq_8_col, a_scale_2d, b_scale_8_2d, offsets, out_mx8_buf
        )
        out_mx8 = out_mx8_buf.squeeze(0)

        # MX4 weight
        bq_4, b_scale_4 = triton_quantize_mx4_unpack(B_bf16)
        bq_4_cast = bq_4.view(torch.float4_e2m1fn_x2)
        out_mx4 = torch.ops.mslk.mx8mx4bf16(aq, bq_4_cast, a_scale, b_scale_4)

        # f4f4bf16
        aq_4, a_scale_4 = triton_quantize_mx4_unpack(A_bf16)
        aq_4_cast = aq_4.view(torch.float4_e2m1fn_x2)
        out_f4f4 = torch.ops.mslk.f4f4bf16(aq_4_cast, bq_4_cast, a_scale_4, b_scale_4)

        # Compute SQNR
        def sqnr(signal, noise):
            diff = signal - noise
            snr = 10 * torch.log10((signal**2).sum() / ((diff**2).sum() + 1e-20))
            return snr.item()

        sqnr_mx8 = sqnr(ref.float(), out_mx8.float())
        sqnr_mx4 = sqnr(ref.float(), out_mx4.float())
        sqnr_f4f4 = sqnr(ref.float(), out_f4f4.float())

        print(f"{M:>6} | {sqnr_mx8:>12.2f} | {sqnr_mx4:>12.2f} | {sqnr_f4f4:>10.2f}")

    print()
    print("Note: MX6 SQNR omitted (no Python MX6 quantizer yet).")
    print("MX6 SQNR expected to be between MX8×MX8 (~28 dB) and MX8×MX4 (~18 dB).")


if __name__ == "__main__":
    main()
