# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

"""Benchmark CK FMHA with FA4 conditional rescaling."""

import math

import torch
from mslk.attention import fmha


def bench(B, H, M, K, dtype=torch.bfloat16, n_warmup=10, n_iters=50):
    q = torch.randn(B, M, H, K, device="cuda", dtype=dtype)
    k = torch.randn(B, M, H, K, device="cuda", dtype=dtype)
    v = torch.randn(B, M, H, K, device="cuda", dtype=dtype)

    # Correctness check
    out = fmha.memory_efficient_attention_forward(
        q, k, v, attn_bias=None, op=fmha.ck.FwOp
    )
    assert not out.isnan().any(), "NaN in output!"

    # Reference check
    q_t = q.float().transpose(1, 2)
    k_t = k.float().transpose(1, 2)
    v_t = v.float().transpose(1, 2)
    ref = torch.matmul(
        torch.softmax(torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(K), dim=-1),
        v_t,
    ).transpose(1, 2)
    max_diff = (out.float() - ref).abs().max().item()
    mean_diff = (out.float() - ref).abs().mean().item()

    # Warmup
    for _ in range(n_warmup):
        fmha.memory_efficient_attention_forward(
            q, k, v, attn_bias=None, op=fmha.ck.FwOp
        )
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fmha.memory_efficient_attention_forward(
            q, k, v, attn_bias=None, op=fmha.ck.FwOp
        )
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / n_iters
    flops = 4 * B * H * M * M * K
    tflops = flops / (elapsed_ms * 1e-3) / 1e12

    print(
        f"  B={B:2d} H={H:2d} M={M:5d} K={K:3d} {str(dtype):20s} | "
        f"{elapsed_ms:8.2f} ms  {tflops:6.1f} TFLOPS | "
        f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}"
    )


def main():
    print("CK FMHA FA4 Benchmark (MI350x)")
    print("=" * 100)
    print(f"  {'Shape':50s} | {'Time':8s}  {'Perf':6s}        | {'Accuracy':30s}")
    print("-" * 100)

    # Typical Flux2 shapes
    bench(B=1, H=24, M=4096, K=128, dtype=torch.bfloat16)
    bench(B=1, H=24, M=4096, K=128, dtype=torch.float16)

    # Various sequence lengths
    bench(B=4, H=32, M=1024, K=128, dtype=torch.bfloat16)
    bench(B=4, H=32, M=2048, K=128, dtype=torch.bfloat16)
    bench(B=4, H=32, M=4096, K=128, dtype=torch.bfloat16)

    # Different head dims
    bench(B=2, H=8, M=2048, K=64, dtype=torch.bfloat16)
    bench(B=2, H=8, M=2048, K=128, dtype=torch.bfloat16)
    bench(B=2, H=8, M=2048, K=256, dtype=torch.bfloat16)

    # Small batch, many heads (GPT-like)
    bench(B=1, H=64, M=2048, K=64, dtype=torch.bfloat16)

    print("=" * 100)
    print("All benchmarks passed correctness check!")


if __name__ == "__main__":
    main()
