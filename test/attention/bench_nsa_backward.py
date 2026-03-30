# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Benchmark NSA forward + backward at various sequence lengths."""

import gc
import time

import torch


def benchmark_nsa_fwd_bwd(
    N: int,
    B: int = 1,
    H: int = 32,
    H_kv: int = 8,
    D: int = 128,
    compress_block_size: int = 64,
    num_selected_blocks: int = 16,
    window_size: int = 512,
    warmup: int = 3,
    iters: int = 10,
    backward: bool = True,
):
    from mslk.attention.sparse_attn import nsa

    Q = torch.randn(
        B, N, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )
    K = torch.randn(
        B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )
    V = torch.randn(
        B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=backward
    )

    def run():
        out = nsa(
            Q,
            K,
            V,
            compress_block_size=compress_block_size,
            num_selected_blocks=num_selected_blocks,
            window_size=window_size,
        )
        if backward:
            loss = out.sum()
            loss.backward()
            Q.grad = None
            K.grad = None
            V.grad = None
        return out

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    tflops_fwd = 2 * B * H * N * N * D / 1e12  # approximate (ignoring sparsity)
    mode = "fwd+bwd" if backward else "fwd"
    print(
        f"N={N:>7d} | {mode} | {elapsed * 1000:>8.2f} ms | "
        f"~{tflops_fwd / elapsed / 1e3:.1f} TFLOPS (dense equiv)"
    )
    return elapsed


def main():
    print("NSA Forward + Backward Benchmark")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    seq_lengths = [
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 3145728,
    ]

    for N in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            benchmark_nsa_fwd_bwd(N, backward=False)
        except Exception as e:
            print(f"N={N:>7d} | fwd     | FAILED: {e}")

    print()
    for N in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            benchmark_nsa_fwd_bwd(N, backward=True)
        except Exception as e:
            print(f"N={N:>7d} | fwd+bwd | FAILED: {e}")


if __name__ == "__main__":
    main()
