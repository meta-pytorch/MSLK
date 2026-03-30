# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Performance benchmark: NSA vs dense FA4 attention."""

import pytest
import torch


def _benchmark_fn(fn, warmup=5, repeat=20):
    """Benchmark a function using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = sorted(times)
    # Use median
    return times[len(times) // 2]


class TestBenchmark:
    """Performance comparison between NSA and dense FA4."""

    @pytest.mark.parametrize("N", [8192, 16384, 32768, 65536, 131072])
    def test_nsa_benchmark(self, N) -> None:
        """Benchmark NSA vs dense FA4 and report speedup."""
        from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func
        from mslk.attention.sparse_attn.nsa_forward import nsa_forward

        B, H, H_kv, D = 2, 32, 32, 128
        dtype = torch.bfloat16

        Q = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)

        # Dense FA4 baseline
        dense_ms = _benchmark_fn(
            lambda: flash_attn_func(Q, K, V, causal=True)
        )

        # NSA
        nsa_ms = _benchmark_fn(
            lambda: nsa_forward(
                Q, K, V,
                compress_block_size=64,
                num_selected_blocks=16,
                window_size=512,
                causal=True,
            )
        )

        speedup = dense_ms / nsa_ms
        print(
            f"\nN={N}: dense={dense_ms:.2f}ms, NSA={nsa_ms:.2f}ms, "
            f"speedup={speedup:.2f}x"
        )

        # At large N, we expect NSA to be competitive
        # (don't fail the test, just report)
