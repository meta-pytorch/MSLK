# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Test CUDA graph capture of NSA forward and backward passes."""

import gc
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault(
    "CUTLASS_CUTE_DSL_KERNEL_CACHE_DIR",
    os.path.expanduser("~/.cache/cutlass_dsl_kernels"),
)

import torch


def test_cuda_graph_nsa_forward():
    """Test that NSA forward can be captured in a CUDA graph."""
    from mslk.attention.sparse_attn import nsa_forward

    print("CUDA Graph NSA Forward Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    for N in [4096, 8192, 16384, 32768, 65536, 131072]:
        B, H, H_kv, D = 1, 32, 8, 128

        gc.collect()
        torch.cuda.empty_cache()

        Q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(B, N, H_kv, D, device="cuda", dtype=torch.bfloat16)

        # Warmup — compile all CuTe DSL kernels
        for _ in range(3):
            _ = nsa_forward(Q, K, V)
        torch.cuda.synchronize()

        # Benchmark: no graph
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            _ = nsa_forward(Q, K, V)
        torch.cuda.synchronize()
        no_graph_ms = (time.perf_counter() - t0) / 20 * 1000

        # Capture CUDA graph
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = nsa_forward(Q, K, V)

            # Benchmark: with graph
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()
            graph_ms = (time.perf_counter() - t0) / 20 * 1000

            speedup = no_graph_ms / graph_ms
            print(
                f"N={N:>6d} | no_graph={no_graph_ms:>6.2f}ms | "
                f"graph={graph_ms:>6.2f}ms | speedup={speedup:.2f}x"
            )
            del g
        except Exception as e:
            print(f"N={N:>6d} | no_graph={no_graph_ms:>6.2f}ms | graph=FAILED: {e}")

        del Q, K, V


def test_cuda_graph_nsa_fwd_bwd():
    """Test that NSA forward+backward can be captured in a CUDA graph."""
    from mslk.attention.sparse_attn import nsa

    print("CUDA Graph NSA Fwd+Bwd Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    for N in [4096, 8192, 16384, 32768, 65536]:
        B, H, H_kv, D = 1, 32, 8, 128

        gc.collect()
        torch.cuda.empty_cache()

        # Static tensors for graph capture — must persist across replay
        Q = torch.randn(
            B, N, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        K = torch.randn(
            B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        V = torch.randn(
            B, N, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        # Warmup — compile all CuTe DSL kernels (fwd + bwd)
        for _ in range(3):
            out = nsa(Q, K, V)
            out.sum().backward()
            Q.grad = K.grad = V.grad = None
        torch.cuda.synchronize()

        # Benchmark: no graph
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            out = nsa(Q, K, V)
            out.sum().backward()
            Q.grad = K.grad = V.grad = None
        torch.cuda.synchronize()
        no_graph_ms = (time.perf_counter() - t0) / 10 * 1000

        # Capture CUDA graph for fwd+bwd using make_graphed_callables
        try:
            # make_graphed_callables wraps a module to use CUDA graphs
            # for both forward and backward
            nsa_graphed = torch.cuda.make_graphed_callables(
                lambda q, k, v: nsa(q, k, v),
                (Q, K, V),
            )

            # Benchmark: with graph
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                out = nsa_graphed(Q, K, V)
                out.sum().backward()
                Q.grad = K.grad = V.grad = None
            torch.cuda.synchronize()
            graph_ms = (time.perf_counter() - t0) / 10 * 1000

            speedup = no_graph_ms / graph_ms
            print(
                f"N={N:>6d} | no_graph={no_graph_ms:>6.2f}ms | "
                f"graph={graph_ms:>6.2f}ms | speedup={speedup:.2f}x"
            )
            del nsa_graphed
        except Exception as e:
            print(f"N={N:>6d} | no_graph={no_graph_ms:>6.2f}ms | graph=FAILED: {e}")

        del Q, K, V


def main():
    test_cuda_graph_nsa_forward()
    print()
    test_cuda_graph_nsa_fwd_bwd()


if __name__ == "__main__":
    main()
