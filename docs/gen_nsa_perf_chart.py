#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Generate NSA performance chart from benchmark data."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Benchmark data from GB200 (B=1, H=32, H_kv=8, D=128)
    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

    # Forward: Dense FA4 vs NSA
    dense_fwd_ms = [0.12, 0.26, 0.26, 0.53, 1.54, 5.47, 24.02, 100.17, 400.45, 1589.36, 6356.49]
    nsa_fwd_ms = [1.54, 2.54, 2.48, 2.57, 4.09, 6.09, 10.31, 23.89, 62.83, 187.37, 623.98]

    # NSA fwd+bwd
    nsa_bwd_seq = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    nsa_fwdbwd_ms = [5.77, 7.98, 12.60, 22.25, 40.29, 77.73, 156.98, 327.90, 722.12, 1661.73]
    nsa_bwd_only_ms = [fb - f for f, fb in zip(nsa_fwd_ms[: len(nsa_bwd_seq)], nsa_fwdbwd_ms)]

    fwd_speedup = [d / n for d, n in zip(dense_fwd_ms, nsa_fwd_ms)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.loglog(seq_lengths, dense_fwd_ms, "o-", color="#d62728", label="Dense FA4", linewidth=2, markersize=6)
    ax.loglog(seq_lengths, nsa_fwd_ms, "s-", color="#2ca02c", label="NSA Sparse", linewidth=2, markersize=6)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Forward Pass Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([1024, 4096, 16384, 65536, 262144, 1048576])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K", "1M"], fontsize=10)
    ax.axvline(x=32768, color="gray", linestyle="--", alpha=0.5)
    ax.annotate("NSA wins\n(>32K)", xy=(65536, 8), fontsize=9, color="gray", ha="center")

    ax = axes[1]
    ax.semilogx(seq_lengths, fwd_speedup, "D-", color="#1f77b4", linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(seq_lengths, 1.0, fwd_speedup, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Speedup (Dense / NSA)", fontsize=12)
    ax.set_title("NSA Forward Speedup vs Dense", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([1024, 4096, 16384, 65536, 262144, 1048576])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K", "1M"], fontsize=10)
    ax.annotate("10.2x", xy=(1048576, 10.19), xytext=(300000, 8.5), fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

    ax = axes[2]
    bar_labels = ["1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K"]
    x = np.arange(len(bar_labels))
    fwd_vals = nsa_fwd_ms[: len(nsa_bwd_seq)]
    ax.bar(x, fwd_vals, 0.4, label="Forward", color="#2ca02c", alpha=0.8)
    ax.bar(x, nsa_bwd_only_ms, 0.4, bottom=fwd_vals, label="Backward", color="#ff7f0e", alpha=0.8)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("NSA Fwd + Bwd Breakdown", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9, rotation=45)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    ax.annotate("OOM at 1M\n(FA4 block-sparse\nbwd needed)", xy=(9.5, 1500), fontsize=9, color="#d62728",
                ha="center", style="italic")

    plt.suptitle("NSA Sparse Attention Performance — GB200 (B=1, H=32, H_kv=8, D=128)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("docs/nsa_bwd_perf.svg", format="svg", bbox_inches="tight", dpi=150)
    print("Saved docs/nsa_bwd_perf.svg")


if __name__ == "__main__":
    main()
