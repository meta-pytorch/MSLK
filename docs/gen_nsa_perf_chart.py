#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Generate NSA performance chart from benchmark data."""

import datetime
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Benchmark data from B200, devgpu016 (B=1, H=32, H_kv=8, D=128)
    # Measured 2026-03-25
    seq_lengths = [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
    ]

    # Forward: Dense FA4 vs NSA
    dense_fwd_ms = [
        0.09,
        0.10,
        0.17,
        0.48,
        1.74,
        6.93,
        30.61,
        121.16,
        485.24,
        1939.75,
        7816.42,
    ]
    nsa_fwd_ms = [
        0.91,
        1.02,
        1.28,
        1.77,
        2.84,
        5.27,
        11.22,
        27.09,
        74.81,
        226.35,
        772.59,
    ]

    # Fwd+Bwd: Dense FA4 vs NSA
    dense_fwdbwd_ms = [
        0.41,
        0.44,
        0.68,
        1.85,
        6.31,
        27.37,
        108.85,
        418.86,
        1669.69,
        6644.73,
        26581.04,
    ]
    nsa_fwdbwd_ms = [
        2.42,
        2.34,
        2.63,
        4.28,
        7.62,
        15.45,
        34.31,
        86.48,
        236.09,
        695.13,
        2309.82,
    ]

    fwd_speedup = [d / n for d, n in zip(dense_fwd_ms, nsa_fwd_ms)]
    fwdbwd_speedup = [d / n for d, n in zip(dense_fwdbwd_ms, nsa_fwdbwd_ms)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Forward latency ---
    ax = axes[0]
    ax.loglog(
        seq_lengths,
        dense_fwd_ms,
        "o-",
        color="#d62728",
        label="Dense FA4",
        linewidth=2,
        markersize=6,
    )
    ax.loglog(
        seq_lengths,
        nsa_fwd_ms,
        "s-",
        color="#2ca02c",
        label="NSA Sparse",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Forward Pass Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([1024, 4096, 16384, 65536, 262144, 1048576])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K", "1M"], fontsize=10)
    ax.axvline(x=25000, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        "NSA wins\n(>~25K)", xy=(50000, 6), fontsize=9, color="gray", ha="center"
    )

    # --- Panel 2: Fwd+Bwd latency (NEW: shows both dense and NSA) ---
    ax = axes[1]
    ax.loglog(
        seq_lengths,
        dense_fwdbwd_ms,
        "o-",
        color="#d62728",
        label="Dense FA4",
        linewidth=2,
        markersize=6,
    )
    ax.loglog(
        seq_lengths,
        nsa_fwdbwd_ms,
        "s-",
        color="#2ca02c",
        label="NSA Sparse",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Fwd + Bwd Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([1024, 4096, 16384, 65536, 262144, 1048576])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K", "1M"], fontsize=10)
    ax.axvline(x=20000, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        "NSA wins\n(>~20K)", xy=(40000, 12), fontsize=9, color="gray", ha="center"
    )

    # --- Panel 3: Speedup comparison (fwd vs fwd+bwd) ---
    ax = axes[2]
    ax.semilogx(
        seq_lengths,
        fwd_speedup,
        "D-",
        color="#1f77b4",
        label="Forward only",
        linewidth=2,
        markersize=6,
    )
    ax.semilogx(
        seq_lengths,
        fwdbwd_speedup,
        "s-",
        color="#ff7f0e",
        label="Fwd + Bwd",
        linewidth=2,
        markersize=6,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(seq_lengths, 1.0, fwd_speedup, alpha=0.10, color="#1f77b4")
    ax.fill_between(seq_lengths, 1.0, fwdbwd_speedup, alpha=0.10, color="#ff7f0e")
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Speedup (Dense / NSA)", fontsize=12)
    ax.set_title("NSA Speedup vs Dense FA4", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([1024, 4096, 16384, 65536, 262144, 1048576])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K", "1M"], fontsize=10)
    ax.annotate(
        "11.5x",
        xy=(1048576, 11.51),
        xytext=(300000, 10.0),
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
        color="#ff7f0e",
    )
    ax.annotate(
        "10.1x",
        xy=(1048576, 10.12),
        xytext=(300000, 7.5),
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
        color="#1f77b4",
    )

    plt.suptitle(
        "NSA Sparse Attention Performance — B200 (B=1, H=32, H_kv=8, D=128)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Add hostname and timestamp
    hostname = platform.node()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99,
        -0.02,
        f"{hostname}  |  {timestamp}",
        fontsize=8,
        color="gray",
        ha="right",
        va="top",
    )

    plt.savefig("docs/nsa_bwd_perf.svg", format="svg", bbox_inches="tight", dpi=150)
    print("Saved docs/nsa_bwd_perf.svg")


if __name__ == "__main__":
    main()
