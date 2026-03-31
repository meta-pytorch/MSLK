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
    # Benchmark data from GB200 (B=1, H=32, H_kv=8, D=128)
    # Measured 2026-03-26
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
    ]

    # Forward: Dense FA4 vs NSA
    dense_fwd_ms = [
        0.10,
        0.19,
        0.27,
        0.53,
        1.53,
        5.16,
        24.06,
        96.53,
        403.51,
        1617.79,
    ]
    nsa_fwd_ms = [
        1.46,
        2.34,
        2.27,
        2.17,
        3.87,
        5.72,
        10.35,
        24.18,
        65.11,
        194.81,
    ]

    # Fwd+Bwd: Dense FA4 vs NSA
    dense_fwdbwd_ms = [
        0.57,
        0.55,
        0.76,
        1.63,
        5.06,
        21.35,
        84.15,
        338.68,
        1356.51,
        5423.31,
    ]
    nsa_fwdbwd_ms = [
        3.33,
        3.83,
        3.62,
        4.51,
        7.45,
        13.83,
        29.00,
        68.78,
        182.45,
        833.78,
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
    ax.set_xticks([1024, 4096, 16384, 65536, 262144])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K"], fontsize=10)
    ax.axvline(x=32000, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        "NSA wins\n(>~32K)", xy=(50000, 6), fontsize=9, color="gray", ha="center"
    )

    # --- Panel 2: Fwd+Bwd latency ---
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
    ax.set_xticks([1024, 4096, 16384, 65536, 262144])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K"], fontsize=10)
    ax.axvline(x=32000, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        "NSA wins\n(>~32K)", xy=(50000, 12), fontsize=9, color="gray", ha="center"
    )

    # --- Panel 3: Speedup comparison ---
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
    ax.set_xticks([1024, 4096, 16384, 65536, 262144])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K"], fontsize=10)

    max_fwd_speedup = max(fwd_speedup)
    max_fwdbwd_speedup = max(fwdbwd_speedup)
    ax.annotate(
        f"{max_fwdbwd_speedup:.1f}x",
        xy=(seq_lengths[fwdbwd_speedup.index(max_fwdbwd_speedup)], max_fwdbwd_speedup),
        xytext=(100000, max_fwdbwd_speedup - 1),
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
        color="#ff7f0e",
    )
    ax.annotate(
        f"{max_fwd_speedup:.1f}x",
        xy=(seq_lengths[fwd_speedup.index(max_fwd_speedup)], max_fwd_speedup),
        xytext=(100000, max_fwd_speedup - 1.5),
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1f77b4"),
        color="#1f77b4",
    )

    plt.suptitle(
        "NSA Sparse Attention Performance — GB200 (B=1, H=32, H_kv=8, D=128)",
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
