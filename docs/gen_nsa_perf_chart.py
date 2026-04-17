#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Generate NSA performance chart from benchmark data."""

import datetime
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # Benchmark data from GB200 (B=1, H=32, H_kv=8, D=128)
    # Measured 2026-03-26 — with all optimizations (mask construction, in-place grad accum)
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
        0.27,
        0.19,
        0.27,
        0.55,
        1.52,
        5.44,
        24.79,
        99.62,
        397.26,
        1600.45,
        6413.33,
    ]
    nsa_fwd_ms = [
        1.79,
        2.54,
        2.53,
        2.24,
        3.15,
        5.78,
        10.78,
        24.15,
        63.07,
        188.18,
        629.22,
    ]

    # Fwd+Bwd: Dense FA4 vs NSA
    dense_fwdbwd_ms = [
        0.56,
        0.57,
        0.78,
        1.65,
        5.11,
        21.02,
        84.46,
        340.14,
        1339.51,
        5374.25,
        27270.16,
    ]
    nsa_fwdbwd_ms = [
        3.70,
        3.62,
        3.51,
        5.32,
        8.02,
        14.17,
        29.46,
        68.41,
        179.66,
        540.77,
        2432.94,
    ]

    # Varlen Fwd+Bwd (2 packed sequences, 60/40 split)
    nsa_varlen_fwdbwd_ms = [
        5.34,
        5.55,
        5.90,
        7.22,
        10.00,
        17.01,
        32.83,
        67.86,
        156.67,
        411.58,
        1224.64,
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

    # --- Panel 2: Fwd+Bwd latency (with varlen) ---
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
        label="NSA (padded)",
        linewidth=2,
        markersize=6,
    )
    ax.loglog(
        seq_lengths,
        nsa_varlen_fwdbwd_ms,
        "^--",
        color="#1f77b4",
        label="NSA (varlen)",
        linewidth=2,
        markersize=5,
    )
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Fwd + Bwd Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([1024, 4096, 16384, 65536, 262144, 1048576])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K", "1M"], fontsize=10)
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
    ax.set_xticks([1024, 4096, 16384, 65536, 262144, 1048576])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K", "1M"], fontsize=10)
    ax.annotate(
        f"{fwdbwd_speedup[-1]:.1f}x",
        xy=(1048576, fwdbwd_speedup[-1]),
        xytext=(300000, fwdbwd_speedup[-1] - 1.5),
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
        color="#ff7f0e",
    )
    ax.annotate(
        f"{fwd_speedup[-1]:.1f}x",
        xy=(1048576, fwd_speedup[-1]),
        xytext=(300000, fwd_speedup[-1] + 0.5),
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

    plt.savefig("docs/nsa_perf.svg", format="svg", bbox_inches="tight", dpi=150)
    print("Saved docs/nsa_perf.svg")


if __name__ == "__main__":
    main()
