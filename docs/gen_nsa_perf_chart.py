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
    # Measured 2026-03-28 — post-optimization (shared selector, compact metadata,
    # compress_factor, pure PyTorch compress+gating)

    # Forward data (4K–2M)
    fwd_seq_lengths = [
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
    ]

    dense_fwd_ms = [
        0.23,
        0.66,
        1.58,
        5.50,
        25.29,
        97.71,
        407.59,
        1633.15,
        6537.68,
        25594.57,
    ]
    nsa_fwd_ms = [
        2.97,
        1.51,
        2.76,
        3.99,
        7.16,
        12.34,
        26.30,
        71.33,
        228.28,
        899.15,
    ]
    # NSA+CmpSparse: no 2M data yet (not benchmarked)
    nsa_cmpsparse_fwd_seq = [
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
    nsa_cmpsparse_fwd_ms = [1.65, 2.42, 3.11, 4.02, 7.04, 12.69, 30.17, 68.32, 171.49]

    # Fwd+Bwd data (64K–1M, warmed runs only)
    fwdbwd_seq_lengths = [65536, 131072, 262144, 524288, 1048576]

    dense_fwdbwd_ms = [25.19, 98.72, 406.35, 1632.08, 6531.88]
    nsa_fwdbwd_ms = [11.47, 21.30, 44.79, 108.71, 303.52]

    fwd_speedup = [d / n for d, n in zip(dense_fwd_ms, nsa_fwd_ms)]
    # CmpSparse speedup uses its own (shorter) seq_lengths list
    dense_fwd_cmpsparse = dense_fwd_ms[: len(nsa_cmpsparse_fwd_ms)]
    fwd_cmpsparse_speedup = [
        d / n for d, n in zip(dense_fwd_cmpsparse, nsa_cmpsparse_fwd_ms)
    ]
    fwdbwd_speedup = [d / n for d, n in zip(dense_fwdbwd_ms, nsa_fwdbwd_ms)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Forward latency ---
    ax = axes[0]
    ax.loglog(
        fwd_seq_lengths,
        dense_fwd_ms,
        "o-",
        color="#d62728",
        label="Dense FA4",
        linewidth=2,
        markersize=6,
    )
    ax.loglog(
        fwd_seq_lengths,
        nsa_fwd_ms,
        "s-",
        color="#2ca02c",
        label="NSA Sparse",
        linewidth=2,
        markersize=6,
    )
    ax.loglog(
        nsa_cmpsparse_fwd_seq,
        nsa_cmpsparse_fwd_ms,
        "^-",
        color="#1f77b4",
        label="NSA + CmpSparse",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Forward Pass Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([4096, 16384, 65536, 262144, 1048576, 2097152])
    ax.set_xticklabels(["4K", "16K", "64K", "256K", "1M", "2M"], fontsize=10)

    # --- Panel 2: Fwd+Bwd latency ---
    ax = axes[1]
    ax.loglog(
        fwdbwd_seq_lengths,
        dense_fwdbwd_ms,
        "o-",
        color="#d62728",
        label="Dense FA4",
        linewidth=2,
        markersize=6,
    )
    ax.loglog(
        fwdbwd_seq_lengths,
        nsa_fwdbwd_ms,
        "s-",
        color="#2ca02c",
        label="NSA Sparse",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Fwd + Bwd Latency (warmed)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([65536, 131072, 262144, 524288, 1048576])
    ax.set_xticklabels(["64K", "128K", "256K", "512K", "1M"], fontsize=10)

    # --- Panel 3: Speedup comparison ---
    ax = axes[2]
    ax.semilogx(
        fwd_seq_lengths,
        fwd_speedup,
        "D-",
        color="#1f77b4",
        label="Fwd (NSA)",
        linewidth=2,
        markersize=6,
    )
    ax.semilogx(
        nsa_cmpsparse_fwd_seq,
        fwd_cmpsparse_speedup,
        "^-",
        color="#9467bd",
        label="Fwd (NSA+CmpSparse)",
        linewidth=2,
        markersize=6,
    )
    ax.semilogx(
        fwdbwd_seq_lengths,
        fwdbwd_speedup,
        "s-",
        color="#ff7f0e",
        label="Fwd+Bwd (NSA)",
        linewidth=2,
        markersize=6,
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(fwd_seq_lengths, 1.0, fwd_speedup, alpha=0.08, color="#1f77b4")
    ax.fill_between(
        nsa_cmpsparse_fwd_seq, 1.0, fwd_cmpsparse_speedup, alpha=0.08, color="#9467bd"
    )
    ax.fill_between(
        fwdbwd_seq_lengths, 1.0, fwdbwd_speedup, alpha=0.08, color="#ff7f0e"
    )
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Speedup (Dense / NSA)", fontsize=12)
    ax.set_title("NSA Speedup vs Dense FA4", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks([4096, 16384, 65536, 262144, 1048576, 2097152])
    ax.set_xticklabels(["4K", "16K", "64K", "256K", "1M", "2M"], fontsize=10)

    # Annotate peak speedups
    # NSA+CmpSparse peaks at 1M (no 2M data)
    ax.annotate(
        f"{fwd_cmpsparse_speedup[-1]:.1f}x",
        xy=(1048576, fwd_cmpsparse_speedup[-1]),
        xytext=(200000, fwd_cmpsparse_speedup[-1] + 1),
        fontsize=11,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#9467bd"},
        color="#9467bd",
    )
    # NSA forward peaks at 2M
    ax.annotate(
        f"{fwd_speedup[-1]:.1f}x",
        xy=(2097152, fwd_speedup[-1]),
        xytext=(500000, fwd_speedup[-1] + 1),
        fontsize=11,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        color="#1f77b4",
    )
    # Fwd+Bwd peaks at 1M
    ax.annotate(
        f"{fwdbwd_speedup[-1]:.1f}x",
        xy=(1048576, fwdbwd_speedup[-1]),
        xytext=(200000, fwdbwd_speedup[-1] - 4),
        fontsize=11,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#ff7f0e"},
        color="#ff7f0e",
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
