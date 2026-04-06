"""
plot_results.py
===============
Generate publication-quality figures from the experiment CSVs.
Run after experiments.py:
    python plot_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import pi, sqrt, floor, log2
import os, warnings

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "quantum": "#2563EB",
    "classical": "#DC2626",
    "theory": "#16A34A",
    "memory": "#9333EA",
    "accent": "#F59E0B",
}


# ---------------------------------------------------------------------------
# Figure 1: Scalability — time and memory
# ---------------------------------------------------------------------------

def plot_scalability():
    try:
        df = pd.read_csv("results/scalability.csv")
    except FileNotFoundError:
        print("scalability.csv not found — skipping"); return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Figure 1 — Classical Simulation Scalability", fontweight="bold")

    n = df["n_qubits"].values

    # 1a: Simulation time
    ax = axes[0]
    ax.semilogy(n, df["sim_time_s"], "o-", color=COLORS["quantum"], lw=2, ms=5, label="Measured")
    # Fit exponential guide
    if len(n) > 3:
        coeffs = np.polyfit(n, np.log(df["sim_time_s"] + 1e-9), 1)
        ax.semilogy(n, np.exp(np.polyval(coeffs, n)), "--", color=COLORS["theory"],
                    alpha=0.7, label=f"Fit: e^({coeffs[0]:.2f}·n)")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Simulation time (s)")
    ax.set_title("Simulation Time")
    ax.legend(fontsize=9)

    # 1b: Memory
    ax = axes[1]
    theory_mem = [(2**ni * 16) / (1024**2) for ni in n]
    ax.semilogy(n, df["mem_delta_mb"], "o-", color=COLORS["memory"],
                lw=2, ms=5, label="Measured (RSS delta)")
    ax.semilogy(n, theory_mem, "--", color=COLORS["theory"],
                lw=1.5, alpha=0.8, label="Theory: 16·2ⁿ bytes")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Consumption")
    ax.legend(fontsize=9)

    # Add RAM thresholds
    for label, gb in [("8 GB RAM", 8192), ("16 GB", 16384), ("64 GB", 65536)]:
        ax.axhline(gb, color="gray", lw=0.8, ls=":")
        ax.text(n[0], gb * 1.1, label, color="gray", fontsize=8)

    # 1c: Circuit metrics
    ax = axes[2]
    ax.plot(n, df["circuit_depth"], "s-", color=COLORS["classical"], lw=2, ms=5, label="Depth")
    ax.plot(n, df["circuit_gate_count"], "^-", color=COLORS["accent"], lw=2, ms=5, label="Gate count")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Count")
    ax.set_title("Circuit Complexity")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig("figures/fig1_scalability.pdf", bbox_inches="tight")
    fig.savefig("figures/fig1_scalability.png", bbox_inches="tight")
    print("  Saved: figures/fig1_scalability.{pdf,png}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Iteration sweep
# ---------------------------------------------------------------------------

def plot_iteration_sweep():
    try:
        df = pd.read_csv("results/iteration_sweep.csv")
    except FileNotFoundError:
        print("iteration_sweep.csv not found — skipping"); return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("Figure 2 — Success Probability vs Grover Iterations", fontweight="bold")

    k = df["k"].values
    ax.plot(k, df["success_probability"], "o", color=COLORS["quantum"],
            ms=6, label="Simulation (sampled)", zorder=5)
    ax.plot(k, df["p_theory"], "-", color=COLORS["theory"],
            lw=2, alpha=0.8, label=r"Theory: $\sin^2((2k+1)\theta)$")

    # Mark optimal k
    n_qubits = int(df["n_qubits"].iloc[0])
    k_opt = floor((pi / 4) * sqrt(2**n_qubits))
    ax.axvline(k_opt, color=COLORS["accent"], ls="--", lw=1.5,
               label=f"k_opt = {k_opt}")
    ax.annotate(f"k_opt = {k_opt}\n(peak)", xy=(k_opt, 1.0),
                xytext=(k_opt + 2, 0.85),
                arrowprops=dict(arrowstyle="->", color="gray"),
                color="gray", fontsize=9)

    ax.set_xlabel("Number of Grover iterations  k")
    ax.set_ylabel("P(success)")
    ax.set_title(f"n = {n_qubits} qubits,  N = {2**n_qubits} items")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    note = (f"Key insight: over-rotating past k_opt reduces P(success).\n"
            f"The sinusoidal pattern reflects state-vector rotation by 2θ per iteration.")
    ax.text(0.02, 0.05, note, transform=ax.transAxes,
            fontsize=8.5, color="gray", va="bottom")

    plt.tight_layout()
    fig.savefig("figures/fig2_iteration_sweep.pdf", bbox_inches="tight")
    fig.savefig("figures/fig2_iteration_sweep.png", bbox_inches="tight")
    print("  Saved: figures/fig2_iteration_sweep.{pdf,png}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Classical vs quantum query complexity
# ---------------------------------------------------------------------------

def plot_classical_comparison():
    try:
        df = pd.read_csv("results/classical_comparison.csv")
    except FileNotFoundError:
        print("classical_comparison.csv not found — skipping"); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Figure 3 — Grover vs Classical Search: Query Complexity", fontweight="bold")

    n = df["n_qubits"].values
    N = df["N"].values

    # 3a: Query count comparison
    ax = axes[0]
    ax.semilogy(n, df["classical_queries_expected"], "s-",
                color=COLORS["classical"], lw=2, ms=5, label="Classical O(N/2)")
    ax.semilogy(n, df["grover_queries"], "o-",
                color=COLORS["quantum"], lw=2, ms=5, label="Grover O(√N)")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Number of oracle queries")
    ax.set_title("Oracle Query Count")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # 3b: Speedup factor
    ax = axes[1]
    ax.plot(n, df["speedup_factor"], "o-", color=COLORS["accent"], lw=2, ms=5)
    # Theoretical: speedup = √N/2
    ax.plot(n, np.sqrt(N) / 2, "--", color="gray", lw=1.5, alpha=0.8, label="Theory: √N/2")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Speedup factor (classical / Grover queries)")
    ax.set_title("Quadratic Speedup Factor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    note = "Note: Grover gives a quadratic speedup, not exponential.\nFor n=30 (N≈10⁹), classical≈500M queries, Grover≈25K queries."
    axes[1].text(0.02, 0.05, note, transform=axes[1].transAxes,
                 fontsize=8.5, color="gray", va="bottom")

    plt.tight_layout()
    fig.savefig("figures/fig3_speedup.pdf", bbox_inches="tight")
    fig.savefig("figures/fig3_speedup.png", bbox_inches="tight")
    print("  Saved: figures/fig3_speedup.{pdf,png}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Circuit depth
# ---------------------------------------------------------------------------

def plot_circuit_depth():
    try:
        df = pd.read_csv("results/circuit_depth.csv")
    except FileNotFoundError:
        print("circuit_depth.csv not found — skipping"); return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Figure 4 — Circuit Depth & Gate Count", fontweight="bold")

    n = df["n_qubits"].values

    ax = axes[0]
    ax.plot(n, df["circuit_depth"], "o-", color=COLORS["quantum"], lw=2, ms=5)
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Circuit depth")
    ax.set_title("Circuit Depth at k_opt Iterations")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(n, df["gate_count"], "s-", color=COLORS["classical"], lw=2, ms=5)
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Total gate count")
    ax.set_title("Total Gate Count (log scale)")
    ax.grid(True, alpha=0.3, which="both")

    note = "On real hardware: circuit depth ∝ accumulated noise.\nThis is why quantum advantage for search is not yet demonstrated at scale."
    axes[1].text(0.02, 0.05, note, transform=axes[1].transAxes,
                 fontsize=8.5, color="gray", va="bottom")

    plt.tight_layout()
    fig.savefig("figures/fig4_circuit_depth.pdf", bbox_inches="tight")
    fig.savefig("figures/fig4_circuit_depth.png", bbox_inches="tight")
    print("  Saved: figures/fig4_circuit_depth.{pdf,png}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: GPU comparison (if data exists)
# ---------------------------------------------------------------------------

def plot_gpu_comparison():
    try:
        df = pd.read_csv("results/gpu_comparison.csv")
    except FileNotFoundError:
        print("gpu_comparison.csv not found — skipping"); return

    if df["gpu_time_s"].isna().all():
        print("  No GPU data available — skipping GPU figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Figure 5 — CPU vs GPU Simulation Time", fontweight="bold")

    n = df["n_qubits"].values
    ax = axes[0]
    ax.semilogy(n, df["cpu_time_s"], "o-", color=COLORS["classical"], lw=2, ms=5, label="CPU")
    if not df["gpu_time_s"].isna().all():
        ax.semilogy(n, df["gpu_time_s"].dropna(), "s-",
                    color=COLORS["quantum"], lw=2, ms=5, label="GPU")
    ax.set_xlabel("Number of qubits  n"); ax.set_ylabel("Simulation time (s)")
    ax.set_title("CPU vs GPU Time"); ax.legend(); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    if not df["speedup"].isna().all():
        ax.plot(df["n_qubits"], df["speedup"], "o-", color=COLORS["accent"], lw=2, ms=5)
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_xlabel("Number of qubits  n"); ax.set_ylabel("GPU speedup (CPU/GPU time)")
        ax.set_title("GPU Acceleration Factor"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("figures/fig5_gpu.pdf", bbox_inches="tight")
    fig.savefig("figures/fig5_gpu.png", bbox_inches="tight")
    print("  Saved: figures/fig5_gpu.{pdf,png}")
    plt.close()


if __name__ == "__main__":
    print("\nGenerating figures from results/...\n")
    plot_scalability()
    plot_iteration_sweep()
    plot_classical_comparison()
    plot_circuit_depth()
    plot_gpu_comparison()
    print("\nAll figures saved in ./figures/")
