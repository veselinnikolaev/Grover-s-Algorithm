"""
plot_results.py
===============
Generate publication-quality figures from the experiment CSVs.
Run after experiments.py (and optionally after hpc experiments, and the
tn_experiments.py tensor-network sweep):
    python plot_results.py

Figures produced
----------------
  fig1_scalability        — simulation time, memory, circuit complexity
  fig2_iteration_sweep    — success probability vs Grover iterations
  fig3_speedup            — Grover vs classical query complexity
  fig4_circuit_depth      — circuit depth & gate count
  fig5_gpu                — CPU vs GPU comparison (gpu_comparison.csv)
  fig6_hpc                — HPC: 3-panel combined dashboard (GPU vs CPU)
  fig7_tn_crossover       — Statevector vs TN memory crossover (results/tn_scalability.csv)
  fig8_bond_threshold     — Min bond dimension (χ) for exact convergence vs n
  fig9_bond_convergence   — Error vs χ convergence curve, for the last n in the sweep
  fig8_bond_threshold     — Min chi needed for exact convergence vs n (results/tn_bond_dimension_threshold.csv)
  fig9_bond_convergence   — Amplitude/error vs chi for one n (results/tn_bond_dimension_scaling_n{5-11}_t{target}.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from math import pi, sqrt, floor, log2
import os, warnings

warnings.filterwarnings("ignore")
os.makedirs("figures/png", exist_ok=True)
os.makedirs("figures/pdf", exist_ok=True)

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
    "quantum":  "#2563EB",
    "classical":"#DC2626",
    "theory":   "#16A34A",
    "memory":   "#9333EA",
    "accent":   "#F59E0B",
    "gpu":      "#2563EB",
    "cpu":      "#DC2626",
    "speedup":  "#F59E0B",
}

_HPC_COLS = ["n_qubits", "N", "sim_time_s", "mem_mb_theoretical", "device"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _exp_fit(n, t):
    """Fit t = exp(a + b*n) via OLS on log(t).  Returns (a, b)."""
    coeffs = np.polyfit(n, np.log(np.array(t, dtype=float) + 1e-12), 1)
    return coeffs[1], coeffs[0]


def _fmt_time(s):
    """Human-readable time label for log-scale y-axes."""
    if s >= 3600:
        return f"{s/3600:.1f} h"
    if s >= 60:
        return f"{s/60:.1f} min"
    return f"{s:.2f} s"


def _save(fig, name):
    """Save figure to both figures/pdf/ and figures/png/ subfolders."""
    fig.savefig(f"figures/pdf/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/png/{name}.png", bbox_inches="tight")
    print(f"  Saved: figures/pdf/{name}.pdf  |  figures/png/{name}.png")


def _load_hpc_gpu():
    path = "results/hpc_scalability_gpu.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_hpc_cpu():
    path = "results/hpc_scalability_cpu.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, header=None, names=_HPC_COLS)


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

    ax = axes[0]
    ax.semilogy(n, df["sim_time_s"], "o-", color=COLORS["quantum"], lw=2, ms=5, label="Measured")
    if len(n) > 3:
        coeffs = np.polyfit(n, np.log(df["sim_time_s"] + 1e-9), 1)
        ax.semilogy(n, np.exp(np.polyval(coeffs, n)), "--", color=COLORS["theory"],
                    alpha=0.7, label=f"Fit: e^({coeffs[0]:.2f}·n)")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Simulation time (s)")
    ax.set_title("Simulation Time")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogy(n, df["mem_delta_mb"], "o-", color=COLORS["memory"],
                lw=2, ms=5, label="Measured (peak RSS delta)")
    ax.semilogy(n, df["statevector_size_mb"], "--", color=COLORS["theory"],
                lw=1.5, alpha=0.8, label="Theory: 16·2ⁿ bytes")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Consumption")
    ax.legend(fontsize=9)
    for label, gb in [("8 GB RAM", 8192), ("16 GB", 16384), ("64 GB", 65536)]:
        ax.axhline(gb, color="gray", lw=0.8, ls=":")
        ax.text(n[0], gb * 1.1, label, color="gray", fontsize=8)

    ax = axes[2]
    ax.plot(n, df["circuit_depth"], "s-", color=COLORS["classical"], lw=2, ms=5, label="Depth")
    ax.plot(n, df["circuit_gate_count"], "^-", color=COLORS["accent"], lw=2, ms=5, label="Gate count")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Count")
    ax.set_title("Circuit Complexity")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "fig1_scalability")
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

    n_qubits = int(df["n_qubits"].iloc[0])
    k_opt = floor((pi / 4) * sqrt(2**n_qubits))
    ax.axvline(k_opt, color=COLORS["accent"], ls="--", lw=1.5, label=f"k_opt = {k_opt}")
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
    ax.text(0.02, 0.05, note, transform=ax.transAxes, fontsize=8.5, color="gray", va="bottom")

    plt.tight_layout()
    _save(fig, "fig2_iteration_sweep")
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

    ax = axes[1]
    ax.plot(n, df["speedup_factor"], "o-", color=COLORS["accent"], lw=2, ms=5)
    ax.plot(n, np.sqrt(N) / 2, "--", color="gray", lw=1.5, alpha=0.8, label="Theory: √N/2")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Speedup factor (classical / Grover queries)")
    ax.set_title("Quadratic Speedup Factor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    note = "Note: Grover gives a quadratic speedup, not exponential.\nFor n=30 (N≈10⁹), classical≈500M queries, Grover≈25K queries."
    axes[1].text(0.02, 0.05, note, transform=axes[1].transAxes, fontsize=8.5, color="gray", va="bottom")

    plt.tight_layout()
    _save(fig, "fig3_speedup")
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
    axes[1].text(0.02, 0.05, note, transform=axes[1].transAxes, fontsize=8.5, color="gray", va="bottom")

    plt.tight_layout()
    _save(fig, "fig4_circuit_depth")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: GPU comparison (gpu_comparison.csv — original experiment)
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
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Simulation time (s)")
    ax.set_title("CPU vs GPU Time")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    if not df["speedup"].isna().all():
        ax.plot(df["n_qubits"], df["speedup"], "o-", color=COLORS["accent"], lw=2, ms=5)
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_xlabel("Number of qubits  n")
        ax.set_ylabel("GPU speedup (CPU/GPU time)")
        ax.set_title("GPU Acceleration Factor")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig5_gpu")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6: HPC — Combined 3-panel dashboard
# ---------------------------------------------------------------------------

def plot_hpc_dashboard(gpu, cpu):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Figure 6 — HPC Results: Grover's Algorithm", fontweight="bold")

    gn, gt = gpu["n_qubits"].values, gpu["sim_time_s"].values
    cn, ct = cpu["n_qubits"].values, cpu["sim_time_s"].values

    # Panel A: Simulation time
    ax = axes[0]
    ax.semilogy(gn, gt, "o-", color=COLORS["gpu"],  lw=2, ms=5, label="GPU")
    ax.semilogy(cn, ct, "s-", color=COLORS["cpu"],  lw=2, ms=5, label="CPU")
    ga, gb = _exp_fit(gn, gt)
    ca, cb = _exp_fit(cn, ct)
    ax.semilogy(np.linspace(gn[0], gn[-1], 200),
                np.exp(ga + gb * np.linspace(gn[0], gn[-1], 200)),
                "--", color=COLORS["gpu"], lw=1.2, alpha=0.6)
    ax.semilogy(np.linspace(cn[0], cn[-1], 200),
                np.exp(ca + cb * np.linspace(cn[0], cn[-1], 200)),
                "--", color=COLORS["cpu"], lw=1.2, alpha=0.6)
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Simulation time (s)")
    ax.set_title("Simulation Time")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_time(v)))
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)

    # Panel B: Memory
    ax = axes[1]
    ax.semilogy(gpu["n_qubits"].values, gpu["mem_mb_theoretical"].values, "o-",
                color=COLORS["memory"], lw=2, ms=5, label="16·2ⁿ bytes")
    for label, gb_val in [("8 GB", 8192), ("16 GB", 16384), ("32 GB", 32768)]:
        ax.axhline(gb_val, color="gray", lw=0.7, ls=":")
        ax.text(gn[0] + 0.1, gb_val * 1.12, label, color="gray", fontsize=7.5)
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Statevector Memory")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1024:.0f} GB" if v >= 1024 else f"{v:.0f} MB"
    ))
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)

    # Panel C: Speedup
    ax = axes[2]
    gpu_idx  = {row.n_qubits: row.sim_time_s for row in gpu.itertuples()}
    cpu_idx  = {row.n_qubits: row.sim_time_s for row in cpu.itertuples()}
    shared   = sorted(set(gpu_idx) & set(cpu_idx))
    speedups = [cpu_idx[n] / gpu_idx[n] for n in shared]
    ax.bar(shared, speedups, color=COLORS["speedup"], edgecolor="none", width=0.6, zorder=3)
    ax.axhline(1, color="gray", ls="--", lw=1, alpha=0.6)
    for x, sp in zip(shared, speedups):
        ax.text(x, sp + 0.15, f"{sp:.1f}×", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Speedup factor")
    ax.set_title("GPU Speedup over CPU")
    ax.set_xticks(shared)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_ylim(0, max(speedups) * 1.25)

    plt.tight_layout()
    _save(fig, "fig6_hpc")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 7: TN vs statevector memory crossover (from tn_experiments.py)
# ---------------------------------------------------------------------------

def plot_tn_crossover():
    try:
        df = pd.read_csv("results/tn_scalability.csv")
    except FileNotFoundError:
        print("tn_scalability.csv not found — skipping (run tn_experiments.py first)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Figure 7 — Statevector vs Tensor-Network Memory Scaling", fontweight="bold")

    n = df["n_qubits"].values
    ax.semilogy(n, df["statevector_bytes"], "o-", color=COLORS["classical"],
                lw=2, ms=5, label="Statevector (16·2ⁿ bytes)")
    ax.semilogy(n, df["tn_bytes_estimate"], "s-", color=COLORS["quantum"],
                lw=2, ms=5, label="TN estimate (16·2^W bytes)")
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Memory (bytes, log scale)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    if df["tn_advantage"].any():
        crossover_n = int(df.loc[df["tn_advantage"], "n_qubits"].min())
        ax.axvline(crossover_n, color=COLORS["accent"], ls="--", lw=1.5)
        ax.text(crossover_n, ax.get_ylim()[1], f" crossover n={crossover_n}",
                color=COLORS["accent"], va="top", fontsize=9)
        note = f"TN becomes memory-favorable at n={crossover_n}."
    else:
        note = "No crossover in this range — TN did not beat statevector memory here."
    ax.text(0.02, 0.05, note, transform=ax.transAxes, fontsize=8.5, color="gray", va="bottom")

    plt.tight_layout()
    _save(fig, "fig7_tn_crossover")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 8: bond dimension threshold vs n (from tn_bond_dimension_experiment.py)
# ---------------------------------------------------------------------------

def plot_bond_threshold():
    try:
        df = pd.read_csv("results/tn_bond_dimension_threshold.csv")
    except FileNotFoundError:
        print("tn_bond_dimension_threshold.csv not found — skipping "
              "(run tn_bond_dimension_experiment.py first)")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle("Figure 8 — Minimum Bond Dimension for Exact Convergence vs n", fontweight="bold")

    ax.semilogy(df["n_qubits"], df["min_chi_for_convergence_worst_case"], "o-",
                color=COLORS["memory"], lw=2, ms=7, label="Worst case (hardest target)")
    if "min_chi_for_convergence_best_case" in df.columns:
        ax.semilogy(df["n_qubits"], df["min_chi_for_convergence_best_case"], "s--",
                    color=COLORS["memory"], lw=1.5, ms=6, alpha=0.5,
                    label="Best case (easiest target)")
        ax.fill_between(df["n_qubits"], df["min_chi_for_convergence_best_case"],
                        df["min_chi_for_convergence_worst_case"],
                        color=COLORS["memory"], alpha=0.1)
    ax.set_xlabel("Number of qubits  n")
    ax.set_ylabel("Min χ for exact convergence (log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    n_targets = int(df["n_targets_tested"].iloc[0]) if "n_targets_tested" in df.columns else "?"
    note = (f"Growth of required bond dimension with n reflects entanglement\n"
            f"growth in Grover's oracle+diffuser structure. Each n tested against\n"
            f"{n_targets} distinct targets — shaded band shows target-dependent spread.")
    ax.text(0.02, 0.05, note, transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

    plt.tight_layout()
    _save(fig, "fig8_bond_threshold")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 9: convergence curve for a specific (n, target) run
# ---------------------------------------------------------------------------

def plot_bond_convergence():
    import glob

    candidates = sorted(glob.glob("results/tn_bond_dimension_scaling_n*_t*.csv"))
    if not candidates:
        print("No tn_bond_dimension_scaling_n*_t*.csv files found — skipping "
              "(run tn_bond_dimension_experiment.py first)")
        return

    # Pick the largest n available (most interesting: closest to the
    # entanglement threshold), and its first target.
    def _n_from_path(p):
        return int(p.split("_n")[1].split("_t")[0])

    path = max(candidates, key=_n_from_path)
    df = pd.read_csv(path)

    numeric = df[df["max_bond"] != "uncapped"].copy()
    numeric["max_bond"] = numeric["max_bond"].astype(int)
    numeric = numeric.sort_values("max_bond")
    n_qubits = int(numeric["n_qubits"].iloc[0]) if not numeric.empty else "?"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(f"Figure 9 — Convergence vs Bond Dimension (n={n_qubits})", fontweight="bold")

    # Honesty fix: don't fake extra precision with a silent +epsilon.
    # Points that converged EXACTLY (error == 0.0) are plotted at a
    # visually-obvious floor and marked with a different color/label,
    # rather than looking like a real 1e-12-level measurement.
    floor = 1e-16
    exact_zero = numeric["error_vs_uncapped"] == 0.0
    display_err = numeric["error_vs_uncapped"].where(~exact_zero, floor)

    ax.semilogy(numeric.loc[~exact_zero, "max_bond"], numeric.loc[~exact_zero, "error_vs_uncapped"],
                "o-", color=COLORS["classical"], lw=2, ms=7, label="Measured error")
    ax.semilogy(numeric.loc[exact_zero, "max_bond"], display_err[exact_zero],
                "o", color=COLORS["theory"], ms=9, label="Exact convergence (error = 0.0)")
    # connect the two segments with a plain line for visual continuity
    ax.semilogy(numeric["max_bond"], display_err, "-", color=COLORS["classical"], lw=1, alpha=0.4)

    ax.set_xlabel("Max bond dimension  χ")
    ax.set_ylabel("|error| vs uncapped amplitude (log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    note = (f"Green points converged EXACTLY (error printed as 0.0) — plotted at an\n"
            f"arbitrary floor purely so the log scale renders; this is NOT a precision\n"
            f"measurement. Data from: {os.path.basename(path)}")
    ax.text(0.02, 0.05, note, transform=ax.transAxes, fontsize=7.5, color="gray", va="bottom")

    plt.tight_layout()
    _save(fig, "fig9_bond_convergence")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nGenerating figures from results/...\n")

    plot_scalability()
    plot_iteration_sweep()
    plot_classical_comparison()
    plot_circuit_depth()
    plot_gpu_comparison()

    hpc_gpu, hpc_cpu = None, None
    try:
        hpc_gpu = _load_hpc_gpu()
        hpc_cpu = _load_hpc_cpu()
    except FileNotFoundError as e:
        print(f"  {e} — skipping HPC figures\n")

    if hpc_gpu is not None and hpc_cpu is not None:
        plot_hpc_dashboard(hpc_gpu, hpc_cpu)

    plot_tn_crossover()
    plot_bond_threshold()
    plot_bond_convergence()

    print("\nAll figures saved in ./figures/pdf/ and ./figures/png/")