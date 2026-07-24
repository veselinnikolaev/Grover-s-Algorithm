"""
tn_bond_dimension_experiment.py
================================
Bond dimension (chi) scaling experiment using quimb's approximate MPS
circuit simulator.

IMPORTANT CONCEPTUAL NOTE — read before running:
The exact contraction-tree approach in tn_experiments.py / tn_metrics.py
(via quimb_circuit + amplitude_rehearse) is EXACT: every qubit index has
dimension 2 and nothing is truncated, so there is no meaningful "bond
dimension" to sweep there beyond W itself. W already IS the memory-cost
number for that method.

"Bond dimension" (chi) as the abstract means it — a cutoff quantifying
how much entanglement is kept — only exists in APPROXIMATE methods like
Matrix Product States (MPS) with truncation. This script uses quimb's
MPS circuit simulator to actually cap chi at various values and measure
both the resulting cost AND the approximation error, since that
trade-off (cost saved vs accuracy lost) is exactly the abstract's third
research question: "at what threshold does entanglement growth
eventually render tensor network contraction intractable?"

This is a genuinely different simulation method from the exact
contraction sweep, not an extension of it — expect to report both as
separate, complementary results.

NOTE ON API STABILITY: quimb's MPS-circuit interface has changed across
versions and I have not been able to verify the exact class/method names
against your installed version. Before trusting this script, run:

    import quimb.tensor as qtn
    print([a for a in dir(qtn) if 'Circuit' in a])
    # then, once you find the right class name, e.g. qtn.CircuitMPS:
    print([a for a in dir(qtn.CircuitMPS) if not a.startswith('_')])

and adjust the class name / method calls below (currently CircuitMPS,
.apply_gates(), .amplitude()) to match what's actually available.

Requires: quimb (already installed)
"""

import os
import warnings

import numpy as np
import pandas as pd
import quimb.tensor as qtn
from qiskit import transpile
from qiskit.transpiler.passes import RemoveBarriers

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)  # cosmetic only —
# extreme low-chi truncation (esp. chi=2) can drive the state's norm to
# numerical zero, and quimb's internal phase-normalization then divides
# by ~0, producing overflow/invalid-value warnings and a NaN amplitude.
# The NaN itself is handled below (treated as non-convergence, same as
# any other unconverged chi), not silently ignored.
os.makedirs("results", exist_ok=True)

from grover_core import build_grover_circuit
from qiskit_quimb import quimb_circuit


def _prepare_for_quimb(qc):
    qc_no_barriers = RemoveBarriers()(qc)
    return transpile(qc_no_barriers, basis_gates=["rz", "sx", "x", "cx"],
                      optimization_level=1)


def experiment_bond_dimension_scaling(n=7, target=0,
                                       max_bond_values=(2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 48, 64),
                                       verbose=True):
    """
    For a fixed n and target, runs the same Grover circuit through
    quimb's MPS simulator at increasing max_bond (chi) caps. Always also
    runs the uncapped MPS simulation as ground truth to compare fidelity
    against.

    n defaults to 7, not 12: MPS simulators are efficient mainly for
    nearest-neighbor gates, and the mcx-decomposition used to prepare
    this circuit produces gates between non-adjacent qubits, which force
    internal SWAP networks. At n=12 this made the sweep hang/stall around
    max_bond=8 in practice. Keep n small here — this experiment's point
    is the chi-vs-error trade-off, not reaching a large qubit count.

    max_bond_values is a finer grid than a first pass (2,4,8,16,32,64)
    would suggest — that coarse grid can only ever report thresholds at
    powers of 2, so a real threshold of, say, chi=20 gets rounded up to
    "32" with no way to tell the difference. This grid adds intermediate
    points (6,10,12,20,24,28,48) specifically so the reported threshold
    reflects where convergence actually happens, not just the nearest
    power of 2 above it.
    """
    print("\n" + "=" * 60)
    print(f"  BOND DIMENSION EXPERIMENT: n={n}, target={target}, sweeping max_bond")
    print("=" * 60)

    target_bitstring = format(target, f"0{n}b")
    qc = build_grover_circuit(n, target)
    qc.remove_final_measurements(inplace=True)
    qc = _prepare_for_quimb(qc)
    gates = quimb_circuit(qc).gates  # NOTE: verify this attribute exists on your quimb version

    records = []

    # Compute the exact (uncapped) amplitude FIRST, regardless of where
    # None sits in max_bond_values — every other chi's error is measured
    # against this. (Previous version computed it last, so every row
    # before it silently reported err=None.)
    print(f"max_bond=uncapped (reference): ", end="", flush=True)
    try:
        circ_exact = qtn.CircuitMPS(n)
        circ_exact.apply_gates(gates)
        exact_amp = circ_exact.amplitude(target_bitstring)
        print(f"|amp|={abs(exact_amp):.6f}")
    except Exception as e:
        print(f"Error — {e}")
        exact_amp = None

    for chi in max_bond_values:
        if chi is None:
            continue  # already computed above as the reference
        print(f"max_bond={str(chi):>5}: ", end="", flush=True)
        try:
            circ_mps = qtn.CircuitMPS(n, max_bond=chi)
            circ_mps.apply_gates(gates)
            amp = circ_mps.amplitude(target_bitstring)

            if np.isnan(amp).any() if hasattr(amp, "__len__") else (amp != amp):
                # NaN from norm underflow at extreme low chi — treat as a
                # failed/non-converged point, not a crash. Still recorded
                # (as a large sentinel error) so it shows up honestly in
                # the CSV rather than silently vanishing.
                print(f"amp=NaN (norm underflow at this chi) — treating as non-converged")
                records.append({
                    "n_qubits": n, "max_bond": chi,
                    "amplitude_mag": None, "error_vs_uncapped": float("inf"),
                })
                continue

            error = None if exact_amp is None else abs(abs(amp) - abs(exact_amp))
            stats = {
                "n_qubits": n,
                "max_bond": chi,
                "amplitude_mag": abs(amp),
                "error_vs_uncapped": error,
            }
            records.append(stats)
            print(f"|amp|={abs(amp):.6f}  err={error}")
        except Exception as e:
            print(f"Error — {e}")
            continue

    if exact_amp is not None:
        records.append({
            "n_qubits": n,
            "max_bond": "uncapped",
            "amplitude_mag": abs(exact_amp),
            "error_vs_uncapped": 0.0,
        })

    df = pd.DataFrame(records)
    out_path = f"results/tn_bond_dimension_scaling_n{n}_t{target}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}  ({len(df)} rows)")
    return df


def find_convergence_threshold(df, tol=1e-9):
    """
    Given one experiment_bond_dimension_scaling() result, returns the
    smallest max_bond at which error_vs_uncapped is effectively zero
    (below tol) AND STAYS below tol for every larger max_bond tested —
    i.e. the minimum chi this circuit actually needs, once truncation
    has genuinely stabilized rather than just dipped low by coincidence.

    Non-monotonic error vs chi is real and expected here: each chi value
    is simulated independently, not as a refinement of the previous run,
    so different singular vectors can get kept at each truncation step.
    A naive "first dip below tol" check could misreport a lucky low
    point as convergence even though a larger chi later spikes back up
    (this didn't happen in practice yet, but the risk is real given how
    much the n=10/n=11 error curves wobble before settling).

    Returns None if nothing in the sweep converges and stays converged.
    """
    numeric = df[df["max_bond"] != "uncapped"].copy()
    numeric["max_bond"] = numeric["max_bond"].astype(int)
    numeric = numeric.sort_values("max_bond").reset_index(drop=True)

    for i in range(len(numeric)):
        if (numeric["error_vs_uncapped"].iloc[i:] <= tol).all():
            return int(numeric["max_bond"].iloc[i])
    return None


def experiment_threshold_vs_n(n_range=(5, 6, 7, 8, 9, 10, 11),
                               max_bond_values=(2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 48, 64),
                               n_targets_per_n=3, verbose=False, seed=0):
    """
    Runs experiment_bond_dimension_scaling() at several n and records the
    minimum chi needed for exact convergence at each — this is the actual
    growth curve for research question 3 ("at what threshold does
    entanglement growth render TN contraction intractable?").

    n_targets_per_n: tests several distinct targets per n (not just
    target=0) and reports the WORST-CASE (max) threshold across them,
    since a single target's entanglement structure isn't necessarily
    representative — different marked states can need different chi.
    Reports the full per-target detail alongside the worst-case summary
    so you can see how much variation there actually is.

    Keep n_range modest (roughly <=13) given MPS's difficulty with the
    non-adjacent gates from the mcx decomposition.
    """
    print("\n" + "=" * 60)
    print("  THRESHOLD SWEEP: min chi needed for exact convergence vs n")
    print(f"  ({n_targets_per_n} targets per n, worst-case threshold reported)")
    print("=" * 60)

    rng = np.random.RandomState(seed)
    summary = []
    per_target_detail = []

    for n in n_range:
        targets = sorted(set(int(t) for t in rng.randint(0, 2 ** n, size=n_targets_per_n)))
        thresholds_this_n = []
        for target in targets:
            df_nt = experiment_bond_dimension_scaling(n=n, target=target,
                                                       max_bond_values=max_bond_values,
                                                       verbose=verbose)
            threshold = find_convergence_threshold(df_nt)
            thresholds_this_n.append(threshold)
            per_target_detail.append({"n_qubits": n, "target": target,
                                       "min_chi_for_convergence": threshold})
            print(f"  n={n}, target={target}: min chi = {threshold}")

        valid = [t for t in thresholds_this_n if t is not None]
        worst_case = max(valid) if valid else None
        summary.append({
            "n_qubits": n,
            "min_chi_for_convergence_worst_case": worst_case,
            "min_chi_for_convergence_best_case": min(valid) if valid else None,
            "n_targets_tested": len(targets),
        })
        print(f"  >> n={n}: worst-case min chi across {len(targets)} targets = {worst_case}")

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("results/tn_bond_dimension_threshold.csv", index=False)
    print(f"\n  Saved: results/tn_bond_dimension_threshold.csv  ({len(df_summary)} rows)")

    pd.DataFrame(per_target_detail).to_csv("results/tn_bond_dimension_threshold_by_target.csv",
                                            index=False)
    print(f"  Saved: results/tn_bond_dimension_threshold_by_target.csv "
          f"({len(per_target_detail)} rows)")

    return df_summary


if __name__ == "__main__":
    experiment_threshold_vs_n(n_range=(5, 6, 7, 8, 9, 10, 11), n_targets_per_n=3)