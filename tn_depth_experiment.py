"""
tn_depth_experiment.py
=======================
Fixes n and varies Grover iteration count (hence circuit depth) to
isolate depth as its own scaling axis, separate from qubit count — one
of the three research questions in the project abstract ("how does
simulation cost scale against circuit depth and bond dimension?").

Uses build_grover_circuit(n, target, n_iterations=...) — grover_core.py
already exposes this parameter (defaults to the optimal count when None),
so no changes to grover_core.py are needed.

Run:
    python tn_depth_experiment.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.transpiler.passes import RemoveBarriers

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.makedirs("results", exist_ok=True)

from grover_core import build_grover_circuit
from tn_metrics import rehearse_amplitude, _make_optimizer


def _prepare_for_quimb(qc):
    qc_no_barriers = RemoveBarriers()(qc)
    return transpile(qc_no_barriers, basis_gates=["rz", "sx", "x", "cx"],
                      optimization_level=1)


def experiment_depth_scaling(n=10, iteration_range=range(1, 8), verbose=True):
    print("\n" + "=" * 60)
    print(f"  DEPTH EXPERIMENT: fixed n={n}, varying Grover iterations")
    print("=" * 60)

    records = []
    target = np.random.randint(0, 2 ** n)
    target_bitstring = format(target, f"0{n}b")

    for k in iteration_range:
        qc = build_grover_circuit(n, target, n_iterations=k)
        qc.remove_final_measurements(inplace=True)
        qc = _prepare_for_quimb(qc)

        optimizer = _make_optimizer()
        print(f"iterations={k:2d} (depth={qc.depth()}): ", end="", flush=True)
        try:
            stats = rehearse_amplitude(qc, target_bitstring, optimize=optimizer, verbose=verbose)
            stats.update({
                "iterations": k,
                "n_qubits": n,
                "circuit_depth": qc.depth(),
                "gate_count": qc.size(),
            })
            records.append(stats)
        except Exception as e:
            print(f"Error — {e}")
            continue

    df = pd.DataFrame(records)
    df.to_csv("results/tn_depth_scaling.csv", index=False)
    print(f"\n  Saved: results/tn_depth_scaling.csv  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    experiment_depth_scaling(n=10, iteration_range=range(1, 8))