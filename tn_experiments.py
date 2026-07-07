"""
tn_experiments.py
==================
Tensor network scalability sweep for Grover's algorithm, built on top of
grover_core.py's circuit constructor. Companion to experiments.py's
experiment_scalability(), but reporting quimb's contraction width (W) and
cost (C) instead of shot-based success probability — these are the
quantities directly comparable to BSP2's measured memory/time.

Run:
    python tn_experiments.py
"""

import numpy as np
import pandas as pd
import os
import warnings

from qiskit import transpile
from qiskit.transpiler.passes import RemoveBarriers

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.makedirs("results", exist_ok=True)

from grover_core import build_grover_circuit
from tn_metrics import rehearse_amplitude, _make_optimizer


def _prepare_for_quimb(qc):
    """
    quimb/qiskit-quimb only understands elementary gates. Grover's oracle
    and diffuser use mcx (arbitrary-arity multi-controlled X), which Aer
    handles natively but qiskit-quimb's converter doesn't recognize.
    Transpiling to a standard universal basis decomposes mcx recursively
    into single/two-qubit gates quimb can ingest. Barriers are also
    stripped since they carry no contraction meaning.

    Note: this decomposition is itself meaningful, not just a workaround —
    it's the same gate-count blowup a real TN/hardware backend would face
    for mcx, so it's fair to include it in the cost measurement rather
    than treat it as noise.
    """
    qc_no_barriers = RemoveBarriers()(qc)
    return transpile(qc_no_barriers, basis_gates=["rz", "sx", "x", "cx"],
                      optimization_level=1)


def experiment_tn_scalability(qubit_range=range(2, 34), verbose=True):
    print("\n" + "="*60)
    print("  TN EXPERIMENT: Contraction width (W) & cost (C) vs qubits")
    print("="*60)

    optimizer = _make_optimizer()  # built once, reused across the whole sweep

    records = []
    for n in qubit_range:
        target = np.random.randint(0, 2**n)
        qc = build_grover_circuit(n, target)
        qc.remove_final_measurements(inplace=True)
        qc = _prepare_for_quimb(qc)

        target_bitstring = format(target, f"0{n}b")

        print(f"n={n:2d}: ", end="", flush=True)
        try:
            stats = rehearse_amplitude(qc, target_bitstring, optimize=optimizer, verbose=verbose)
            stats.update({"n_qubits": n, "N": 2**n, "target": target})
            records.append(stats)
        except Exception as e:
            print(f"Error — {e}")
            break

    df = pd.DataFrame(records)
    df.to_csv("results/tn_scalability.csv", index=False)
    print(f"\n  Saved: results/tn_scalability.csv  ({len(df)} rows)")
    return df

if __name__ == "__main__":
    experiment_tn_scalability(qubit_range=range(2, 34))