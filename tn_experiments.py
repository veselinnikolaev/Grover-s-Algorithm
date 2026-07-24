"""
tn_experiments.py
==================
Tensor network scalability sweep for Grover's algorithm, built on top of
grover_core.py's circuit constructor. Companion to experiments.py's
experiment_scalability(), but reporting quimb's contraction width (W) and
cost (C) instead of shot-based success probability — these are the
quantities directly comparable to BSP2's measured memory/time.

Two experiments live here:

  experiment_tn_scalability(): the main rehearsed W/C sweep across a wide
  qubit range (2..33), now also computing the statevector-vs-TN memory
  CROSSOVER directly — this answers the abstract's research question 1
  ("at what qubit count do tensor networks become more memory-efficient
  than the statevector approach?").

  experiment_real_memory_validation(): a smaller, tractable range where
  the contraction is actually RUN (not just rehearsed) and real memory is
  measured, to validate that rehearsed W is a trustworthy predictor
  before trusting it out to n=33.

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
from tn_metrics import rehearse_amplitude, measure_real_contraction, _make_optimizer


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


def _statevector_bytes(n):
    """16 bytes per complex128 amplitude x 2^n amplitudes — matches the
    abstract's own stated definition of the statevector memory wall."""
    return 16 * (2 ** n)


def _tn_bytes_estimate(W):
    """W is log2 of the largest intermediate tensor (in elements); convert
    to bytes the same way as the statevector figure for a fair, directly
    comparable number."""
    return 16 * (2 ** W)


def experiment_tn_scalability(qubit_range=range(2, 14), verbose=True, use_greedy=False):
    """
    use_greedy defaults to False now: greedy-only (max_repeats=1) was
    tested and produced unstable, sometimes-nonsensical trees (e.g.
    W=45 at n=7, W=325 at n=11 — both far exceeding even the full
    statevector size for that n), while barely improving wall time.
    The bounded multi-trial search (_make_optimizer) gives consistently
    sane W/C values and is kept as the default despite being slower.

    qubit_range now defaults to 2..13, not 2..33: empirically, wall
    time grows roughly ~4x per qubit past n~8 regardless of optimizer
    choice, pointing to a structural bottleneck (gate-count blowup from
    the mcx decomposition on this highly-entangling circuit), not a
    tunable search parameter. Treat the reached ceiling itself as a
    reportable finding — "exact TN contraction became impractical
    beyond n~X" is a legitimate answer to research question 1, not a
    failure to fix further.
    """
    print("\n" + "=" * 60)
    print("  TN EXPERIMENT: Contraction width (W) & cost (C) vs qubits")
    print(f"  optimizer: {'greedy (fast, single-shot)' if use_greedy else 'multi-trial search (bounded, sane W/C)'}")
    print("=" * 60)

    records = []
    errors = []
    for n in qubit_range:
        target = np.random.randint(0, 2 ** n)
        qc = build_grover_circuit(n, target)
        qc.remove_final_measurements(inplace=True)
        qc = _prepare_for_quimb(qc)

        target_bitstring = format(target, f"0{n}b")
        optimizer = _make_greedy_optimizer() if use_greedy else _make_optimizer()

        print(f"n={n:2d}: ", end="", flush=True)
        try:
            stats = rehearse_amplitude(qc, target_bitstring, optimize=optimizer, verbose=verbose)
            sv_bytes = _statevector_bytes(n)
            tn_bytes = _tn_bytes_estimate(stats["contraction_width_W"])
            stats.update({
                "n_qubits": n,
                "N": 2 ** n,
                "target": target,
                "circuit_depth": qc.depth(),
                "gate_count": qc.size(),
                "statevector_bytes": sv_bytes,
                "tn_bytes_estimate": tn_bytes,
                "tn_advantage": tn_bytes < sv_bytes,
            })
            records.append(stats)

            # save after every n, not just at the end — a long sweep to
            # n=33 can take a while per point, and losing everything to
            # one Ctrl+C or crash isn't worth risking
            pd.DataFrame(records).to_csv("results/tn_scalability.csv", index=False)
        except Exception as e:
            print(f"Error — {e}")
            errors.append({"n_qubits": n, "error": str(e)})
            pd.DataFrame(errors).to_csv("results/tn_scalability_errors.csv", index=False)
            # continue rather than break: one bad n shouldn't silently
            # discard every larger n that might otherwise have worked
            continue

    df = pd.DataFrame(records)
    print(f"\n  Saved: results/tn_scalability.csv  ({len(df)} rows)")

    if errors:
        print(f"  {len(errors)} qubit count(s) failed — see results/tn_scalability_errors.csv")

    if not df.empty and df["tn_advantage"].any():
        crossover_n = int(df.loc[df["tn_advantage"], "n_qubits"].min())
        print(f"  >> Estimated crossover: TN becomes memory-favorable at n={crossover_n}")
    elif not df.empty:
        print("  >> No crossover found in this range — TN did not beat statevector "
              "memory here (a legitimate, reportable result given Grover's "
              "oracle is a highly entangling circuit).")

    return df


def experiment_real_memory_validation(qubit_range=range(2, 16), verbose=True):
    """
    Validates rehearsed W against ACTUAL measured memory for a small,
    tractable range of n. This is the ground-truth check the abstract's
    'as the algorithm runs' language calls for — rehearsal alone
    estimates cost without ever running the contraction, so it needs to
    be checked against real numbers at least once before you trust it
    for n up to 33.

    Keep qubit_range small — this really executes each contraction.
    """
    print("\n" + "=" * 60)
    print("  REAL CONTRACTION: measured memory vs rehearsed W")
    print("=" * 60)

    records = []
    for n in qubit_range:
        target = np.random.randint(0, 2 ** n)
        qc = build_grover_circuit(n, target)
        qc.remove_final_measurements(inplace=True)
        qc = _prepare_for_quimb(qc)
        target_bitstring = format(target, f"0{n}b")

        print(f"n={n:2d}: ", end="", flush=True)
        try:
            rehearsed = rehearse_amplitude(qc, target_bitstring,
                                            optimize=_make_optimizer(), verbose=False)
            real = measure_real_contraction(qc, target_bitstring,
                                             optimize=_make_optimizer(), verbose=verbose)
            real.update({
                "n_qubits": n,
                "rehearsed_W": rehearsed["contraction_width_W"],
                "rehearsed_tn_bytes_estimate": _tn_bytes_estimate(rehearsed["contraction_width_W"]),
            })
            records.append(real)
        except Exception as e:
            print(f"Error — {e}")
            continue

    df = pd.DataFrame(records)
    df.to_csv("results/tn_real_memory_validation.csv", index=False)
    print(f"\n  Saved: results/tn_real_memory_validation.csv  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    experiment_tn_scalability(qubit_range=range(2, 14))
    experiment_real_memory_validation(qubit_range=range(2, 14))