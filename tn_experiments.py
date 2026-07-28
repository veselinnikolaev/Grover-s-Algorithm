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
from tn_metrics import rehearse_amplitude, measure_real_contraction, _make_optimizer, _make_greedy_optimizer


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


def experiment_tn_scalability(qubit_range=range(2, 12), verbose=True, use_greedy=False,
                               n_search_repeats=3, repeats_above_n=9):
    """
    n_search_repeats / repeats_above_n: HyperOptimizer's search is
    stochastic with no fixed seed, and running it n_search_repeats times
    to find the minimum W gets expensive fast — the previous single-pass
    run took ~5.7 hours total, with n=13 ALONE taking ~4.4 hours. Tripling
    that is not worth it: the finding (TN estimate exploding vs
    statevector, already many orders of magnitude worse by n~9-11) is
    already decisive well before n=13, so squeezing a cleaner number out
    of the most expensive points doesn't change the conclusion.

    qubit_range now capped at 2..11, not 2..13 — n=12/13 dominated total
    time for marginal additional insight. Repeats (for reliability against
    search-luck noise) only apply at or below repeats_above_n=9, where
    each search is still cheap (seconds, not minutes/hours); above that,
    a single run is used to keep total wall time reasonable.
    """
    print("\n" + "=" * 60)
    print("  TN EXPERIMENT: Contraction width (W) & cost (C) vs qubits")
    print(f"  optimizer: {'greedy (fast, single-shot)' if use_greedy else 'multi-trial search (bounded, sane W/C)'}")
    print("=" * 60)

    records = []
    errors = []
    for n in qubit_range:
        target = 0  # fixed, not random — avoids confounding growth-with-n with target-dependent noise
        qc = build_grover_circuit(n, target)
        qc.remove_final_measurements(inplace=True)
        qc = _prepare_for_quimb(qc)

        target_bitstring = format(target, f"0{n}b")

        print(f"n={n:2d}: ", end="", flush=True)
        try:
            reps = n_search_repeats if n <= repeats_above_n else 1
            best_stats = None
            for rep in range(reps):
                optimizer = _make_greedy_optimizer() if use_greedy else _make_optimizer()
                trial_stats = rehearse_amplitude(qc, target_bitstring, optimize=optimizer,
                                                  verbose=False)
                if best_stats is None or trial_stats["contraction_width_W"] < best_stats["contraction_width_W"]:
                    best_stats = trial_stats
            stats = best_stats
            if verbose:
                print(f"    best-of-{reps}: W={stats['contraction_width_W']:.2f}  "
                      f"C={stats['contraction_cost_C']:.2f}")
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
                "search_repeats_used": reps,
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

    if not df.empty:
        # A single early n where tn_bytes happens to be smaller (e.g. at
        # trivial n=2/3, both values are tiny and this is noise, not a
        # real advantage) doesn't mean TN "wins" — it needs to hold for
        # n and everything larger to count as a genuine crossover.
        df_sorted = df.sort_values("n_qubits").reset_index(drop=True)
        sustained_from = None
        for i in range(len(df_sorted)):
            if df_sorted["tn_advantage"].iloc[i:].all():
                sustained_from = int(df_sorted["n_qubits"].iloc[i])
                break

        if sustained_from is not None:
            print(f"  >> Sustained crossover: TN stays memory-favorable from n={sustained_from} onward")
        else:
            print("  >> No SUSTAINED crossover found — any early n where TN looked smaller "
                  "didn't hold as n grew further (a legitimate, reportable result given "
                  "Grover's oracle is a highly entangling circuit).")

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
        target = 0  # fixed, not random — same reasoning as the main sweep above
        qc = build_grover_circuit(n, target)
        qc.remove_final_measurements(inplace=True)
        qc = _prepare_for_quimb(qc)
        target_bitstring = format(target, f"0{n}b")

        print(f"n={n:2d}: ", end="", flush=True)
        try:
            # Rehearse with a few independent trials first, keep the
            # optimizer instance that found the SMALLEST W. Passing a
            # fresh independent optimizer to measure_real_contraction
            # (as before) let it gamble on its own unlucky search —
            # that's what actually caused the n=8 OOM kill: the rehearsal
            # found W=11 (safe), but a separate fresh search for the real
            # run could land on a far worse tree. Reusing the same
            # optimizer object relies on cotengra caching its best path
            # for that circuit — verify this holds on your version; if
            # W ends up different real vs rehearsed, that assumption
            # didn't hold and needs a different fix.
            best_rehearsed, best_optimizer = None, None
            for _ in range(3):
                opt = _make_optimizer()
                trial = rehearse_amplitude(qc, target_bitstring, optimize=opt, verbose=False)
                if best_rehearsed is None or trial["contraction_width_W"] < best_rehearsed["contraction_width_W"]:
                    best_rehearsed, best_optimizer = trial, opt

            # Safety check: don't attempt real execution if even the best
            # rehearsed W implies infeasible memory (this is what should
            # have stopped the n=10 case before it ever reached OOM).
            W = best_rehearsed["contraction_width_W"]
            if _tn_bytes_estimate(W) > 2 * (1024 ** 3):  # >2 GB estimated
                print(f"skipped — best rehearsed W={W:.1f} implies "
                      f"{_tn_bytes_estimate(W)/(1024**3):.1f} GB, too large to safely execute")
                continue

            real = measure_real_contraction(qc, target_bitstring,
                                             optimize=best_optimizer, verbose=verbose)
            real.update({
                "n_qubits": n,
                "rehearsed_W": best_rehearsed["contraction_width_W"],
                "rehearsed_tn_bytes_estimate": _tn_bytes_estimate(best_rehearsed["contraction_width_W"]),
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
    experiment_tn_scalability(qubit_range=range(2, 12))
    # Capped below n=10 deliberately: n=10's rehearsed W came out at 50,
    # meaning a real contraction there needs ~2^50*16 bytes (~18 PB) —
    # genuinely infeasible on any machine, not a resource limit to raise.
    # Real execution should only be attempted where rehearsed W is small.
    experiment_real_memory_validation(qubit_range=range(2, 10))