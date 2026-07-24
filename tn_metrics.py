"""
tn_metrics.py
=============
Extracts contraction cost (C) and contraction width (W) — the standard
quimb/cotengra metrics for tensor-network simulation cost — from a Qiskit
circuit, using the qiskit-quimb bridge.

W (contraction width) is the log2 size of the largest intermediate tensor
formed during contraction. It's the direct memory-cost analogue of the
statevector's 16*2^n bytes, and the number we compare against BSP2's
measured memory wall.
C (contraction cost) is the log10 FLOP estimate for the full contraction —
the TN-side analogue of BSP2's measured sim_time_s.

Two measurement modes are provided:

1. rehearse_amplitude(): quimb finds a contraction path and reports its
   cost WITHOUT running the contraction. Cheap, works even for n where
   actually contracting would be too slow/memory-heavy. This gives
   ESTIMATED W/C, not measured memory.

2. measure_real_contraction(): actually performs the contraction and
   measures real process memory. Only feasible for smaller n. This is
   the ground-truth check needed to validate that rehearsed W actually
   predicts real memory before trusting the rehearsed numbers out to
   n=33 — the abstract's "as the algorithm runs" language calls for
   this, not rehearsal alone.

Path-finding itself (via cotengra) is bounded with max_time/max_repeats so
that a bad n can't silently stall an unattended sweep — without this, the
optimizer search time (not the actual contraction cost) can dominate wall
clock, especially without the optional `kahypar` partitioner installed.

Requires:
    pip install quimb cotengra qiskit-quimb psutil
Recommended (much better contraction trees, especially at larger n):
    pip install kahypar
"""

import os
import time
import tracemalloc

import cotengra as ctg
from qiskit_quimb import quimb_circuit


def _make_optimizer(max_time=10, max_repeats=8):
    """
    Bounded cotengra optimizer. Tries kahypar-based partitioning if
    available (best quality trees), falls back to greedy otherwise so
    the sweep still runs on machines without kahypar installed.

    max_time/max_repeats kept small and identical across all n so that
    the sweep gives a comparable, bounded estimate at every qubit count
    rather than an unbounded best-effort search that gets slower as the
    decomposed circuit grows.
    """
    return ctg.HyperOptimizer(
        methods=["greedy", "kahypar"],
        max_time=max_time,
        max_repeats=max_repeats,
        progbar=False,
    )


def rehearse_amplitude(qc, target_bitstring, optimize=None, verbose=True,
                        max_time=30, max_repeats=32):
    """
    Convert a measurement-free Qiskit circuit to a quimb Circuit, then
    rehearse the contraction needed to compute the amplitude of
    `target_bitstring` (i.e. the same quantity Grover's oracle marks).

    Returns dict with contraction_width_W, contraction_cost_C (log10 FLOPs),
    plus conversion time and rehearsal time reported SEPARATELY.

    Reporting these two times separately matters: at n=8 a run of this
    sweep saw ~43s wall time against a supposedly-capped 10s cotengra
    search, meaning something outside the capped search (very plausibly
    the qiskit->quimb conversion step, given the mcx decomposition's
    gate-count blowup) is what's actually eating the time. If convert_time_s
    is the dominant term, the fix is upstream (fewer decomposed gates or a
    faster conversion path), not a bigger max_time/max_repeats budget.

    optimize: pass a cotengra optimizer instance to reuse across calls
    (recommended for a sweep — building a fresh HyperOptimizer per n adds
    overhead). If None, a bounded one-off optimizer is built internally.
    """
    if optimize is None:
        optimize = _make_optimizer(max_time=max_time, max_repeats=max_repeats)

    t_convert_0 = time.perf_counter()
    circ = quimb_circuit(qc)
    t_convert = time.perf_counter() - t_convert_0

    # NOTE: verify this method name once on your setup:
    #   print([a for a in dir(circ) if 'rehearse' in a])
    # Confirmed present as `amplitude_rehearse` as of this project's setup.
    t0 = time.perf_counter()
    info = circ.amplitude_rehearse(target_bitstring, optimize=optimize)
    t_rehearse = time.perf_counter() - t0

    stats = {
        "contraction_width_W": info["W"],
        "contraction_cost_C": info["C"],
        "convert_time_s": t_convert,
        "rehearsal_time_s": t_rehearse,
        "rehearsal_time_total_s": t_convert + t_rehearse,
    }

    if verbose:
        print(f"    W={stats['contraction_width_W']:.2f}  "
              f"C={stats['contraction_cost_C']:.2f}  "
              f"convert={t_convert:.3f}s  rehearse={t_rehearse:.3f}s")

    return stats


def measure_real_contraction(qc, target_bitstring, optimize=None, verbose=True):
    """
    Actually performs the contraction (not a rehearsal) and measures real
    memory two ways:
      - tracemalloc: tracks Python-allocator memory (numpy arrays go
        through this on most builds, but C-level buffers outside
        Python's allocator can be undercounted).
      - psutil RSS delta: tracks total process memory, a coarser but
        more trustworthy real-world number.

    Only run this for small/moderate n (start around n<=16-18) — unlike
    rehearse_amplitude, this really executes the contraction, so it pays
    the full cost the rehearsal is designed to let you avoid.
    """
    import psutil

    if optimize is None:
        optimize = _make_optimizer()

    circ = quimb_circuit(qc)

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    tracemalloc.start()
    t0 = time.perf_counter()
    amp = circ.amplitude(target_bitstring, optimize=optimize)
    elapsed = time.perf_counter() - t0
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = proc.memory_info().rss

    stats = {
        "amplitude_real_part": amp.real if hasattr(amp, "real") else amp,
        "real_time_s": elapsed,
        "tracemalloc_peak_mb": tracemalloc_peak / (1024 ** 2),
        "psutil_rss_delta_mb": (rss_after - rss_before) / (1024 ** 2),
    }
    if verbose:
        print(f"    [real] time={elapsed:.3f}s  "
              f"tracemalloc_peak={stats['tracemalloc_peak_mb']:.2f}MB  "
              f"rss_delta={stats['psutil_rss_delta_mb']:.2f}MB")
    return stats