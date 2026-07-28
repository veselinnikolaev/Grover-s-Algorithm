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

    CAVEAT (confirmed empirically): max_time/max_repeats bound the
    search's TOTAL BUDGET across repeated trials, but not the cost of a
    SINGLE trial. As the mcx-decomposed circuit grows with n, each
    individual trial's cost-evaluation itself gets slower, so past some
    n the cap stops being effective — observed roughly 4x growth per
    qubit from n=6 to n=11 (5.6s -> 1044s), which would put n=20+ at
    days, not seconds. For a wide sweep (e.g. up to n=33), use
    _make_greedy_optimizer() below instead.
    """
    return ctg.HyperOptimizer(
        methods=["greedy", "kahypar"],
        max_time=max_time,
        max_repeats=max_repeats,
        progbar=False,
    )


def _make_greedy_optimizer():
    """
    Single-shot deterministic greedy optimizer — no repeated trials, no
    search budget to blow past. Cost stays roughly proportional to
    circuit size instead of compounding, which is what makes a full
    n=2..33 sweep actually finishable.

    Tradeoff: greedy alone typically finds a WORSE (higher W/C) tree than
    HyperOptimizer's best-of-many-trials search would. This is a
    real methodological tradeoff worth stating explicitly in a report:
    the reported W/C with this optimizer are upper bounds on the true
    cost, not the best achievable estimate — use _make_optimizer() (the
    bounded multi-trial search) for smaller n where its cost is still
    affordable, and this for the wide sweep where it isn't.
    """
    return ctg.HyperOptimizer(
        methods=["greedy"],
        max_repeats=1,
        max_time=5,
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
      - resource.getrusage ru_maxrss: the process's PEAK resident set
        size over its ENTIRE lifetime so far, in KB on Linux. Used
        instead of a psutil before/after RSS delta, because a delta
        undercounts in a long-lived process: once the allocator has
        grown the process once (e.g. a big earlier n), it often reuses
        those already-mapped pages for later smaller allocations instead
        of requesting fresh memory from the OS, making rss_after -
        rss_before read as ~0 even though real memory was used. ru_maxrss
        is monotonically non-decreasing and doesn't have this problem —
        but note it's a LIFETIME peak, not this call's peak alone, so
        run each n in its own fresh subprocess if you need a clean
        per-n number rather than a running high-water mark.

    Only run this for small/moderate n where the REHEARSED W is small
    (roughly W<30) — unlike rehearse_amplitude, this really executes the
    contraction and pays the full 2^W-element cost; a large rehearsed W
    means this will exhaust memory (see the n=10 case where W=50 implied
    ~18 PB and the process was OOM-killed — that's not a bug, the real
    contraction genuinely needs that much memory).
    """
    import resource

    if optimize is None:
        optimize = _make_optimizer()

    circ = quimb_circuit(qc)

    tracemalloc.start()
    t0 = time.perf_counter()
    amp = circ.amplitude(target_bitstring, optimize=optimize)
    elapsed = time.perf_counter() - t0
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ru_maxrss is in KB on Linux (bytes on macOS) — this codebase
    # targets WSL/Linux, so KB is assumed here
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    stats = {
        "amplitude_real_part": amp.real if hasattr(amp, "real") else amp,
        "real_time_s": elapsed,
        "tracemalloc_peak_mb": tracemalloc_peak / (1024 ** 2),
        "peak_rss_mb_lifetime": peak_rss_mb,
    }
    if verbose:
        print(f"    [real] time={elapsed:.3f}s  "
              f"tracemalloc_peak={stats['tracemalloc_peak_mb']:.2f}MB  "
              f"peak_rss(lifetime)={stats['peak_rss_mb_lifetime']:.2f}MB")
    return stats