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

Uses `rehearse` mode: quimb finds a contraction path and reports its cost
WITHOUT running the contraction. This means we can get cost/width estimates
even for n where actually contracting would be too slow or memory-heavy —
exactly the regime past BSP2's statevector wall (n=31/32) that we care about.

Path-finding itself (via cotengra) is bounded with max_time/max_repeats so
that a bad n can't silently stall an unattended sweep — without this, the
optimizer search time (not the actual contraction cost) can dominate wall
clock, especially without the optional `kahypar` partitioner installed.

Requires:
    pip install quimb cotengra qiskit-quimb
Recommended (much better contraction trees, especially at larger n):
    pip install kahypar
"""

import time
import cotengra as ctg
from qiskit_quimb import quimb_circuit


def _make_optimizer(max_time=30, max_repeats=32):
    """
    Bounded cotengra optimizer. Tries kahypar-based partitioning if
    available (best quality trees), falls back to greedy otherwise so
    the sweep still runs on machines without kahypar installed.
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
    and rehearsal wall time.

    optimize: pass a cotengra optimizer instance to reuse across calls
    (recommended for a sweep — building a fresh HyperOptimizer per n adds
    overhead). If None, a bounded one-off optimizer is built internally.
    """
    if optimize is None:
        optimize = _make_optimizer(max_time=max_time, max_repeats=max_repeats)

    circ = quimb_circuit(qc)

    t0 = time.perf_counter()

    # NOTE: verify this method name once on your setup:
    #   print([a for a in dir(circ) if 'rehearse' in a])
    # If `amplitude_rehearse` doesn't exist under that exact name, fall back
    # to: tn = circ.amplitude_tn(target_bitstring); tn.contraction_cost(...)
    # for C, and inspect the cotengra tree object for width.
    info = circ.amplitude_rehearse(target_bitstring, optimize=optimize)

    elapsed = time.perf_counter() - t0

    stats = {
        "contraction_width_W": info["W"],
        "contraction_cost_C": info["C"],
        "rehearsal_time_s": elapsed,
    }

    if verbose:
        print(f"    W={stats['contraction_width_W']:.2f}  "
              f"C={stats['contraction_cost_C']:.2f}  "
              f"rehearse_time={elapsed:.3f}s")

    return stats