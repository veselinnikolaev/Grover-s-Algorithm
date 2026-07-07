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

Requires:
    pip install quimb cotengra qiskit-quimb
"""

import time
import quimb.tensor as qtn
from qiskit_quimb import quimb_circuit


def rehearse_amplitude(qc, target_bitstring, optimize="auto-hq", verbose=True):
    """
    Convert a measurement-free Qiskit circuit to a quimb Circuit, then
    rehearse the contraction needed to compute the amplitude of
    `target_bitstring` (i.e. the same quantity Grover's oracle marks).

    Returns dict with contraction_width_W, contraction_cost_C (log10 FLOPs),
    and rehearsal wall time.
    """
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