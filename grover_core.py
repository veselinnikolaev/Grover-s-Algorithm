"""
grover_core.py
==============
Classical simulation of Grover's Search Algorithm using Qiskit's statevector
simulator.

Bit ordering (critical!)
------------------------
Qiskit measures qubit i -> classical bit i, printed with bit 0 RIGHTMOST.
Our oracle marks qubit i = bit i of target integer (LSB = qubit 0).
Result: the Qiskit measurement key for target N is just format(N, '0nb'),
i.e. standard binary — NO reversal needed when oracle is written LSB-first.
"""

import numpy as np
import time
import psutil
import os
from math import pi, sqrt, floor

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def build_oracle(n_qubits: int, target: int) -> QuantumCircuit:
    """
    Phase oracle: |x> -> -|x> if x==target, else |x>.
    Qubit i encodes bit i of target (LSB = qubit 0).
    """
    oracle = QuantumCircuit(n_qubits, name="Oracle")

    # Flip qubit i wherever bit i of target is 0
    for i in range(n_qubits):
        if not (target >> i) & 1:
            oracle.x(i)

    # Multi-controlled Z (phase flip when all qubits are |1>)
    if n_qubits == 1:
        oracle.z(0)
    elif n_qubits == 2:
        oracle.cz(0, 1)
    else:
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)

    # Undo flips
    for i in range(n_qubits):
        if not (target >> i) & 1:
            oracle.x(i)

    return oracle


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """
    Grover diffuser: D = 2|s><s| - I = H^n (2|0><0|-I) H^n
    Each iteration rotates the state by 2*arcsin(1/sqrt(N)).
    """
    diffuser = QuantumCircuit(n_qubits, name="Diffuser")

    if n_qubits == 1:
        diffuser.h(0)
        diffuser.z(0)
        diffuser.h(0)
        return diffuser

    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))

    if n_qubits == 2:
        diffuser.cz(0, 1)
    else:
        diffuser.h(n_qubits - 1)
        diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        diffuser.h(n_qubits - 1)

    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))
    return diffuser


def build_grover_circuit(n_qubits: int, target: int, n_iterations: int = None) -> QuantumCircuit:
    """Full Grover circuit: superposition -> k*(oracle+diffuser) -> measure."""
    N = 2 ** n_qubits
    if n_iterations is None:
        n_iterations = max(1, floor((pi / 4) * sqrt(N)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    qc.barrier()

    oracle = build_oracle(n_qubits, target)
    diffuser = build_diffuser(n_qubits)

    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
        qc.barrier()

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def simulate_grover(n_qubits: int, target: int, n_shots: int = 1024,
                    n_iterations: int = None, verbose: bool = True):
    """Run Grover simulation and return stats dict."""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)

    t0 = time.perf_counter()
    qc = build_grover_circuit(n_qubits, target, n_iterations)
    t_built = time.perf_counter()

    backend = AerSimulator(method="statevector")
    job = backend.run(qc, shots=n_shots)
    result = job.result()
    t_done = time.perf_counter()

    mem_after = process.memory_info().rss / (1024 ** 2)
    counts = result.get_counts()
    total_shots = sum(counts.values())

    # Oracle marks qubit i = bit i of target (LSB=qubit 0).
    # Qiskit prints classical bit 0 rightmost, so bit 0 of the key string
    # is also the rightmost character — matching standard binary format.
    # Therefore: target key = format(target, '0nb'), NO reversal.
    target_key = format(target, f"0{n_qubits}b")
    success_count = counts.get(target_key, 0)
    success_probability = success_count / total_shots

    top_result = max(counts, key=counts.get)
    top_int = int(top_result, 2)

    N = 2 ** n_qubits
    optimal_iters = max(1, floor((pi / 4) * sqrt(N)))
    actual_iters = n_iterations if n_iterations is not None else optimal_iters

    stats = {
        "n_qubits": n_qubits, "N": N, "target": target,
        "n_iterations": actual_iters, "optimal_iterations": optimal_iters,
        "n_shots": n_shots, "success_count": success_count,
        "success_probability": success_probability,
        "build_time_s": t_built - t0, "sim_time_s": t_done - t_built,
        "total_time_s": t_done - t0,
        "mem_before_mb": mem_before, "mem_after_mb": mem_after,
        "mem_delta_mb": mem_after - mem_before,
        "statevector_size_mb": (2 ** n_qubits * 16) / (1024 ** 2),
        "counts": counts,
        "circuit_depth": qc.depth(), "circuit_gate_count": qc.size(),
    }

    if verbose:
        match = "✓ MATCH" if top_int == target else f"✗ GOT {top_int}"
        print(f"\n{'='*60}")
        print(f"  Grover Search — {n_qubits} qubits, N = {N}")
        print(f"  Target:          {target}  (binary: {target_key})")
        print(f"  Top measurement: {top_int}  ({top_result})  {match}")
        print(f"  Iterations:      {actual_iters}  (optimal ≈ {optimal_iters})")
        print(f"  Circuit depth:   {stats['circuit_depth']}")
        print(f"  Circuit gates:   {stats['circuit_gate_count']}")
        print(f"  Build time:      {stats['build_time_s']:.4f} s")
        print(f"  Sim time:        {stats['sim_time_s']:.4f} s")
        print(f"  Memory delta:    {stats['mem_delta_mb']:.1f} MB")
        print(f"  Statevec size:   {stats['statevector_size_mb']:.2f} MB (theoretical)")
        print(f"  P(success):      {success_probability:.4f}  "
              f"({'✓ FOUND' if success_probability > 0.5 else '✗ LOW'})")
        print(f"{'='*60}")

    return stats
