"""
grover_core.py
==============
Classical simulation of Grover's Search Algorithm using Qiskit's statevector
simulator. This module provides a clean, well-documented implementation that
mirrors the mathematical structure described in the project abstract.

Mathematical background
-----------------------
For an n-qubit system searching among N = 2^n items:
  - The oracle operator O flips the phase of the target state |t>:
        O|x> = -|x>  if x == target
        O|x> =  |x>  otherwise
  - The diffusion (inversion about average) operator D = 2|s><s| - I,
    where |s> = H^⊗n |0...0> is the uniform superposition.
  - One Grover iteration = D ∘ O
  - Optimal number of iterations ≈ (π/4)√N

The state vector has 2^n complex amplitudes. This is the root cause of
classical intractability: 30 qubits already requires ~16 GB of RAM.
"""

import numpy as np
import time
import psutil
import os
from math import pi, sqrt, floor

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler


# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------

def build_oracle(n_qubits: int, target: int) -> QuantumCircuit:
    """
    Build a phase oracle that flips the amplitude sign of |target>.

    Strategy: flip all qubits where the target bitstring has a '0', apply
    an n-controlled-Z gate, then flip back. This implements:
        |x> -> (-1)^[x == target] |x>

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the search space.
    target : int
        The index (0 to 2^n - 1) of the item we are searching for.

    Returns
    -------
    QuantumCircuit
        A circuit acting as the phase oracle.
    """
    oracle = QuantumCircuit(n_qubits, name="Oracle")

    # Convert target to binary string, MSB first
    target_bits = format(target, f"0{n_qubits}b")

    # Flip qubits corresponding to '0' bits in the target
    for i, bit in enumerate(reversed(target_bits)):
        if bit == '0':
            oracle.x(i)

    # Multi-controlled Z gate: flips phase only when all qubits are |1>
    # For 1 qubit: just Z. For 2: CZ. For n: use H + MCX + H trick.
    if n_qubits == 1:
        oracle.z(0)
    else:
        # Apply H to last qubit, MCX, H — equivalent to multi-controlled Z
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)

    # Undo the initial flips
    for i, bit in enumerate(reversed(target_bits)):
        if bit == '0':
            oracle.x(i)

    return oracle


# ---------------------------------------------------------------------------
# Diffusion operator (Grover diffuser)
# ---------------------------------------------------------------------------

def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """
    Build the Grover diffusion operator: D = 2|s><s| - I

    This is implemented as:
        H^⊗n · (2|0><0| - I) · H^⊗n

    The inner operator (2|0><0| - I) flips the phase of everything
    *except* |0...0>. Combined with Hadamards, this produces inversion
    about the mean of all amplitudes.

    Geometric insight: each Grover iteration rotates the state vector
    by angle 2θ where sin(θ) = 1/√N. After k iterations the angle from
    the target is (π/2 - (2k+1)θ), which reaches 0 after ~π/(4θ) ≈ (π/4)√N steps.
    """
    diffuser = QuantumCircuit(n_qubits, name="Diffuser")

    # Map back to computational basis
    diffuser.h(range(n_qubits))

    # Flip phase of everything except |0...0>
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))

    # Return to Hadamard basis
    diffuser.h(range(n_qubits))

    return diffuser


# ---------------------------------------------------------------------------
# Full Grover circuit
# ---------------------------------------------------------------------------

def build_grover_circuit(n_qubits: int, target: int, n_iterations: int = None) -> QuantumCircuit:
    """
    Assemble the full Grover search circuit.

    Steps:
    1. Initialize uniform superposition: H^⊗n |0...0>
    2. Apply Grover iterations (Oracle + Diffuser) k times
    3. Measure all qubits

    Parameters
    ----------
    n_qubits : int
    target : int
    n_iterations : int, optional
        Defaults to the theoretically optimal floor(π/4 · √N).

    Returns
    -------
    QuantumCircuit
    """
    N = 2 ** n_qubits
    if n_iterations is None:
        n_iterations = floor((pi / 4) * sqrt(N))
        n_iterations = max(1, n_iterations)  # at least 1 iteration

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: uniform superposition
    qc.h(range(n_qubits))
    qc.barrier()

    # Step 2: Grover iterations
    oracle = build_oracle(n_qubits, target)
    diffuser = build_diffuser(n_qubits)

    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
        qc.barrier()

    # Step 3: measurement
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def simulate_grover(n_qubits: int, target: int, n_shots: int = 1024,
                    n_iterations: int = None, verbose: bool = True):
    """
    Build and run the Grover circuit using Qiskit Aer statevector simulator.

    Returns a dict with timing, memory, and measurement results.
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB

    t_build_start = time.perf_counter()
    qc = build_grover_circuit(n_qubits, target, n_iterations)
    t_build_end = time.perf_counter()

    # Use statevector-based Aer simulator for exact simulation
    backend = AerSimulator(method="statevector")

    t_sim_start = time.perf_counter()
    job = backend.run(qc, shots=n_shots)
    result = job.result()
    t_sim_end = time.perf_counter()

    mem_after = process.memory_info().rss / (1024 ** 2)

    counts = result.get_counts()
    total_shots = sum(counts.values())
    target_bitstr = format(target, f"0{n_qubits}b")
    # Qiskit returns bits in reversed order (qubit 0 = rightmost)
    target_bitstr_qiskit = target_bitstr[::-1]
    success_count = counts.get(target_bitstr_qiskit, 0)
    success_probability = success_count / total_shots

    N = 2 ** n_qubits
    optimal_iters = floor((pi / 4) * sqrt(N))
    actual_iters = n_iterations if n_iterations is not None else optimal_iters

    stats = {
        "n_qubits": n_qubits,
        "N": N,
        "target": target,
        "n_iterations": actual_iters,
        "optimal_iterations": optimal_iters,
        "n_shots": n_shots,
        "success_count": success_count,
        "success_probability": success_probability,
        "build_time_s": t_build_end - t_build_start,
        "sim_time_s": t_sim_end - t_sim_start,
        "total_time_s": (t_build_end - t_build_start) + (t_sim_end - t_sim_start),
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
        "mem_delta_mb": mem_after - mem_before,
        "statevector_size_mb": (2 ** n_qubits * 16) / (1024 ** 2),  # complex128
        "counts": counts,
        "circuit_depth": qc.depth(),
        "circuit_gate_count": qc.size(),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Grover Search — {n_qubits} qubits, N = {N}")
        print(f"  Target:          {target} ({target_bitstr})")
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
