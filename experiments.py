"""
experiments.py
==============
Systematic experimental evaluation of Grover's algorithm simulation.

Experiments performed:
  1. Scalability sweep     — time & memory vs number of qubits
  2. Iteration sweep       — success probability vs k (Grover iterations)
  3. Multi-target baseline — classical O(N) linear search comparison
  4. Architecture notes    — CPU vs GPU vs HPC guidance

Run this file directly:
    python experiments.py

Results are saved to ./results/ as CSV files for plotting.
"""

import numpy as np
import pandas as pd
import time
import os
import warnings
from math import pi, sqrt, floor

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.makedirs("results", exist_ok=True)

from grover_core import simulate_grover, build_grover_circuit


# ===========================================================================
# Experiment 1: Scalability — time and memory vs qubit count
# ===========================================================================

def experiment_scalability(qubit_range=range(1, 21), n_shots=512, verbose=True):
    """
    Measure simulation time and memory usage as a function of n_qubits.

    Key insight: the statevector has 2^n complex128 entries (16 bytes each).
    Memory grows as: M(n) = 16 × 2^n bytes
    
        n=20  →  16 MB
        n=25  →  512 MB
        n=28  →  4 GB
        n=30  →  16 GB   ← typical RAM limit
        n=33  →  128 GB  ← HPC territory

    This is the fundamental wall that prevents classical simulation of
    large quantum systems — not algorithmic complexity, but sheer memory.
    """
    print("\n" + "="*60)
    print("  EXPERIMENT 1: Scalability (time & memory vs qubits)")
    print("="*60)

    records = []
    for n in qubit_range:
        theoretical_mem_mb = (2**n * 16) / (1024**2)
        # Safety check: skip if theoretical memory > 8 GB
        if theoretical_mem_mb > 8192:
            print(f"  n={n:2d}: Skipped — theoretical memory {theoretical_mem_mb/1024:.1f} GB exceeds limit")
            break

        target = np.random.randint(0, 2**n)
        try:
            stats = simulate_grover(n, target, n_shots=n_shots, verbose=verbose)
            records.append(stats)
        except MemoryError:
            print(f"  n={n:2d}: MemoryError — this is the classical simulation wall!")
            break
        except Exception as e:
            print(f"  n={n:2d}: Error — {e}")
            break

    df = pd.DataFrame(records)
    df.to_csv("results/scalability.csv", index=False)
    print(f"\n  Saved: results/scalability.csv  ({len(df)} rows)")
    return df


# ===========================================================================
# Experiment 2: Iteration sweep — P(success) vs k
# ===========================================================================

def experiment_iteration_sweep(n_qubits=4, n_shots=2048):
    """
    For a fixed n, vary the number of Grover iterations k from 1 to 3·k_opt.
    
    Expected behavior:
    - P(success) rises with k, peaks near k_opt = floor(π/4 · √N)
    - Then *overshoots* and falls — this is unique to quantum algorithms!
      Over-rotating past the target state reduces the success probability.
    - The pattern is sinusoidal: P(k) ≈ sin²((2k+1)·arcsin(1/√N))

    This experiment beautifully illustrates the geometric picture:
    each iteration rotates the state by a fixed angle in 2D subspace,
    and rotating too far is just as bad as not rotating enough.
    """
    print("\n" + "="*60)
    print(f"  EXPERIMENT 2: Iteration Sweep  (n={n_qubits}, N={2**n_qubits})")
    print("="*60)

    N = 2 ** n_qubits
    k_opt = floor((pi / 4) * sqrt(N))
    target = N // 3  # fixed, arbitrary target

    records = []
    max_k = min(3 * k_opt + 2, 60)

    for k in range(1, max_k + 1):
        stats = simulate_grover(n_qubits, target, n_shots=n_shots,
                                n_iterations=k, verbose=False)
        # Also compute theoretical probability
        theta = np.arcsin(1 / sqrt(N))
        p_theory = np.sin((2 * k + 1) * theta) ** 2
        stats["k"] = k
        stats["p_theory"] = p_theory
        records.append(stats)
        if k % 5 == 0 or k == k_opt:
            marker = " ← k_opt" if k == k_opt else ""
            print(f"  k={k:3d}: P(success)={stats['success_probability']:.4f}  "
                  f"[theory: {p_theory:.4f}]{marker}")

    df = pd.DataFrame(records)
    df.to_csv("results/iteration_sweep.csv", index=False)
    print(f"\n  Saved: results/iteration_sweep.csv")
    return df


# ===========================================================================
# Experiment 3: Classical vs quantum comparison
# ===========================================================================

def experiment_classical_comparison(qubit_range=range(1, 18), n_trials=5):
    """
    Compare Grover's O(√N) query complexity with classical O(N) linear search.

    Important nuance: Grover's speedup is *quadratic*, not exponential.
    It does NOT break RSA or AES entirely — it halves the effective key length.
    The practical impact depends on context.

    We simulate classical search by measuring actual Python loop time,
    then compare against our quantum simulation time.
    Note: simulation time ≠ quantum hardware time. Classical simulation
    is inherently slow; the comparison of interest is *query count*.
    """
    print("\n" + "="*60)
    print("  EXPERIMENT 3: Classical vs Grover — Query Complexity")
    print("="*60)

    records = []
    for n in qubit_range:
        N = 2 ** n

        # Classical linear search — expected O(N/2) on average
        classical_times = []
        for _ in range(n_trials):
            database = list(range(N))
            target = np.random.randint(0, N)
            t0 = time.perf_counter()
            _ = database.index(target)
            classical_times.append(time.perf_counter() - t0)

        classical_avg_time = np.mean(classical_times)
        classical_queries = N / 2  # expected

        # Grover query count (theoretical — no simulation overhead)
        grover_queries = floor((pi / 4) * sqrt(N))
        speedup = classical_queries / grover_queries

        records.append({
            "n_qubits": n,
            "N": N,
            "classical_queries_expected": classical_queries,
            "grover_queries": grover_queries,
            "speedup_factor": speedup,
            "classical_time_s": classical_avg_time,
        })
        print(f"  n={n:2d}, N={N:8d}: classical≈{classical_queries:.0f} queries, "
              f"Grover≈{grover_queries:4d}, speedup={speedup:.1f}×")

    df = pd.DataFrame(records)
    df.to_csv("results/classical_comparison.csv", index=False)
    print(f"\n  Saved: results/classical_comparison.csv")
    return df


# ===========================================================================
# Experiment 4: Circuit depth analysis
# ===========================================================================

def experiment_circuit_depth(qubit_range=range(2, 16)):
    """
    Measure how circuit depth and gate count scale with qubits and iterations.
    
    On real quantum hardware, circuit depth is critical: each gate introduces
    noise, and qubits decohere. This is why Grover on real hardware today
    is limited to tiny registers. The simulation has no such constraint.
    """
    print("\n" + "="*60)
    print("  EXPERIMENT 4: Circuit Depth & Gate Count")
    print("="*60)

    records = []
    for n in qubit_range:
        N = 2 ** n
        k_opt = max(1, floor((pi / 4) * sqrt(N)))
        qc = build_grover_circuit(n, target=0, n_iterations=k_opt)
        records.append({
            "n_qubits": n,
            "N": N,
            "k_opt": k_opt,
            "circuit_depth": qc.depth(),
            "gate_count": qc.size(),
        })
        print(f"  n={n:2d}: depth={qc.depth():6d}, gates={qc.size():8d}, k_opt={k_opt}")

    df = pd.DataFrame(records)
    df.to_csv("results/circuit_depth.csv", index=False)
    print(f"\n  Saved: results/circuit_depth.csv")
    return df


# ===========================================================================
# Experiment 5: GPU simulation (if available)
# ===========================================================================

def experiment_gpu_comparison(qubit_range=range(10, 26), n_shots=256):
    """
    Compare CPU vs GPU statevector simulation using Qiskit Aer.

    Qiskit Aer supports GPU acceleration via CUDA (method='statevector' with
    device='GPU'). GPU parallelism helps because each Grover step applies
    the same unitary to all 2^n amplitudes simultaneously — perfect for SIMD.

    For small n: CPU wins (GPU overhead dominates)
    For large n: GPU wins (bandwidth + parallelism)
    The crossover is typically around n=20-22 for modern hardware.
    """
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    print("\n" + "="*60)
    print("  EXPERIMENT 5: CPU vs GPU Simulation")
    print("="*60)

    # Check GPU availability
    try:
        gpu_backend = AerSimulator(method="statevector", device="GPU")
        gpu_available = True
        print("  GPU backend: available ✓")
    except Exception:
        gpu_available = False
        print("  GPU backend: not available — running CPU only")
        print("  (Install qiskit-aer-gpu and CUDA to enable GPU simulation)")

    cpu_backend = AerSimulator(method="statevector")
    records = []

    for n in qubit_range:
        theoretical_mem_mb = (2**n * 16) / (1024**2)
        if theoretical_mem_mb > 4096:
            print(f"  n={n}: skipping — {theoretical_mem_mb/1024:.1f} GB needed")
            break

        target = np.random.randint(0, 2**n)
        qc = build_grover_circuit(n, target)
        row = {"n_qubits": n, "N": 2**n, "mem_mb": theoretical_mem_mb}

        # CPU timing
        t0 = time.perf_counter()
        result_cpu = cpu_backend.run(qc, shots=n_shots).result()
        row["cpu_time_s"] = time.perf_counter() - t0

        # GPU timing (if available)
        if gpu_available:
            try:
                t0 = time.perf_counter()
                result_gpu = gpu_backend.run(qc, shots=n_shots).result()
                row["gpu_time_s"] = time.perf_counter() - t0
                row["speedup"] = row["cpu_time_s"] / row["gpu_time_s"]
            except Exception as e:
                row["gpu_time_s"] = None
                row["speedup"] = None
                print(f"  n={n}: GPU error — {e}")
        else:
            row["gpu_time_s"] = None
            row["speedup"] = None

        records.append(row)
        gpu_str = f", GPU={row['gpu_time_s']:.3f}s" if row["gpu_time_s"] else ""
        print(f"  n={n:2d}: CPU={row['cpu_time_s']:.3f}s{gpu_str}  "
              f"(mem={theoretical_mem_mb:.1f} MB)")

    df = pd.DataFrame(records)
    df.to_csv("results/gpu_comparison.csv", index=False)
    print(f"\n  Saved: results/gpu_comparison.csv")
    return df


# ===========================================================================
# Hand-worked 2-qubit example (for the report)
# ===========================================================================

def two_qubit_hand_example():
    """
    Trace through Grover's algorithm on n=2, N=4, target=|11> (index 3).
    
    This matches the 'hand-derived two-qubit example using matrix formalism'
    mentioned in your abstract. We verify each step numerically.

    Initial state after H⊗H:
        |s> = 1/2 (|00> + |01> + |10> + |11>)

    After oracle (marks |11>):
        1/2 (|00> + |01> + |10> - |11>)

    After diffuser (invert about mean):
        |11>  (ideally, for N=4 exactly 1 iteration suffices)
    """
    print("\n" + "="*60)
    print("  2-QUBIT WORKED EXAMPLE  (n=2, N=4, target=|11>=3)")
    print("="*60)

    # Build each step manually with numpy
    I = np.eye(2)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    HH = np.kron(H, H)  # H ⊗ H

    # Initial state |00>
    psi_0 = np.array([1, 0, 0, 0], dtype=complex)
    psi_s = HH @ psi_0
    print(f"\n  After H⊗H:     {np.round(psi_s, 4)}")
    print(f"  (Uniform: all amplitudes = {1/2:.4f})")

    # Oracle for target=3 (|11>): flip sign of index 3
    O = np.diag([1, 1, 1, -1]).astype(complex)
    psi_after_oracle = O @ psi_s
    print(f"\n  After Oracle:  {np.round(psi_after_oracle, 4)}")

    # Diffuser: D = 2|s><s| - I
    s = psi_s.reshape(-1, 1)
    D = 2 * (s @ s.conj().T) - np.eye(4)
    psi_final = D @ psi_after_oracle
    print(f"\n  After Diffuser:{np.round(psi_final, 4)}")
    print(f"\n  Probabilities: {np.round(np.abs(psi_final)**2, 4)}")
    print(f"  P(target=|11>)= {np.abs(psi_final[3])**2:.4f}  ← should be 1.0")

    # Now verify with Qiskit
    print("\n  Verifying with Qiskit simulation...")
    stats = simulate_grover(2, target=3, n_shots=4096, n_iterations=1)
    print(f"  Qiskit P(success) = {stats['success_probability']:.4f}")
    print("="*60)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  Grover's Algorithm — Classical Simulation Experiments  #")
    print("#"*60)

    # Run the 2-qubit hand example first (fast)
    two_qubit_hand_example()

    # Experiment 1: scalability (up to ~20 qubits for a laptop)
    df_scale = experiment_scalability(qubit_range=range(2, 21), n_shots=512)

    # Experiment 2: iteration sweep on n=5
    df_iter = experiment_iteration_sweep(n_qubits=5, n_shots=2048)

    # Experiment 3: classical vs quantum query complexity
    df_compare = experiment_classical_comparison(qubit_range=range(1, 18))

    # Experiment 4: circuit depth
    df_depth = experiment_circuit_depth(qubit_range=range(2, 16))

    # Experiment 5: GPU (will show CPU-only if no GPU)
    df_gpu = experiment_gpu_comparison(qubit_range=range(10, 24), n_shots=256)

    print("\n\nAll experiments complete. Results saved in ./results/")
    print("Run:  python plot_results.py  to generate figures.")
