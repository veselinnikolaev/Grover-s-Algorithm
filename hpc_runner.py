"""
hpc_runner.py
=============
High-Performance Computing runner for large qubit counts (n > 22).
Submit this as a batch job on a cluster with SLURM.

Example SLURM script (save as submit.sh):
------------------------------------------
#!/bin/bash -l
#SBATCH --job-name=grover_sim
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --output=grover_%j.log
#SBATCH --error=grover_%j.err
#SBATCH --partition=batch

module load lang/Python/3.11.5-GCCcore-13.2.0

cd ~/grover_hpc
source venv/bin/activate

python hpc_runner.py --min_qubits 20 --max_qubits 30 --shots 128
------------------------------------------

For GPU partition:
#!/bin/bash -l
#SBATCH --job-name=grover_gpu
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:volta:1
#SBATCH --output=grover_gpu_%j.log
#SBATCH --error=grover_gpu_%j.err
#SBATCH --partition=gpu

module load lang/Python/3.11.5-GCCcore-13.2.0
module load system/CUDA/12.6.0

export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

cd ~/grover_hpc
source venv_gpu/bin/activate

python hpc_runner.py --min_qubits 20 --max_qubits 33 --gpu --shots 128
"""

import argparse
import numpy as np
import time
import os
import psutil
import gc

os.makedirs("results", exist_ok=True)


def get_available_ram_mb():
    return psutil.virtual_memory().available / (1024**2)


def run_hpc_scalability(min_n, max_n, n_shots, use_gpu=False):
    """
    Extended scalability experiment for HPC.
    Memory check before each run to avoid OOM kills.
    """
    from qiskit_aer import AerSimulator
    from grover_core import build_grover_circuit

    device = "GPU" if use_gpu else "CPU"
    backend = AerSimulator(method="statevector", device=device)

    records = []
    import csv
    out_path = f"results/hpc_scalability_{'gpu' if use_gpu else 'cpu'}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_qubits", "N", "sim_time_s", "mem_mb_theoretical", "device"])

        for n in range(min_n, max_n + 1):
            mem_needed = (2**n * 16) / (1024**2)
            available = get_available_ram_mb()

            print(f"n={n:2d}: need {mem_needed:.0f} MB, available {available:.0f} MB", flush=True)

            # Safety margin: require 2× the statevector size free
            if mem_needed * 2 > available:
                print(f"  → STOPPING: insufficient RAM (need {mem_needed*2:.0f} MB, have {available:.0f} MB)")
                break

            target = np.random.randint(0, 2**n)
            qc = build_grover_circuit(n, target)

            t0 = time.perf_counter()
            try:
                result = backend.run(qc, shots=n_shots).result()
                elapsed = time.perf_counter() - t0
            except MemoryError:
                print(f"  → MemoryError at n={n} — this is the classical wall!")
                break

            print(f"  → time={elapsed:.2f}s  (mem={mem_needed:.0f} MB)", flush=True)
            writer.writerow([n, 2**n, elapsed, mem_needed, device])
            f.flush()

            # Explicit garbage collection between large runs
            del result
            gc.collect()

    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_qubits", type=int, default=20)
    parser.add_argument("--max_qubits", type=int, default=30)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    print(f"HPC Grover simulation: n={args.min_qubits}..{args.max_qubits}, "
          f"device={'GPU' if args.gpu else 'CPU'}, shots={args.shots}")
    run_hpc_scalability(args.min_qubits, args.max_qubits, args.shots, args.gpu)
