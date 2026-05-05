# Grover's Algorithm — Classical Simulation
**Bachelor Semester Project | University of Luxembourg**
Veselin Petrov Nikolaev · Supervisors: Marko Rancic, Maximilian Streitberger

---

## The Goal
This project classically simulates Grover's quantum search algorithm to:
1. Understand **why** it works (mathematical + geometric structure)
2. Measure **where** classical simulation breaks down (exponential memory wall)
3. Compare **CPU vs GPU** performance on HPC infrastructure
4. Evaluate the **realistic impact** of the quadratic speedup

---

## File Structure
```
grover-s-algorithm/
│
├── grover_core.py        ← Oracle, diffuser, full circuit, simulation runner
├── experiments.py        ← All 5 experiments (run this first)
├── plot_results.py       ← Generate PDF/PNG figures from results
├── hpc_runner.py         ← SLURM-ready script for n > 22 on HPC
├── requirements.txt
├── README.md
│
├── hpc/
│   ├── jobs/
│   │   ├── submit_cpu.sh         ← SLURM batch script (CPU, AION cluster)
│   │   └── submit_gpu.sh         ← SLURM batch script (GPU, IRIS cluster)
│   └── logs/
│       ├── grover_cpu_11711885.log
│       └── grover_gpu_5379637.log
│
├── results/
│   ├── scalability.csv
│   ├── iteration_sweep.csv
│   ├── classical_comparison.csv
│   ├── circuit_depth.csv
│   ├── gpu_comparison.csv
│   ├── hpc_scalability_cpu.csv
│   └── hpc_scalability_gpu.csv
│
└── figures/
    ├── fig1_scalability.png/pdf
    ├── fig2_iteration_sweep.png/pdf
    ├── fig3_speedup.png/pdf
    ├── fig4_circuit_depth.png/pdf
    ├── fig5_gpu.png/pdf
    └── fig6_hpc.png/pdf
```

---

## Requirements

- **Python 3.9–3.12** (Python 3.13+ is not supported by qiskit-aer-gpu)
- Tested on Python 3.11.5

---

## Setup

```bash
pip install -r requirements.txt
```

### GPU Support

GPU simulation requires:
- **Linux** (x86_64) — qiskit-aer-gpu has no Windows native wheels
- **NVIDIA GPU** with CUDA 11.x or 12.x drivers
- **Python 3.9–3.12**

> **Note:** Pin exact versions as shown below. `qiskit>=1.3` breaks the
> `convert_to_target` import in qiskit-aer-gpu 0.15.x.

```bash
pip install "qiskit==1.3.0" "qiskit-aer-gpu==0.15.0" --no-deps
pip install custatevec-cu12 cutensornet-cu12 cutensor-cu12 --no-deps
pip install numpy pandas matplotlib psutil
```

---

## HPC Cluster Setup (University of Luxembourg)

Two clusters were used: **AION** (CPU) and **IRIS** (GPU).

### CPU — AION Cluster

```bash
ssh -p 8022 <user>@access-aion.uni.lu
module load lang/Python/3.11.5-GCCcore-13.2.0
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sbatch submit.sh
```

**submit.sh:**
```bash
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
```

### GPU — IRIS Cluster

IRIS uses **Tesla V100-SXM2-32GB** GPUs (Volta architecture) with CUDA 12.6.

```bash
ssh -p 8022 <user>@access-iris.uni.lu
# Run an interactive GPU job to set up the environment
srun --partition=gpu --gres=gpu:volta:1 --time=00:30:00 --mem=8G --pty bash -l

module load lang/Python/3.11.5-GCCcore-13.2.0
module load system/CUDA/12.6.0
python3 -m venv venv_gpu
source venv_gpu/bin/activate
pip install qiskit==1.3.0 numpy pandas matplotlib psutil
pip install qiskit-aer-gpu==0.15.0 --no-deps
pip install custatevec-cu12 cutensornet-cu12 cutensor-cu12 --no-deps
exit

sbatch submit_gpu.sh
```

**submit_gpu.sh:**
```bash
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
```

---

## Running

```bash
# 1. Run all experiments (takes ~10–30 min depending on hardware)
python experiments.py

# 2. Generate all figures
python plot_results.py

# 3. On HPC cluster (large n):
sbatch hpc/jobs/submit_cpu.sh     # CPU on AION
sbatch hpc/jobs/submit_gpu.sh    # GPU on IRIS
```

---

## Experiments

| # | Name | What it measures |
|---|------|-----------------|
| 1 | Scalability | sim time & memory vs n_qubits |
| 2 | Iteration sweep | P(success) vs k — shows overshoot |
| 3 | Classical comparison | query count: classical O(N/2) vs Grover O(√N) |
| 4 | Circuit depth | depth & gate count vs n |
| 5 | GPU comparison | CPU vs GPU speedup, crossover point |
| — | 2-qubit example | step-by-step numpy trace, verified with Qiskit |

---

## Sample Output — Local (RTX 4060 Laptop, WSL2/Ubuntu 22.04)

Experiment 5 measures the CPU vs GPU crossover point on local hardware:

```
n=10: CPU=0.015s, GPU=1.658s   ← GPU cold-start / CUDA init penalty
n=11: CPU=0.026s, GPU=0.055s
n=12: CPU=0.041s, GPU=0.079s
n=13: CPU=0.091s, GPU=0.107s
n=14: CPU=0.210s, GPU=0.156s   ← crossover: GPU faster from here
n=15: CPU=0.144s, GPU=0.183s
n=16: CPU=0.248s, GPU=0.273s
n=17: CPU=0.501s, GPU=0.440s
n=18: CPU=1.082s, GPU=0.766s
n=19: CPU=2.620s, GPU=1.920s
n=20: CPU=8.435s, GPU=4.494s
n=21: CPU=19.661s, GPU=12.606s
n=22: CPU=113.298s, GPU=32.823s
n=23: CPU=347.398s, GPU=99.271s   ← GPU ~3.5x faster, gap widening
```

GPU crossover occurs around **n=14** on this hardware (statevector ≈ 256 KB).
For large n, the GPU advantage grows as memory bandwidth and parallelism dominate.

---

## HPC Results

Simulations were run on the University of Luxembourg HPC infrastructure.
CPU job: AION cluster, AMD EPYC, 32 cores, 128 GB RAM.
GPU job: IRIS cluster, NVIDIA Tesla V100-SXM2-32GB, 32 GB VRAM, 128 GB RAM.
Both used 128 shots per circuit.

### Raw Results

| n | N | CPU time (s) | GPU time (s) | Speedup |
|---|---|-------------|-------------|---------|
| 20 | 1,048,576 | 8.40 | 3.07 | 2.7× |
| 21 | 2,097,152 | 15.27 | 5.54 | 2.8× |
| 22 | 4,194,304 | 64.73 | 11.47 | 5.6× |
| 23 | 8,388,608 | 202.22 | 25.99 | 7.8× |
| 24 | 16,777,216 | 714.59 | 64.52 | 11.1× |
| 25 | 33,554,432 | 1766.42 | 173.74 | 10.2× |
| 26 | 67,108,864 | 5753.91 | 485.30 | 11.9× |
| 27 | 134,217,728 | 16677.98 | 1385.81 | 12.0× |
| 28 | 268,435,456 | 48927.59 | 4009.61 | 12.2× |
| 29 | 536,870,912 | — (time limit) | 11708.95 | — |
| 30 | 1,073,741,824 | — | 34077.69 | — |
| 31 | 2,147,483,648 | — | 93411.24 | — |
| 32 | 4,294,967,296 | — | stopped (mem) | — |

CPU reached n=28 before hitting the 48-hour wall during n=29.
GPU reached n=31 before running out of V100 VRAM (32 GB) at n=32.

### Key Observations

The GPU advantage grows consistently with qubit count — from ~2.7× at n=20 to ~12× at n=28. This reflects the V100's memory bandwidth and parallel execution becoming dominant as the statevector grows. The crossover where GPU becomes meaningfully faster occurs around n=21–22 on this hardware.

Simulation time roughly quadruples with each additional qubit on both devices, consistent with the 2× state space growth combined with the O(√N) circuit depth increase in Grover's algorithm.

---

## Key Insight: The Memory Wall

The statevector for n qubits has 2^n complex128 entries (16 bytes each):

| n  | Memory    | CPU time   | GPU time   |
|----|-----------|------------|------------|
| 20 | 16 MB     | 8s         | 3s         |
| 25 | 512 MB    | ~29 min    | ~3 min     |
| 28 | 4 GB      | ~13.6 hrs  | ~1.1 hrs   |
| 31 | 32 GB     | —          | ~26 hrs    |
| 32 | 64 GB     | —          | beyond V100 VRAM |
| 33 | 128 GB    | beyond RAM | —          |

This is not an algorithmic limitation — it is a consequence of quantum mechanics
requiring us to track all 2^n amplitudes simultaneously. Real quantum hardware
does not have this problem: it *is* the superposition, not a simulation of it.

---

## Grover's Speedup: Realistic Assessment

- Classical unstructured search: **O(N)** queries on average
- Grover: **O(√N)** queries — a quadratic, not exponential speedup
- For N = 2^28 (~268M): classical ≈ 134M queries → Grover ≈ 8192 queries
- **Practical limits today**: decoherence limits real quantum hardware to ~10–20
  qubits for error-free Grover. The simulation is noise-free; real hardware is not.
- **Classical simulation wall**: even with a 32 GB V100 GPU, we can only simulate
  up to n=31. A 33-qubit circuit would require 128 GB of contiguous GPU memory —
  beyond any current single GPU.
