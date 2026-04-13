# Grover's Algorithm — Classical Simulation
**Bachelor Semester Project | University of Luxembourg**
Veselin Petrov Nikolaev · Supervisors: Marko Rancic, Maximilian Streitberger

---

## The Goal
This project classically simulates Grover's quantum search algorithm to:
1. Understand **why** it works (mathematical + geometric structure)
2. Measure **where** classical simulation breaks down (exponential memory wall)
3. Compare **CPU vs GPU vs HPC** performance
4. Evaluate the **realistic impact** of the quadratic speedup

---

## File Structure
```
grover_simulation/
│
├── grover_core.py        ← Oracle, diffuser, full circuit, simulation runner
├── experiments.py        ← All 5 experiments (run this first)
├── plot_results.py       ← Generate PDF/PNG figures from results
├── hpc_runner.py         ← SLURM-ready script for n > 22 on HPC
├── requirements.txt
│
├── results/              ← CSV output from experiments (sample included)
└── figures/              ← PDF/PNG figures for the report (sample included)
```

---

## Requirements

- **Python 3.9–3.12** (Python 3.13+ is not supported by qiskit-aer-gpu)
- Tested on Python 3.12.13

---

## Setup

```bash
pip install -r requirements.txt
```

### GPU Support (optional)

GPU simulation requires:
- **Linux** (x86_64) — qiskit-aer-gpu has no Windows native wheels
- **NVIDIA GPU** with CUDA 11.x or 12.x drivers
- **Python 3.9–3.12**

On Windows, use WSL2 with Ubuntu 22.04:
```bash
# Inside WSL2
sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y
python3.12 -m venv ~/venv_grover
source ~/venv_grover/bin/activate
pip install "qiskit==1.2.4" "qiskit-aer-gpu==0.15.1"
pip install numpy pandas matplotlib psutil
```

> **Note:** Use the exact pinned versions above. `qiskit>=1.3` breaks the
> `convert_to_target` import in qiskit-aer-gpu 0.15.x.

If no GPU is available, Experiment 5 will automatically run CPU-only and
skip the GPU column.

---

## Running

```bash
# 1. Run all experiments (takes ~10–30 min depending on hardware)
python experiments.py

# 2. Generate all figures
python plot_results.py

# 3. On HPC cluster (large n):
sbatch submit.sh   # see hpc_runner.py for SLURM script
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

## Sample Output (Experiment 5 — RTX 4060 Laptop, WSL2/Ubuntu 22.04)

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
For large n, the GPU advantage grows as bandwidth and parallelism dominate.

---

## Key Insight: The Memory Wall

The statevector for n qubits has 2^n complex128 entries (16 bytes each):

| n  | Memory    |
|----|-----------|
| 20 | 16 MB     |
| 25 | 512 MB    |
| 28 | 4 GB      |
| 30 | 16 GB     |
| 33 | 128 GB    |

This is *not* an algorithmic limitation — it's a consequence of quantum mechanics
requiring us to track all amplitudes simultaneously. Real quantum hardware doesn't
have this problem: it *is* the superposition, not a simulation of it.

---

## Grover's Speedup: Realistic Assessment

- Classical unstructured search: **O(N)** queries on average
- Grover: **O(√N)** queries — a quadratic, not exponential speedup
- For N = 10^9 (30 qubits): classical ≈ 500M queries → Grover ≈ 25K queries
- **Practical limits today**: decoherence limits real quantum hardware to ~10–20
  qubits for error-free Grover. The simulation is noise-free; real hardware is not.
