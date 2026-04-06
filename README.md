# Grover's Algorithm — Classical Simulation
**Bachelor Semester Project | University of Luxembourg**
Veselin Petrov Nikolaev · Supervisors: Marko Rancic, Maximilian Streitberger

---

## Das Ziel (The Goal)

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
├── results/              ← CSV output from experiments
└── figures/              ← PDF/PNG figures for the report
```

---

## Setup

```bash
pip install -r requirements.txt
```

For GPU support (requires CUDA + NVIDIA GPU):
```bash
pip install qiskit-aer-gpu
```

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
| 5 | GPU comparison | CPU vs GPU speedup |
| — | 2-qubit example | step-by-step numpy trace, verified with Qiskit |

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
- **Practical limits today**: decoherence limits real quantum hardware to ~10–20 qubits
  for error-free Grover. The simulation is noise-free; real hardware is not.
