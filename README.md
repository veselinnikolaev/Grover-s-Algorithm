# Grover's Algorithm — Classical Simulation and Tensor Network Analysis

Bachelor Semester Project | University of Luxembourg
Veselin Petrov Nikolaev · Supervisors: Marko Rančić, Maximilian Streitberger

## The Goal

### Part 1: Statevector Simulation (BSP2, completed)

Classically simulate Grover's algorithm to understand its mathematical/geometric
structure, measure where classical simulation breaks down (memory wall), compare
CPU vs GPU performance on HPC infrastructure, and evaluate the realistic impact
of the quadratic speedup.

**Result:** statevector simulation reached n≈31–32 qubits on GPU hardware before
hitting the memory wall (16·2^n bytes).

### Part 2: Tensor Network Simulation (current project)

Investigate whether tensor-network methods can push past the statevector memory
wall, using `quimb` + `cotengra` (+ optional `kahypar`), in place of the
originally proposed NVIDIA cuTensorNet backend. The backend was changed because
the research questions concern tensor-network *methods* (width, cost, bond
dimension) rather than a specific hardware-accelerated implementation — see the
report for the full rationale.

**Key finding:** tensor networks do **not** achieve a sustained memory advantage
for Grover's algorithm in the tested range (n=2–11). The required bond dimension
for exact MPS convergence grows from χ=4 at n=5 to χ=32 at n=11, and the
exact-contraction memory estimate exceeds statevector memory by roughly 6–7
orders of magnitude by n=11 — Grover's oracle generates entanglement that
defeats tensor-network compression, rather than benefiting from it.

## File Structure

```
grover-s-algorithm/
│
├── STATEVECTOR SIMULATION (BSP2)
│   ├── grover_core.py              ← oracle, diffuser, circuit builder, simulation runner
│   ├── experiments.py              ← 5 statevector experiments + 2-qubit hand example
│   ├── hpc_runner.py                ← SLURM-ready runner for n > 22 on HPC
│
├── TENSOR NETWORK SIMULATION (current)
│   ├── tn_metrics.py                ← W/C rehearsal, real-contraction memory measurement
│   ├── tn_experiments.py            ← main TN scalability sweep + real-memory validation
│   ├── tn_depth_experiment.py       ← contraction cost vs circuit depth (fixed n, varying iterations)
│   ├── tn_bond_dimension_experiment.py  ← MPS bond-dimension threshold sweep
│
├── SHARED
│   ├── plot_results.py             ← generates ALL figures (1–10) from results/*.csv
│   ├── requirements.txt
│   ├── README.md
│
├── report/
│   ├── report_english.tex          ← BSP2 statevector report (Figures 1–6)
│   └── tensor_network_report.tex   ← current TN report (Figures 7–10)
│
├── hpc/
│   └── jobs/
│       ├── submit_cpu.sh           ← SLURM, AION (CPU)
│       └── submit_gpu.sh           ← SLURM, IRIS (GPU)
│
├── results/
│   ├── scalability.csv, iteration_sweep.csv, classical_comparison.csv,
│   │   circuit_depth.csv, gpu_comparison.csv,
│   │   hpc_scalability_cpu.csv, hpc_scalability_gpu.csv     ← BSP2 (Figs 1–6)
│   │
│   ├── tn_scalability.csv                        ← statevector vs TN memory (Fig 7)
│   ├── tn_real_memory_validation.csv             ← rehearsed vs real memory (Fig 10)
│   ├── tn_depth_scaling.csv                       ← cost vs circuit depth (unplotted)
│   ├── tn_bond_dimension_threshold.csv           ← min χ per n, worst/best case (Fig 8)
│   ├── tn_bond_dimension_threshold_by_target.csv ← every individual (n, target) trial
│   └── tn_bond_dimension_scaling_n{n}_t{target}.csv  ← per-target detail (Fig 9 source)
│
└── figures/
    ├── png/ and pdf/
    │   ├── fig1_scalability, fig2_iteration_sweep, fig3_speedup,
    │   │   fig4_circuit_depth, fig5_gpu, fig6_hpc            ← BSP2
    │   └── fig7_tn_crossover, fig8_bond_threshold,
    │       fig9_bond_convergence, fig10_rehearsed_vs_real     ← current TN work
```

## Requirements

Python 3.9–3.12 (Python 3.13+ not supported by `qiskit-aer-gpu`). Tested on 3.12.3.

```bash
pip install qiskit qiskit-aer numpy pandas matplotlib psutil       # statevector side
pip install quimb cotengra qiskit-quimb psutil                     # TN side
pip install kahypar                                                 # optional, better contraction trees
```

`kahypar` sometimes fails to build on some systems — this is fine, `cotengra` falls
back to a `greedy` method automatically.

### GPU support (statevector side only)

Requires Linux (x86_64), NVIDIA GPU with CUDA 11.x/12.x, Python 3.9–3.12.

```bash
pip install "qiskit==1.3.0" "qiskit-aer-gpu==0.15.0" --no-deps
pip install custatevec-cu12 cutensornet-cu12 cutensor-cu12 --no-deps
```

The tensor-network side of this project currently runs CPU-only via `quimb`/`cotengra`;
no GPU-accelerated tensor-network backend (e.g. cuTensorNet) has been integrated.

## Running

```bash
# Statevector experiments (BSP2)
python experiments.py

# Tensor network experiments (current)
python tn_experiments.py                    # Figures 7 & 10 data
python tn_bond_dimension_experiment.py      # Figures 8 & 9 data
python tn_depth_experiment.py               # depth-scaling data (currently unplotted)

# All figures (1-10) from whatever results/*.csv exist
python plot_results.py

# HPC (statevector only, n > 22)
sbatch hpc/jobs/submit_cpu.sh   # AION
sbatch hpc/jobs/submit_gpu.sh   # IRIS
```

**Note on run time:** the tensor-network sweep is not fast — `tn_experiments.py`
runs each n up to n=9 with 3 independent search repeats (for search-result
stability, see below), and single runs for n=10/11; expect this to take on the
order of an hour depending on hardware. `tn_bond_dimension_experiment.py` tests
multiple targets per n and can also take a while.

## Known Limitations (read before citing exact numbers)

- **`cotengra`'s search is stochastic with no fixed seed.** Even with a fixed
  target circuit, repeated searches can find contraction trees of noticeably
  different quality, especially past n≈9. We mitigate this by taking the best
  of 3 independent searches for n≤9, but n=10/11 in `tn_scalability.csv` are
  single-run and should be read as order-of-magnitude only (flagged with hollow
  markers in Figure 7).
- **The `16·2^W` formula is a lower bound on memory, not a prediction.** Figure
  10 shows real measured memory exceeding this estimate by 1–3 orders of
  magnitude, since W only captures the single largest intermediate tensor, not
  total memory across the whole contraction plus library overhead.
- **MPS-based bond-dimension experiments struggle with n≳12** due to the
  non-adjacent-qubit gates produced by the `mcx` decomposition (MPS is only
  efficient for local, nearest-neighbor gate structure) — this caused stalls
  and NaN/overflow edge cases at n=12 during development, which is why the
  bond-dimension sweep is capped at n=11.
- **Real-contraction memory validation is capped at n=9**: at n=10, the
  rehearsed contraction width implied roughly 18 petabytes, so real execution
  there is not attempted (a safety check now skips real execution automatically
  if the rehearsed width implies more than 2 GB).

## Reports

- `report/report_english.tex` — BSP2 statevector report (Figures 1–6):
  establishes the statevector memory wall at n≈31–32.
- `report/tensor_network_report.tex` — current tensor-network report (Figures
  7–10): answers the three TN research questions, explains the backend change
  from cuTensorNet to quimb/cotengra/kahypar, and is explicit about the search
  instability and estimate-vs-real-memory gaps described above.

Compile either with `pdflatex <file>.tex`.
