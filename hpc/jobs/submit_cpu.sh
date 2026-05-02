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
