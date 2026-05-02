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