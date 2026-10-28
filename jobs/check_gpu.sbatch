#!/bin/bash
# Quick check (a couple of minutes) that the env loads and PyTorch can see the GPU.
#
#   sbatch jobs/check_gpu.sbatch
#   cat logs/slurm/check_gpu-<jobid>.out     # look for "cuda True" and an A100
#SBATCH --job-name=check_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --account=<allocation-name>
#SBATCH --output=logs/slurm/%x-%j.out

source "$HOME/miniforge3/bin/activate" temporec || { echo "could not activate the temporec env"; exit 1; }
set -uo pipefail
nvidia-smi
python -c "import torch, rectools; print('torch', torch.__version__, 'rectools', rectools.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
