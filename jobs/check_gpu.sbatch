#!/bin/bash
# Two-minute sanity check: environment loads and PyTorch sees the GPU.
# Usage: sbatch jobs/check_gpu.sbatch   then   cat logs/slurm/check_gpu-<jobid>.out
#SBATCH --job-name=check_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --account=<allocation-name>
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
source "$HOME/miniforge3/bin/activate" temporec
nvidia-smi
python -c "import torch, rectools; print('torch', torch.__version__, 'rectools', rectools.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
