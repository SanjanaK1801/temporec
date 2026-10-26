#!/bin/bash
# Run one training script on one A100.
# Usage (from the repo root): sbatch jobs/train.sbatch src/train_lastfm.py --model hstu --seed 0
#SBATCH --job-name=temporec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=<allocation-name>
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
source "$HOME/miniforge3/bin/activate" temporec
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/slurm results
echo "job $SLURM_JOB_ID on $(hostname): python $*"
nvidia-smi --query-gpu=name,memory.total --format=csv
python "$@"
