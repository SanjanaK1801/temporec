#!/bin/bash
#SBATCH --job-name=split_v2
#SBATCH --account=msml612-class
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x-%j.out

source $HOME/miniforge3/bin/activate temporec
cd ~/temporec
echo "commit: $(git rev-parse --short HEAD)"
D=/scratch/zt1/project/msml612/shared/next_song_prediction/data
python src/split_lastfm.py --data $D/processed_v2/lastfm --val-frac 0.1 --test-frac 0.1