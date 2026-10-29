#!/bin/bash
#SBATCH --job-name=clean_lastfm
#SBATCH --account=msml612-class
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

module load python/3.10.10/gcc/11.3.0/nocuda
source ~/temporec/venv_data/bin/activate
cd ~/temporec
python src/clean_lastfm.py --tag full