#!/bin/bash
#SBATCH --job-name=prepare_v2
#SBATCH --account=msml612-class
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/%x-%j.out

source $HOME/miniforge3/bin/activate temporec
cd ~/temporec
echo "commit: $(git rev-parse --short HEAD)"
D=/scratch/zt1/project/msml612/shared/next_song_prediction/data
python src/prepare_lastfm.py \
  --raw $D/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv \
  --out $D/processed_v2/lastfm \
  --min-item-plays 5 --min-user-events 20 \
  --dedupe-seconds 30 --last-date 2009-06-30