# Zaratan jobs

## First time

1. Log in with `ssh <uid>@login.zaratan.umd.edu` (it asks for multi-factor authentication).
2. Run `sbalance` to see our allocation, and put its name after `--account=` in both `.sbatch`
   files.
3. Set up the environment once:

   ```bash
   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
   source $HOME/miniforge3/bin/activate
   conda create -n temporec python=3.11 -y && conda activate temporec
   pip install -r requirements.txt
   ```

4. Clone the repo into scratch (faster, and much more room than home). From the repo root, run
   `mkdir -p logs/slurm` once. Slurm writes job output there and won't create the folder itself,
   so a job fails before it even starts if it's missing.
5. Run `sbatch jobs/check_gpu.sbatch` and check that the output says `cuda True` with an A100.

## Training

```bash
sbatch jobs/train.sbatch src/train_lastfm.py --model hstu --seed 0
squeue -u $USER          # what's queued or running
scancel <jobid>          # stop a job
```

Output goes to `logs/slurm/`. Before every new kind of full run, do a quick
`--sample-frac 0.1 --epochs 2` debug run first; it's much cheaper to find a typo that way.

Cost: one A100 is 48 service units (SU) per hour, so an 8-hour job can use up to 384 SU.

## When a job dies

Nothing a job finished is thrown away:

- `logs/slurm/jobs.tsv` has a line when each job starts and one when it ends: `finished`,
  `failed (exit N)`, `stopped (TERM)` (scancel or the time limit), and `time limit in 10 min`
  as a warning beforehand. Each line has the git commit and the exact command, so you can
  rerun it.
- `logs/slurm/<name>-<jobid>.out` has everything the run printed, including the traceback.
- `logs/lightning/<run>/version_*/metrics.csv` has train and validation loss for every epoch
  that finished, and `checkpoints/` next to it holds the weights after the latest epoch.

`python src/plot_curves.py` works on these partial logs too, so a run that got cut off still
gives a loss curve. Rerunning the same command starts training from scratch in a new
`version_*` folder; the old one stays where it is.
