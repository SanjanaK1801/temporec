# Zaratan jobs

1. Log in: `ssh <uid>@login.zaratan.umd.edu` (multi-factor authentication).
2. Check allocations: `sbalance`. Put the allocation name in both `.sbatch` files (`--account`).
3. One-time environment:

   ```bash
   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
   source $HOME/miniforge3/bin/activate
   conda create -n temporec python=3.11 -y && conda activate temporec
   pip install -r requirements.txt
   ```

4. Clone the repo into your scratch space (fast storage, more room than home), then from its root:
   `sbatch jobs/check_gpu.sbatch` and confirm the output says `cuda True`.
5. Train: `sbatch jobs/train.sbatch src/train_lastfm.py --model hstu --seed 0`.
   Watch with `squeue -u $USER`; output in `logs/slurm/`. Cancel with `scancel <jobid>`.

Cost: one A100 is 48 service units (SU) per hour, so an 8-hour job uses up to 384 SU.
Run a `--sample-frac 0.1 --epochs 2` debug run before every new full job.
