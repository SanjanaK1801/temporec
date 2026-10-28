#!/bin/bash
# Run one training script on one A100. From the repo root:
#
#   sbatch jobs/train.sbatch src/train_lastfm.py --model hstu --seed 0
#
# Everything after the script name goes straight to python.
#
# If a job dies (time limit, scancel, a Python error) we still keep what it got through:
#   logs/slurm/<name>-<jobid>.out    everything the run printed
#   logs/lightning/<run>/version_*/  metrics.csv with train/val loss for every finished epoch,
#                                    and checkpoints/ with the weights after the latest epoch
#   logs/slurm/jobs.tsv              one line when a job starts and one when it ends (or gets
#                                    stopped), with the git commit and the exact command
#SBATCH --job-name=temporec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=<allocation-name>
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --signal=B:USR1@600

# Activate first: conda's activate script doesn't like `set -u`.
source "$HOME/miniforge3/bin/activate" temporec || { echo "could not activate the temporec env"; exit 1; }
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/slurm results

CMD="python $*"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
JOBS=logs/slurm/jobs.tsv
[ -f "$JOBS" ] || printf 'job_id\ttime\tevent\tcommit\tcommand\n' > "$JOBS"
note() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$SLURM_JOB_ID" "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$COMMIT" "$CMD" >> "$JOBS"
}

echo "job $SLURM_JOB_ID on $(hostname), commit $COMMIT: $CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv
note started

# Python runs in the background so the traps below fire straight away, not after it exits.
python "$@" &
PY=$!

# --signal above makes Slurm send USR1 ten minutes before the time limit. We can't stop the
# clock, but we can write down that this run is about to be cut short.
trap 'note "time limit in 10 min"' USR1
# scancel and the time limit itself send TERM: pass it on to python and write it down.
trap 'note "stopped (TERM)"; kill -TERM "$PY" 2>/dev/null' TERM

# `wait` also returns whenever a trap fires, so keep waiting until python is really gone,
# then ask once more for its exit code.
while kill -0 "$PY" 2>/dev/null; do wait "$PY"; done
wait "$PY"
status=$?

if [ "$status" -eq 0 ]; then note finished; else note "failed (exit $status)"; fi
exit "$status"
