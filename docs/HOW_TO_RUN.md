# How to run

Everything below starts from a fresh clone. Commands were run on Zaratan (Linux, A100 GPU).

## 1. Install

```bash
git clone https://github.com/SanjanaK1801/temporec.git && cd temporec
git checkout interim
conda create -n temporec python=3.11 -y && conda activate temporec
pip install -r requirements.txt "pyarrow<26"
python -m pytest -q
```

`pyarrow<26` is required: pyarrow 26 needs NumPy 2, and RecTools needs NumPy below 2.

## 2. Get the data (not included in the repo)

Last.fm 1K (Celma, 2010; Zenodo DOI 10.5281/zenodo.6090214) is distributed with permission of Last.fm for non-commercial use, so we do not redistribute it. Please reference Last.fm when using it. MovieLens-1M is from GroupLens.

```bash
bash scripts/download_data.sh
```

The script downloads both datasets into `data/raw`, and checks the Last.fm archive against the published MD5 (`a79a6808f54f73354789a9fb02cb1e41`). It needs `wget`, `tar`, `unzip` and `md5sum`.

## 3. Build the Last.fm data

```bash
RAW=data/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv
OUT=data/processed_v2/lastfm
python src/prepare_lastfm.py --raw $RAW --out $OUT \
  --min-item-plays 5 --min-user-events 20 --dedupe-seconds 30 --last-date 2009-06-30
python src/split_lastfm.py --data $OUT --val-frac 0.1 --test-frac 0.1
export LASTFM_DATA=$OUT
```

Expected output: 17,406,337 events, 983 users, 528,287 tracks after cleaning; train 13,925,068, validation 1,513,590, test 1,269,816 events after the split.

## 4. Baselines and models

```bash
python src/next_track_baselines.py --per-user 100 --seed 0
python src/train_lastfm.py --model sasrec --seed 0 --epochs 40 --patience 10
python src/train_lastfm.py --model hstu --seed 0 --epochs 40 --patience 10
```

Models: `sasrec`, `bert4rec`, `hstu`, `hstu_large`, `temporec`. Add `--no-time-attention` for the ablation. Results go to `results/lastfm.csv`.

Quick check without a GPU (tiny sample, one epoch):

```bash
TEMPOREC_CPU=1 python src/train_lastfm.py --model sasrec --sample-frac 0.02 --epochs 1 --results results/debug.csv
```

## 5. Evaluation protocol

Each of up to 100 test listens per user (seeded) is scored on its own: the model ranks the whole catalog using that user's earlier listens (train, validation and earlier test), with the listen's timestamp as the request time. Metrics: HR@10, NDCG@10, catalog coverage, split into replays and new tracks.

## 6. Constraints

- Full training needs a GPU. We used one NVIDIA A100 (40 GB) on Zaratan.
- Cleaning the full file took about 3 minutes and peaked at about 10 GB of RAM (measured with Slurm's MaxRSS); the split and the data loader peaked at about 4 GB. Training memory depends on the model and is reported with the run results.
- Data: 19.15M listens (2.4 GB file). Housekeeping files and checkpoints go in `data/` and `logs/`, which Git ignores.
- Compute is a shared course allocation (A100 = 48 service units per hour), so the experiment grid is small: one setting, seed 0.
- Limits of the method: about 76% of sampled test listens are replays, 4% of events are dropped because their track or user is unseen in training, and ID-based models cannot score new tracks.