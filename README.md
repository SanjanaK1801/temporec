# TempoRec: Time-Aware Next-Track Prediction with HSTU

DATA/MSML 612 Deep Learning group project: [Member A name], [Member B name].
Proposal: [docs/proposal.pdf](docs/proposal.pdf).

## What this is

Music streaming apps predict the next song from what a listener played before. A listening history
is a time sequence: (track, artist, timestamp) events, unevenly spaced and often thousands long.

[HSTU](https://arxiv.org/abs/2402.17152) (Zhai et al., ICML 2024) is a transformer recommender from
Meta whose attention uses the time between events. It was evaluated on movies and books. We
reproduce it on MovieLens-1M, then apply it to music listening (Last.fm 1K) and adapt it as
**TempoRec**: HSTU plus an artist feature, the time of the recommendation request, and tuning for
long histories. We ask whether time awareness matters more for music than for movies.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 1. Interim report | Branch `interim`. Repo and environment, Last.fm pipeline, evaluation, MovieLens-1M reproduction, Last.fm baselines, first TempoRec run. Tracking issue `[TR-0]`. | In progress |
| 2. Final report | Branch `final`, merged into `main` once at the end. BERT4Rec, ablations, sequence-length and model-size scaling, 3 training seeds, final report and presentation. | Planned |

## Repository layout

```text
docs/        proposal and reports
jobs/        Slurm scripts for the Zaratan cluster
src/         all code (run from the repo root: python src/<script>.py)
tests/       pytest unit tests (python -m pytest)
data/        datasets, never committed
logs/        Slurm output and per-epoch training logs, never committed
results/     one CSV row per run, plus generated tables
figures/     plots for the reports
```

## Setup

```bash
conda create -n temporec python=3.11 -y && conda activate temporec
pip install -r requirements.txt
python -m pytest -q
```

## Reproduce

```bash
# data (Last.fm 1K and MovieLens-1M)
mkdir -p data/raw && cd data/raw
wget -O lastfm-dataset-1K.tar.gz "https://zenodo.org/records/6090214/files/lastfm-dataset-1K.tar.gz?download=1"
tar -xzf lastfm-dataset-1K.tar.gz
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip && unzip ml-1m.zip
cd ../..
python src/prepare_lastfm.py --raw data/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv --out data/processed/lastfm
python src/split_lastfm.py --data data/processed/lastfm --test-days 30
python src/dataset_figures.py

# MovieLens-1M reproduction (compare with the HSTU paper)
for m in hstu hstu_large sasrec; do sbatch jobs/train.sbatch src/train_movielens.py --model $m --seed 0; done

# Last.fm 1K
for m in popular recent sasrec hstu temporec; do sbatch jobs/train.sbatch src/train_lastfm.py --model $m --seed 0; done

# figures and tables
python src/plot_curves.py
python src/make_tables.py > results/tables.md
```

## Citation and credits

- J. Zhai et al., *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*, ICML 2024. [arXiv:2402.17152](https://arxiv.org/abs/2402.17152).
- HSTU, SASRec and BERT4Rec implementations and HSTU training settings from [RecTools](https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_HSTU_tutorial.html) (Apache 2.0), adapted to our problem.
- Last.fm 1K: O. Celma, *Music Recommendation and Discovery*, Springer 2010, [Zenodo](https://zenodo.org/records/6090214); distributed with permission of Last.fm.
- MovieLens-1M: F. M. Harper and J. A. Konstan, ACM TiiS 2015.
