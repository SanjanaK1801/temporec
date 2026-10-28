# TempoRec: Time-Aware Next-Track Prediction with HSTU

DATA/MSML 612 Deep Learning group project: Fatima Wasim and Sanjana Reddy.
Proposal: [docs/proposal.pdf](docs/proposal.pdf).

## What this is

Music streaming services such as Spotify recommend the next song from what a listener played
before. A listening history is a time sequence: (track, artist, timestamp) events, unevenly spaced
and often thousands of events long.

Transformers are the standard model for such sequences. SASRec applied self-attention to next-item
prediction, and TiSASRec added the time gaps between events. [HSTU](https://arxiv.org/abs/2402.17152)
(Zhai et al., ICML 2024), from Meta, changes the attention mechanism and adds attention biases for
both position and the time between events. It reported clear gains over SASRec on MovieLens and
Amazon Books, but to our knowledge it has not been evaluated on a public music listening dataset.

Music is harder than movies or shopping: people replay the same songs, listen in sessions and
follow daily routines, so *when* someone listens matters as much as what they played before. We
reproduce HSTU on MovieLens-1M, then apply it to music listening (Last.fm 1K) and adapt it as
**TempoRec**: HSTU plus an artist feature, the time of the recommendation request, and tuning for
long histories.

## Problem and research questions

Given a listener's chronologically ordered history of (track, artist, timestamp) events and the
time of the request, predict the next track they will play, as a ranked list over all tracks in the
catalog.

1. Does HSTU, adapted to music, predict the next track better than older transformers (SASRec,
   BERT4Rec) and simple baselines (most popular, recently played)?
2. Does HSTU's time-aware attention matter more for music than for movies? On MovieLens-1M,
   switching it off barely changes accuracy.
3. Do our music-specific additions (an artist feature and long-history tuning) improve on vanilla
   HSTU, and do they help with replays, new discoveries, or both?

## Data

Both datasets are public and free for research use. Raw and processed data live in `data/` and are
never committed.

| Dataset | Role | Size (approx.) | Source |
|---|---|---|---|
| Last.fm 1K | Our problem: music listening | 19M events, 992 users | [Zenodo](https://zenodo.org/records/6090214) |
| MovieLens-1M | Reproduce the HSTU paper's results | 1M ratings, 6,040 users | [GroupLens](https://grouplens.org/datasets/movielens/1m/) |

Last.fm 1K holds each user's full listening history up to June 2009. Our pipeline builds a track
ID from the lowercased `artist - title`, drops two listens with impossible dates (2010 and 2013),
drops repeat scrobbles within 30 seconds, keeps tracks with at least 5 plays and users with at
least 20 events, and splits globally by time: the last 60 days are the test set. That leaves
17.4M listens from 983 users over 528K tracks, with 798 users in the test set. (Listening thins
out in May and June 2009, so a 30-day window would leave only 162 test users.) MovieLens-1M uses
the paper's leave-one-out protocol.

## Evaluation

- **Full ranking** over every track in the catalog, no sampled negatives.
- **HR@10**: share of test users with at least one of their test tracks in the top 10 (on
  MovieLens-1M, the single held-out item).
- **NDCG@10** (primary): like HR@10, but rewards ranking the true track higher.
- **Recall@10** and **catalog coverage@10** (share of tracks that appear in at least one top-10
  list, a check against recommending only popular songs).
- **Training cost**: epochs run and minutes per run, recorded with every result.
- On Last.fm, every run reports two rows: **all** (replays count) and **discovery** (only tracks the
  user never played before).
- During training, train and validation loss are logged every epoch, training stops early when
  validation stops improving for 5 epochs, and the curves are plotted for every run.

Verification targets on MovieLens-1M (HR@10 / NDCG@10):

| Model | HSTU paper | RecTools reproduction |
|---|---|---|
| HSTU | 0.3097 / 0.1720 | 0.3226 / 0.1880 |
| HSTU-large | 0.3294 / 0.1893 | 0.3642 / 0.2164 |
| SASRec | — | 0.2877 / 0.1622 |

On Last.fm 1K, where no HSTU results are published, the target is to beat vanilla HSTU, SASRec,
BERT4Rec, most popular and recently played under identical settings, averaged over 3 seeds.

## Approach

PyTorch with PyTorch Lightning for the training loop (per-epoch logging, checkpoints, early
stopping). Models come from [RecTools](https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_HSTU_tutorial.html)
(Apache 2.0), which implements HSTU, SASRec and BERT4Rec with the same training loop, so all models
are compared fairly. Training starts from the RecTools HSTU settings: sampled-softmax loss with 128
negatives, learning rate 0.001, batch size 128, up to 100 epochs. Last.fm runs on UMD's Zaratan
cluster (one A100 per job); MovieLens-1M and data preparation run on a laptop.

| From the existing implementation | Our adaptation |
|---|---|
| HSTU trained on movie and book ratings | Applied to 19M music listening events, with our own data pipeline |
| Item IDs only | Artist added as an item feature (movie genre on MovieLens-1M) |
| Default sequence length and size | Tuned for long histories: 50 to 500 events, 2 versus 8 layers |
| Leave-one-out evaluation | Time-based split, plus separate results for replays and new songs |
| Time-aware attention evaluated on movies | Ablation testing whether time awareness matters more for music |

## Plan: interim report and final project

### Interim report (branch `interim`, due Thu Oct 29, 2026)

Goal: a working, verified pipeline end to end and first results. Tracking issue
[`[TR-0]`](https://github.com/SanjanaK1801/temporec/issues/1); one issue, branch and PR per sub-task.
The Status column is filled in automatically after every merge into `interim`.

<!-- status:start -->
| ID | Sub-task | What is delivered | Status |
|---|---|---|---|
| TR-01 | Repository scaffold and conventions | This repo, pinned environment, README and proposal | In progress ([#12](https://github.com/SanjanaK1801/temporec/pull/12), draft), 0/4 done |
| TR-02 | Zaratan environment and job scripts | Conda env on Zaratan, GPU check and training job scripts | In progress ([#13](https://github.com/SanjanaK1801/temporec/pull/13), draft), 0/3 done |
| TR-03 | Last.fm pipeline and time split | Cleaned Last.fm 1K, 60-day time split, dataset statistics and figures | In progress ([#14](https://github.com/SanjanaK1801/temporec/pull/14), draft), 0/6 done |
| TR-04 | Evaluation module and simple baselines | Shared `evaluate()` (HR, recall, NDCG, coverage at 10), most-popular and recently-played baselines, unit tests | In progress ([#15](https://github.com/SanjanaK1801/temporec/pull/15), draft), 0/3 done |
| TR-05 | Model builders and per-epoch logging | SASRec, BERT4Rec, HSTU, HSTU-large, TempoRec builders; per-epoch loss logs; early stopping | In progress ([#16](https://github.com/SanjanaK1801/temporec/pull/16), draft), 0/4 done |
| TR-06 | MovieLens-1M reproduction | HSTU and SASRec (HSTU-large if time) compared with the paper and RecTools numbers | In progress ([#17](https://github.com/SanjanaK1801/temporec/pull/17), draft), 0/4 done |
| TR-07 | Last.fm training and baselines | Popular, recent, SASRec and HSTU on Last.fm (seed 0), `all` and `discovery` results | In progress ([#18](https://github.com/SanjanaK1801/temporec/pull/18), draft), 0/4 done |
| TR-08 | TempoRec on Last.fm | First TempoRec run (sequence length 200, 500 if time) compared with HSTU | In progress ([#19](https://github.com/SanjanaK1801/temporec/pull/19), draft), 0/3 done |
| TR-09 | Epoch curves and results tables | Loss-by-epoch figures for every run and the results tables | In progress ([#20](https://github.com/SanjanaK1801/temporec/pull/20), draft), 0/3 done |
<!-- status:end -->

Interim results are **preliminary**: single seed, and some runs may use a sample of users.

### Final project (branch `final`, Nov 30 – Dec 11, 2026)

`final` is created from `interim` after the interim report and merged into `main` once, at the end.

- **BERT4Rec** on Last.fm 1K, completing the baseline set for question 1.
- **Time-attention ablation** on both MovieLens-1M and Last.fm 1K (question 2).
- **TempoRec ablations**: without the artist feature, without time-aware attention (question 3).
- **Scaling study**: sequence length 50, 200 and 500; model size 2 versus 8 layers.
- **Three training seeds** (0, 1, 2) for every reported configuration, reported as mean ± std.
- **Replays versus discoveries**: `all` and `discovery` results for every model.
- **Training cost**: time per epoch and total GPU hours per model.
- Final results tables and figures, final report and presentation.

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

Branches: `main` <- `final` <- `interim` <- `tr/*`. All work lands through PRs.

## Setup

```bash
conda create -n temporec python=3.11 -y && conda activate temporec
pip install -r requirements.txt
python -m pytest -q
```

`rectools[torch]==0.19.0` requires `numpy<2` on Python 3.11, so `requirements.txt` caps `pyarrow`
below 26 (pyarrow 26 needs NumPy 2).

## Reproduce

```bash
# data (Last.fm 1K and MovieLens-1M); without wget, use curl -L -o <file> <url>
mkdir -p data/raw && cd data/raw
wget -O lastfm-dataset-1K.tar.gz "https://zenodo.org/records/6090214/files/lastfm-dataset-1K.tar.gz?download=1"
tar -xzf lastfm-dataset-1K.tar.gz
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip && unzip ml-1m.zip
cd ../..
# prepare saves a checkpoint after each step; if it dies, rerun the same command and it picks up
# from the last finished step (--fresh starts over)
python src/prepare_lastfm.py --raw data/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv --out data/processed/lastfm
python src/split_lastfm.py --data data/processed/lastfm --test-days 60
python src/dataset_figures.py

# MovieLens-1M reproduction (compare with the HSTU paper)
for m in hstu hstu_large sasrec; do sbatch jobs/train.sbatch src/train_movielens.py --model $m --seed 0; done

# Last.fm 1K
for m in popular recent sasrec hstu temporec; do sbatch jobs/train.sbatch src/train_lastfm.py --model $m --seed 0; done

# figures and tables
python src/plot_curves.py
python src/make_tables.py > results/tables.md
```

## References

1. W.-C. Kang, J. McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018.
2. J. Li, Y. Wang, J. McAuley, *Time Interval Aware Self-Attention for Sequential Recommendation*, WSDM 2020.
3. J. Zhai et al., *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*, ICML 2024. [arXiv:2402.17152](https://arxiv.org/abs/2402.17152). Official code: meta-recsys/generative-recommenders.
4. MTS, *RecTools: Transformers HSTU tutorial*, [RecTools documentation](https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_HSTU_tutorial.html), accessed Oct 2026. HSTU, SASRec and BERT4Rec implementations and HSTU training settings (Apache 2.0), adapted to our problem.
5. O. Celma, *Music Recommendation and Discovery*, Springer 2010. Dataset: Last.fm 1K on [Zenodo](https://zenodo.org/records/6090214), distributed with permission of Last.fm.
6. F. M. Harper, J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 2015.
7. W. Krichene, S. Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020.
8. F. Sun et al., *BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer*, CIKM 2019.
