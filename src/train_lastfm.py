"""Train one model on Last.fm and score it with the sampled next-track protocol (TR-07).

    python src/train_lastfm.py --model sasrec --seed 0
    python src/train_lastfm.py --model hstu --seed 0 --sample-frac 0.1 --epochs 2   # debug run

Targets are always sampled with --target-seed (default 0), so every model is scored on the
same 62,733 test listens as the baselines, whatever the training seed.

TempoRec's own components (TR-08) are flags on any HSTU model, so the ablation ladder is:
    python src/train_lastfm.py --model hstu --seed 0
    python src/train_lastfm.py --model hstu --artist --seed 0
    python src/train_lastfm.py --model hstu --artist --time-embeddings --seed 0
    python src/train_lastfm.py --model hstu --artist --repeat-gate --seed 0
    python src/train_lastfm.py --model hstu --artist --time-embeddings --repeat-gate --seed 0
--window-negatives is experimental: without a correction for the non-uniform sampling it pushes the
gate towards 0 (debug run on 10% of users), so it stays off in the ladder for now.
Every model is scored with next_track.evaluate (temporec.score_next), which includes the gate;
for models without it the top 10s are the same as with recommend() (see next_track.py).
"""
import argparse
import time

import numpy as np
import pandas as pd
from lightning_fabric import seed_everything
from rectools import Columns
from rectools.dataset import Dataset

from evaluation import save_result
from lastfm_data import DATA, load_sequences, sample_test_targets
from models import TRANSFORMERS, build_model, chunk_histories, load_best_weights, make_trainer_func
from next_track import evaluate, model_scorer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=TRANSFORMERS)
    p.add_argument("--data", default=DATA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--chunk-len", type=int, default=200)
    p.add_argument("--per-user", type=int, default=100)
    p.add_argument("--sample-frac", type=float, default=1.0, help="use only this share of users (debug)")
    p.add_argument("--no-time-attention", action="store_true")
    p.add_argument("--artist", action="store_true", help="artist as an item feature (always on for temporec)")
    p.add_argument("--time-embeddings", action="store_true", help="TR-08: hour-of-day and day-of-week embeddings")
    p.add_argument("--repeat-gate", action="store_true", help="TR-08: repeat-or-explore gate")
    p.add_argument("--gate-window", type=int, default=50, help="recent plays the gate looks at (20 on a laptop)")
    p.add_argument("--window-negatives", type=int, default=0,
                   help="negatives drawn from the listener's recent plays, so the gate learns to switch off")
    p.add_argument("--results", default="results/lastfm.csv")
    args = p.parse_args()
    seed_everything(args.seed, workers=True)

    seq = load_sequences(args.data)
    train = pd.read_parquet(f"{args.data}/train.parquet")
    targets = sample_test_targets(seq, args.per_user, args.target_seed)

    if args.sample_frac < 1.0:  # debug: a fixed random share of users
        rng = np.random.default_rng(42)
        users = np.array(sorted(train[Columns.User].unique()))
        keep = rng.choice(users, max(2, int(len(users) * args.sample_frac)), replace=False)
        train = train[train[Columns.User].isin(keep)]
        target_users = seq["users"][np.searchsorted(seq["starts"], targets, side="right") - 1]
        targets = targets[np.isin(target_users, keep)]

    use_features = args.model == "temporec" or args.artist
    feats = None
    if use_features:
        feats = pd.read_parquet(f"{args.data}/item_features.parquet")
        feats = feats[feats["id"].isin(train[Columns.Item].unique())]
    train = chunk_histories(train, args.chunk_len)
    dataset = Dataset.construct(train, item_features_df=feats,
                                cat_item_features=["artist"] if use_features else ())

    name = args.model + "".join(f"+{part}" for part, on in (
        ("artist", args.artist and args.model != "temporec"), ("time", args.time_embeddings),
        ("gate", args.repeat_gate), (f"wneg{args.window_negatives}", args.window_negatives > 0)) if on)
    name += "_notime" if args.no_time_attention else ""
    run = f"lastfm_{name}_s{args.seed}"
    # filter_viewed=False: 76% of listens are replays, so replays must stay hittable
    trainer = make_trainer_func("logs/lightning", run, args.epochs, args.patience, filter_viewed=False)
    overrides = {"session_max_len": args.chunk_len}
    if args.no_time_attention:
        overrides["relative_time_attention"] = False
    model = build_model(args.model, trainer, use_features, time_embeddings=args.time_embeddings,
                        repeat_gate=args.repeat_gate, gate_window=args.gate_window,
                        window_negatives=args.window_negatives, **overrides)

    start = time.time()
    model.fit(dataset)
    minutes = round((time.time() - start) / 60, 1)
    best_epoch = load_best_weights(model)
    epochs_run = model.fit_trainer.current_epoch

    # Each target is ranked from the history just before it; the request time is its timestamp.
    score_fn, to_ids, n_skip = model_scorer(model, seq["catalog"])
    res, _ = evaluate(score_fn, seq, targets, to_ids, args.chunk_len, n_skip)
    backbone = model.lightning_model.torch_model

    row = {"dataset": "lastfm", "protocol": f"next_track_{args.per_user}pu", "model": name,
           "seed": args.seed, "sample_frac": args.sample_frac, "targets": len(targets), "max_len": args.chunk_len,
           "max_epochs": args.epochs, "epochs_run": epochs_run, "best_epoch": best_epoch,
           "patience": args.patience, "minutes": minutes,
           "min_per_epoch": round(minutes / max(epochs_run, 1), 2),
           "hr@10": res["all_hr@10"], "ndcg@10": res["all_ndcg@10"],
           "hr@10_replay": res["replay_hr@10"], "ndcg@10_replay": res["replay_ndcg@10"],
           "hr@10_new": res["new_hr@10"], "ndcg@10_new": res["new_ndcg@10"],
           "coverage@10": res["coverage@10"], "unscorable": res["unscorable"]}
    if args.repeat_gate:
        row.update({"gate_window": args.gate_window, "window_negatives": args.window_negatives,
                    "gate_replay": res["replay_gate"], "gate_new": res["new_gate"],
                    "gate_decay": float(backbone.decay), "gate_scale": float(backbone.log_scale.exp())})
    print(row)
    save_result(args.results, row)


if __name__ == "__main__":
    main()