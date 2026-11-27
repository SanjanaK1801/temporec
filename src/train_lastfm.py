"""Train one model on Last.fm and score it with the sampled next-track protocol (TR-07).

    python src/train_lastfm.py --model sasrec --seed 0
    python src/train_lastfm.py --model hstu --seed 0 --sample-frac 0.1 --epochs 2   # debug run

Targets are always sampled with --target-seed (default 0), so every model is scored on the
same 62,733 test listens as the baselines, whatever the training seed.
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
from next_track_eval import build_mini_inputs, score_reco


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

    use_features = args.model == "temporec"
    feats = None
    if use_features:
        feats = pd.read_parquet(f"{args.data}/item_features.parquet")
        feats = feats[feats["id"].isin(train[Columns.Item].unique())]
    train = chunk_histories(train, args.chunk_len)
    dataset = Dataset.construct(train, item_features_df=feats,
                                cat_item_features=["artist"] if use_features else ())

    name = args.model + ("_notime" if args.no_time_attention else "")
    run = f"lastfm_{name}_s{args.seed}"
    # filter_viewed=False: 76% of listens are replays, so replays must stay hittable
    trainer = make_trainer_func("logs/lightning", run, args.epochs, args.patience, filter_viewed=False)
    overrides = {"session_max_len": args.chunk_len}
    if args.no_time_attention:
        overrides["relative_time_attention"] = False
    model = build_model(args.model, trainer, use_features, **overrides)

    start = time.time()
    model.fit(dataset)
    minutes = round((time.time() - start) / 60, 1)
    best_epoch = load_best_weights(model)
    epochs_run = model.fit_trainer.current_epoch

    # One mini-user per target: history before the target, request time = target's timestamp.
    inter, context = build_mini_inputs(seq, targets, hist_len=args.chunk_len)
    mini = Dataset.construct(inter)
    reco = model.recommend(users=context[Columns.User].to_numpy(), dataset=mini, k=10,
                           filter_viewed=False,
                           context=context if model.require_recommend_context else None)
    truth = seq["catalog"][seq["item"][targets]]
    res = score_reco(reco, truth, seq["replay"][targets], seq["n_items"])

    row = {"dataset": "lastfm", "protocol": f"next_track_{args.per_user}pu", "model": name,
           "seed": args.seed, "sample_frac": args.sample_frac, "targets": len(targets),
           "max_epochs": args.epochs, "epochs_run": epochs_run, "best_epoch": best_epoch,
           "patience": args.patience, "minutes": minutes,
           "min_per_epoch": round(minutes / max(epochs_run, 1), 2),
           "hr@10": res["all"]["HR@10"], "ndcg@10": res["all"]["NDCG@10"],
           "hr@10_replay": res["replay"]["HR@10"], "ndcg@10_replay": res["replay"]["NDCG@10"],
           "hr@10_new": res["new"]["HR@10"], "ndcg@10_new": res["new"]["NDCG@10"],
           "coverage@10": res["coverage"]}
    print(row)
    save_result(args.results, row)


if __name__ == "__main__":
    main()