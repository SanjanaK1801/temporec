"""Train and evaluate one model on Last.fm 1K (TR-07, models and training).

Reads the split written by split_lastfm.py (train.parquet, test.parquet). Reports two rows:
    all        every test play counts, replays included
    discovery  only tracks the user never played in train

Usage:
    python src/train_lastfm.py --model hstu --seed 0
    python src/train_lastfm.py --model hstu --sample-frac 0.1 --epochs 2 --results results/debug.csv
"""
import argparse
import time

import pandas as pd
from lightning_fabric import seed_everything
from rectools import Columns
from rectools.dataset import Dataset
from rectools.dataset.context import get_context

from baselines import most_popular, recently_played
from evaluation import drop_seen, evaluate, first_time_listens, save_result
from models import TRANSFORMERS, build_model, chunk_histories, make_trainer_func

DATA_SEED = 42  # fixed: which users go into a --sample-frac subset


def load_split(data_dir: str, sample_frac: float):
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    if sample_frac < 1:
        users = train[Columns.User].drop_duplicates().sample(frac=sample_frac, random_state=DATA_SEED)
        train, test = train[train[Columns.User].isin(users)], test[test[Columns.User].isin(users)]
        test = test[test[Columns.Item].isin(train[Columns.Item])]
    return train, test


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["popular", "recent", "sasrec", "bert4rec", "hstu", "hstu_large"])
    p.add_argument("--data", default="data/processed/lastfm")
    p.add_argument("--max-len", type=int, default=200, help="sequence length = chunk length")
    p.add_argument("--sample-frac", type=float, default=1.0, help="fraction of users, for debugging")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="training seed: 0, 1 or 2")
    p.add_argument("--results", default="results/lastfm.csv")
    return p.parse_args()


def fit_transformer(args, train, overrides=None, item_features=None):
    """Chunk histories, build the dataset and model, train. Returns (model, dataset)."""
    train_seq = chunk_histories(train, args.max_len)
    use_features = item_features is not None
    dataset = Dataset.construct(
        train_seq,
        item_features_df=item_features,
        cat_item_features=list(item_features["feature"].unique()) if use_features else (),
    )
    run = f"lastfm_{args.model}_L{args.max_len}_s{args.seed}"
    trainer_func = make_trainer_func("logs/lightning", run, args.epochs)
    model = build_model(args.model, trainer_func, use_features,
                        session_max_len=args.max_len, **(overrides or {}))
    model.fit(dataset)
    return model, dataset


def recommend(model, dataset, test, users, k=500):
    """Rank tracks for each test user. HSTU needs the request time: the user's first test play."""
    context = None
    if getattr(model, "require_recommend_context", False):
        first = test.sort_values(Columns.Datetime, kind="stable").drop_duplicates(Columns.User)
        context = get_context(first[[Columns.User, Columns.Datetime]])
    return model.recommend(users=users, dataset=dataset, k=k, filter_viewed=False, context=context)


def report(args, base: dict, reco_long, test, train) -> None:
    """Evaluate the 'all' and 'discovery' rankings and append both rows to the results CSV."""
    reco_all = reco_long[reco_long[Columns.Rank] <= 10]
    rows = [{**base, "eval": "all", **evaluate(reco_all, test, train)}]
    if args.model != "recent":  # 'recent' only recommends replays, so it has no discovery result
        reco_new = drop_seen(reco_long, train, k=10)
        rows.append({**base, "eval": "discovery", **evaluate(reco_new, first_time_listens(test, train), train)})
    for row in rows:
        print(row)
        save_result(args.results, row)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, workers=True)
    train, test = load_split(args.data, args.sample_frac)
    users = test[Columns.User].unique()
    print(f"{len(train):,} train events, {len(test):,} test events, {len(users):,} test users")

    start, epochs_run = time.time(), None
    if args.model == "popular":
        reco_long = most_popular(train, users, k=500)
    elif args.model == "recent":
        reco_long = recently_played(train, users, k=10)
    else:
        model, dataset = fit_transformer(args, train)
        epochs_run = model.fit_trainer.current_epoch
        reco_long = recommend(model, dataset, test, users)

    base = {"model": args.model, "max_len": args.max_len, "seed": args.seed, "sample_frac": args.sample_frac,
            "epochs_run": epochs_run, "minutes": round((time.time() - start) / 60, 1)}
    report(args, base, reco_long, test, train)


if __name__ == "__main__":
    main()
