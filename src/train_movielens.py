"""Reproduce HSTU / SASRec on MovieLens-1M with the HSTU paper's leave-one-out protocol (TR-06).

Usage:
    python src/train_movielens.py --model hstu --seed 0
    python src/train_movielens.py --model temporec --seed 0   # HSTU + genre feature

Published references (HR@10 / NDCG@10): HSTU paper 0.3097 / 0.1720, HSTU-large 0.3294 / 0.1893.
RecTools reproduction: HSTU 0.3226 / 0.1880, HSTU-large 0.3642 / 0.2164, SASRec 0.2877 / 0.1622.
"""
import argparse
import time

import pandas as pd
from lightning_fabric import seed_everything
from rectools import Columns
from rectools.dataset import Dataset
from rectools.dataset.context import get_context

from evaluation import evaluate, save_result
from models import TRANSFORMERS, build_model, make_trainer_func


def load_movielens(raw_dir: str):
    ratings = pd.read_csv(
        f"{raw_dir}/ratings.dat", sep="::", engine="python",
        names=[Columns.User, Columns.Item, "rating", Columns.Datetime],
    )
    ratings[Columns.Datetime] = pd.to_datetime(ratings[Columns.Datetime], unit="s")
    ratings[Columns.Weight] = 1.0
    ratings = ratings.drop(columns="rating")

    movies = pd.read_csv(f"{raw_dir}/movies.dat", sep="::", engine="python", encoding="latin-1",
                         names=["id", "title", "genres"])
    genres = movies.assign(value=movies["genres"].str.split("|")).explode("value")
    genres = genres[genres["id"].isin(ratings[Columns.Item].unique())]
    return ratings, genres.assign(feature="genre")[["id", "feature", "value"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=TRANSFORMERS)
    p.add_argument("--raw", default="data/raw/ml-1m")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--results", default="results/movielens.csv")
    args = p.parse_args()

    seed_everything(args.seed, workers=True)
    ratings, genres = load_movielens(args.raw)

    # Leave-one-out: each user's last rating is the test target. A stable sort keeps tied
    # timestamps in file order (see the RecTools tutorial's note on ties).
    ratings = ratings.sort_values([Columns.User, Columns.Datetime], kind="stable")
    is_last = ratings.groupby(Columns.User).cumcount(ascending=False) == 0
    train, test = ratings[~is_last], ratings[is_last]

    use_features = args.model == "temporec"
    dataset = Dataset.construct(train, item_features_df=genres if use_features else None,
                                cat_item_features=["genre"] if use_features else ())
    run = f"ml1m_{args.model}_s{args.seed}"
    model = build_model(args.model, make_trainer_func("logs/lightning", run, args.epochs), use_features)

    start = time.time()
    model.fit(dataset)
    minutes = round((time.time() - start) / 60, 1)

    context = get_context(test[[Columns.User, Columns.Datetime]]) if model.require_recommend_context else None
    reco = model.recommend(users=test[Columns.User].unique(), dataset=dataset, k=10,
                           filter_viewed=True, context=context)
    row = {"dataset": "ml-1m", "model": args.model, "seed": args.seed,
           "epochs_run": model.fit_trainer.current_epoch, "minutes": minutes, **evaluate(reco, test, train)}
    print(row)
    save_result(args.results, row)


if __name__ == "__main__":
    main()
