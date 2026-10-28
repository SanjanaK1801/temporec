"""Split the cleaned Last.fm 1K listens into train and test by time (TR-03, data and eval).

Usage:
    python src/split_lastfm.py --data data/processed/lastfm --test-days 30

Reads interactions.parquet (from prepare_lastfm.py) and writes train.parquet, test.parquet and
split.json to the same folder.

How the split works:
- One cutoff for everyone: the last listen in the data minus --test-days. Train is everything
  before the cutoff, test is everything from the cutoff on.
- Test listens by users or of tracks that never show up in train are dropped, because no model
  can score them.
- There is no separate validation file. The training scripts hold out each user's last training
  listen for that.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from checkpoint import write_json, write_parquet


def time_split(df: pd.DataFrame, test_days: int):
    cutoff = df["datetime"].max() - pd.Timedelta(days=test_days)
    train = df[df["datetime"] < cutoff]
    test = df[df["datetime"] >= cutoff]
    test = test[test["user_id"].isin(train["user_id"]) & test["item_id"].isin(train["item_id"])]
    return train, test, cutoff


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/lastfm")
    p.add_argument("--test-days", type=int, default=30)
    args = p.parse_args(argv)

    data = Path(args.data)
    df = pd.read_parquet(data / "interactions.parquet")
    train, test, cutoff = time_split(df, args.test_days)
    write_parquet(train, data / "train.parquet")
    write_parquet(test, data / "test.parquet")

    # Share of test listens that are tracks the user never played in train (the "discovery" part).
    seen = train[["user_id", "item_id"]].drop_duplicates()
    first_time = test.merge(seen, how="left", indicator=True)["_merge"].eq("left_only")
    summary = {
        "cutoff": str(cutoff),
        "test_days": args.test_days,
        "train_events": int(len(train)),
        "test_events": int(len(test)),
        "test_users": int(test["user_id"].nunique()),
        "test_first_time_share": float(first_time.mean()) if len(test) else 0.0,
    }
    write_json(summary, data / "split.json")
    print(json.dumps(summary, indent=2))
    if summary["test_users"] < 200:
        print("WARNING: fewer than 200 test users; consider a larger --test-days")


if __name__ == "__main__":
    main()
