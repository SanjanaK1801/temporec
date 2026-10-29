"""Split the cleaned Last.fm 1K listens into train, validation and test by time (TR-03, data and eval).

Usage:
    python src/split_lastfm.py --data data/processed/lastfm

Reads interactions.parquet (from prepare_lastfm.py) and writes train.parquet, val.parquet,
test.parquet and split.json to the same folder.

How the split works:
- Two cutoffs for everyone, at the 80% and 90% points of all listen times (--val-frac and
  --test-frac, 0.1 each). Train is everything before the first cutoff, validation is between the
  two, test is everything from the second one on. This is how the interim report's split was
  made too; on our cleaned data the cutoffs land at 2008-10-10 17:56 and 2009-01-29 22:44.
- Validation and test only keep listens a model can score: tracks and users that show up in
  train. Users who start listening after the first cutoff are left out.
- Validation is there for early stopping, test is only looked at for the final numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from checkpoint import write_json, write_parquet


def time_split(df: pd.DataFrame, val_frac: float, test_frac: float):
    # Cut on whole seconds, the same way the interim report's split did.
    seconds = (df["datetime"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    c1, c2 = (pd.Timestamp(int(np.quantile(seconds, q)), unit="s")
              for q in (1 - val_frac - test_frac, 1 - test_frac))
    train = df[df["datetime"] < c1]
    val = df[(df["datetime"] >= c1) & (df["datetime"] < c2)]
    test = df[df["datetime"] >= c2]
    known = train["user_id"].unique(), train["item_id"].unique()
    val = val[val["user_id"].isin(known[0]) & val["item_id"].isin(known[1])]
    test = test[test["user_id"].isin(known[0]) & test["item_id"].isin(known[1])]
    return train, val, test, (c1, c2)


def first_time_share(part: pd.DataFrame, history: pd.DataFrame) -> float:
    """Share of listens in `part` that are tracks the user never played in `history` (the "discovery" part)."""
    if not len(part):
        return 0.0
    seen = history[["user_id", "item_id"]].drop_duplicates()
    return float(part.merge(seen, how="left", indicator=True)["_merge"].eq("left_only").mean())


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/lastfm")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    args = p.parse_args(argv)

    data = Path(args.data)
    df = pd.read_parquet(data / "interactions.parquet")
    train, val, test, (c1, c2) = time_split(df, args.val_frac, args.test_frac)
    write_parquet(train, data / "train.parquet")
    write_parquet(val, data / "val.parquet")
    write_parquet(test, data / "test.parquet")

    summary = {
        "val_cutoff": str(c1),
        "test_cutoff": str(c2),
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "train_events": int(len(train)),
        "val_events": int(len(val)),
        "test_events": int(len(test)),
        "train_users": int(train["user_id"].nunique()),
        "val_users": int(val["user_id"].nunique()),
        "test_users": int(test["user_id"].nunique()),
        "val_first_time_share": first_time_share(val, train),
        "test_first_time_share": first_time_share(test, pd.concat([train, val])),
    }
    write_json(summary, data / "split.json")
    print(json.dumps(summary, indent=2))
    if summary["test_users"] < 200:
        print("WARNING: fewer than 200 test users; consider a larger --test-frac")


if __name__ == "__main__":
    main()
