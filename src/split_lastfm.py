"""Time-based train/test split of the cleaned Last.fm 1K interactions (TR-03, data and eval).

Usage:
    python src/split_lastfm.py --data data/processed/lastfm --test-days 30

Reads  interactions.parquet (from prepare_lastfm.py)
Writes train.parquet, test.parquet, split.json in the same folder.

Rules:
- One global cutoff = last event time minus --test-days. Train = before cutoff, test = at or after.
- Test rows whose user or track never appears in train are dropped (models cannot score them).
- Validation is not a separate file: the training scripts hold out each user's last training item.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def time_split(df: pd.DataFrame, test_days: int):
    cutoff = df["datetime"].max() - pd.Timedelta(days=test_days)
    train = df[df["datetime"] < cutoff]
    test = df[df["datetime"] >= cutoff]
    test = test[test["user_id"].isin(train["user_id"]) & test["item_id"].isin(train["item_id"])]
    return train, test, cutoff


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/lastfm")
    p.add_argument("--test-days", type=int, default=30)
    args = p.parse_args()

    data = Path(args.data)
    df = pd.read_parquet(data / "interactions.parquet")
    train, test, cutoff = time_split(df, args.test_days)
    train.to_parquet(data / "train.parquet", index=False)
    test.to_parquet(data / "test.parquet", index=False)

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
    (data / "split.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["test_users"] < 200:
        print("WARNING: fewer than 200 test users; consider a larger --test-days")


if __name__ == "__main__":
    main()
