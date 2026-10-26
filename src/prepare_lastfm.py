"""Clean, deduplicate and filter the Last.fm 1K dataset (TR-03, data and eval).

Usage:
    python src/prepare_lastfm.py --raw data/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv \
        --out data/processed/lastfm

Writes:
    interactions.parquet   cleaned events: user_id, item_id, datetime, weight
    item_features.parquet  artist of each track, in RecTools' long format (id, feature, value)
    stats.json             dataset statistics after each step (for the report)
"""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd

RAW_COLUMNS = ["user", "timestamp", "artist_mbid", "artist_name", "track_mbid", "track_name"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        quoting=csv.QUOTE_NONE,  # track names contain stray quote characters
        on_bad_lines="skip",
        dtype=str,
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    return df.drop(columns="timestamp")


def stats(df: pd.DataFrame, step: str) -> dict:
    return {
        "step": step,
        "events": int(len(df)),
        "users": int(df["user_id"].nunique()) if "user_id" in df else int(df["user"].nunique()),
        "items": int(df["item_id"].nunique()) if "item_id" in df else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-item-plays", type=int, default=5)
    p.add_argument("--min-user-events", type=int, default=20)
    p.add_argument("--dedupe-seconds", type=int, default=30)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    df = load_raw(args.raw)
    log.append({"step": "raw", "events": int(len(df)), "users": int(df["user"].nunique()), "items": None})

    # 1. Clean: need a timestamp, an artist and a track name.
    df = df.dropna(subset=["datetime", "artist_name", "track_name"])
    # Track ID = lowercased, stripped "artist - title" for every row. Many rows lack a MusicBrainz ID,
    # so mixing MBIDs and text IDs would give the same song two different IDs.
    artist = df["artist_name"].str.strip().str.lower()
    df["item_id"] = artist + " - " + df["track_name"].str.strip().str.lower()
    df["artist_id"] = artist
    df = df.rename(columns={"user": "user_id"})
    df = df.sort_values(["user_id", "datetime"], kind="stable").reset_index(drop=True)
    log.append(stats(df, "cleaned"))

    # 2. Deduplicate: same user, same track, within N seconds of the previous play.
    same_user = df["user_id"].eq(df["user_id"].shift())
    same_item = df["item_id"].eq(df["item_id"].shift())
    gap = (df["datetime"] - df["datetime"].shift()).dt.total_seconds()
    dup = same_user & same_item & (gap < args.dedupe_seconds)
    df = df[~dup].reset_index(drop=True)
    log.append(stats(df, "deduplicated"))

    # 3. Filter rare tracks and light users (two passes so both conditions hold).
    for _ in range(2):
        item_counts = df["item_id"].map(df["item_id"].value_counts())
        df = df[item_counts >= args.min_item_plays]
        user_counts = df["user_id"].map(df["user_id"].value_counts())
        df = df[user_counts >= args.min_user_events]
    df = df.reset_index(drop=True)
    log.append(stats(df, "filtered"))

    # Item features: artist of each track (most common artist if a track ID maps to several).
    artist = df.groupby("item_id")["artist_id"].agg(lambda s: s.mode().iat[0]).reset_index()
    item_features = artist.rename(columns={"item_id": "id", "artist_id": "value"})
    item_features["feature"] = "artist"
    item_features[["id", "feature", "value"]].to_parquet(out / "item_features.parquet", index=False)

    interactions = df[["user_id", "item_id", "datetime"]].copy()
    interactions["weight"] = 1.0
    interactions.to_parquet(out / "interactions.parquet", index=False)

    summary = {
        "steps": log,
        "first_event": str(df["datetime"].min()),
        "last_event": str(df["datetime"].max()),
        "artists": int(df["artist_id"].nunique()),
        "events_per_user_median": float(df.groupby("user_id").size().median()),
        "repeat_share": float(df.duplicated(["user_id", "item_id"]).mean()),
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
