"""Clean, deduplicate and filter the Last.fm 1K listening data (TR-03, data and eval).

Usage:
    python src/prepare_lastfm.py --raw data/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv \
        --out data/processed/lastfm

Writes to --out:
    interactions.parquet   one row per listen: user_id, item_id, datetime, weight
    item_features.parquet  the artist of each track, in RecTools' long format (id, feature, value)
    stats.json             events, users and tracks left after each step, for the report

The raw file has about 19M rows; a full run takes a few minutes and around 10 GB of memory on a
laptop. Listens after --last-date (default 2009-06-30) are dropped: the data ends in June 2009
and the two rows dated later are clock errors.

After each step the table is saved to --out/checkpoints/, so if the run dies later on
(our first one did, while writing the output) the next run starts from the last finished step
instead of reading the 2.5 GB file again. Add --fresh to ignore those checkpoints.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

from checkpoint import StageCheckpoints, write_json, write_parquet

RAW_COLUMNS = ["user", "timestamp", "artist_mbid", "artist_name", "track_mbid", "track_name"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        quoting=csv.QUOTE_NONE,  # some track names contain stray " characters
        on_bad_lines="skip",
        dtype=str,
    )
    # Timestamps are in UTC. RecTools wants plain timestamps, so we drop the timezone.
    df["datetime"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    return df.drop(columns="timestamp")


def count(df: pd.DataFrame, step: str) -> dict:
    """How much data is left after a step."""
    return {
        "step": step,
        "events": int(len(df)),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
    }


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep listens with a time, an artist and a track name, and give every track a text ID.

    Many rows have no MusicBrainz ID, so the track ID is always "artist - title" in lower case.
    Mixing MBIDs and text IDs would give the same song two different IDs.
    """
    df = raw.dropna(subset=["datetime", "artist_name", "track_name"])
    artist = df["artist_name"].str.strip().str.lower()
    df = pd.DataFrame({
        "user_id": df["user"],
        "item_id": artist + " - " + df["track_name"].str.strip().str.lower(),
        "artist_id": artist,
        "datetime": df["datetime"],
    })
    return df.sort_values(["user_id", "datetime"], kind="stable").reset_index(drop=True)


def drop_after(df: pd.DataFrame, last_date: str) -> pd.DataFrame:
    """Drop listens dated after the end of the data collection.

    The dataset ends in June 2009, but two rows claim to be from 2010 and 2013. They are clock
    errors, and since the test window is measured back from the latest listen, a single one of
    them would leave the test set almost empty.
    """
    end = pd.Timestamp(last_date) + pd.Timedelta(days=1)  # keep the whole last day
    return df[df["datetime"] < end].reset_index(drop=True)


def drop_repeat_scrobbles(df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Drop a listen if the user's previous listen was the same track less than `seconds` earlier.

    These are usually one play logged twice. `df` has to be sorted by user and time.
    """
    same_user = df["user_id"].eq(df["user_id"].shift())
    same_item = df["item_id"].eq(df["item_id"].shift())
    gap = (df["datetime"] - df["datetime"].shift()).dt.total_seconds()
    repeat = same_user & same_item & (gap < seconds)
    return df[~repeat].reset_index(drop=True)


def drop_rare(df: pd.DataFrame, min_item_plays: int, min_user_events: int) -> pd.DataFrame:
    """Drop tracks with too few plays and users with too few listens.

    Dropping tracks can push a user under the limit and the other way round, so we go round
    twice. That isn't guaranteed to settle completely, but it gets very close.
    """
    for _ in range(2):
        df = df[df["item_id"].map(df["item_id"].value_counts()) >= min_item_plays]
        df = df[df["user_id"].map(df["user_id"].value_counts()) >= min_user_events]
    return df.reset_index(drop=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-item-plays", type=int, default=5)
    p.add_argument("--min-user-events", type=int, default=20)
    p.add_argument("--dedupe-seconds", type=int, default=30)
    p.add_argument("--last-date", default="2009-06-30", help="drop listens after this day (bad timestamps)")
    p.add_argument("--fresh", action="store_true", help="ignore checkpoints from earlier runs")
    args = p.parse_args(argv)
    out = Path(args.out)

    # Each step's settings include the ones before it, so changing e.g. --min-item-plays
    # reuses the cleaned and deduplicated tables but redoes the filtering.
    raw_file = Path(args.raw)
    s_clean = {"raw": raw_file.name, "raw_bytes": raw_file.stat().st_size, "last_date": args.last_date}
    s_dedupe = {**s_clean, "dedupe_seconds": args.dedupe_seconds}
    s_filter = {**s_dedupe, "min_item_plays": args.min_item_plays, "min_user_events": args.min_user_events}

    def load_and_clean(_):
        raw = load_raw(args.raw)
        raw_count = {"step": "raw", "events": int(len(raw)), "users": int(raw["user"].nunique()), "items": None}
        df = clean(raw)
        cleaned_count = count(df, "cleaned")
        df = drop_after(df, args.last_date)
        return df, [raw_count, cleaned_count, count(df, f"up to {args.last_date}")]

    steps = [
        ("cleaned", s_clean, load_and_clean),
        ("deduplicated", s_dedupe, lambda df: (drop_repeat_scrobbles(df, args.dedupe_seconds), None)),
        ("filtered", s_filter, lambda df: (drop_rare(df, args.min_item_plays, args.min_user_events), None)),
    ]

    # Start after the latest step that has a usable checkpoint.
    ckpt = StageCheckpoints(out / "checkpoints", resume=not args.fresh)
    df, log, start = None, [], 0
    for i in reversed(range(len(steps))):
        saved = ckpt.load(steps[i][0], steps[i][1])
        if saved is not None:
            df, info = saved
            log, start = info["log"], i + 1
            print(f"picked up from checkpoint '{steps[i][0]}' ({len(df):,} events)")
            break

    for name, settings, run in steps[start:]:
        df, counts = run(df)
        counts = counts or [count(df, name)]
        log = log + counts
        ckpt.save(name, df, settings, {"log": log})
        for c in counts:
            print(c)

    # Item features: the artist of each track (the most common one if a track ID has several).
    artist = df.groupby("item_id")["artist_id"].agg(lambda s: s.mode().iat[0]).reset_index()
    item_features = artist.rename(columns={"item_id": "id", "artist_id": "value"})
    item_features["feature"] = "artist"
    write_parquet(item_features[["id", "feature", "value"]], out / "item_features.parquet")

    interactions = df[["user_id", "item_id", "datetime"]].copy()
    interactions["weight"] = 1.0
    write_parquet(interactions, out / "interactions.parquet")

    summary = {
        "steps": log,
        "first_event": str(df["datetime"].min()),
        "last_event": str(df["datetime"].max()),
        "artists": int(df["artist_id"].nunique()),
        "events_per_user_median": float(df.groupby("user_id").size().median()),
        "repeat_share": float(df.duplicated(["user_id", "item_id"]).mean()),
    }
    write_json(summary, out / "stats.json")  # written last, so it only exists for a finished run
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
