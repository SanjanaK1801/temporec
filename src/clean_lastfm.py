# It implements the decisions from the plan: 
# a track is (artist, title) lowercased and stripped, 
# items with fewer than 5 plays are dropped, 
# users with fewer than 20 events are dropped, 
# and each user’s events are sorted oldest first.

import argparse
import os
import numpy as np
import pandas as pd

RAW = "/scratch/zt1/project/msml612/shared/next_song_prediction/data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
OUT_DIR = "/scratch/zt1/project/msml612/shared/next_song_prediction/data/processed"
MIN_ITEM_COUNT = 5
MIN_USER_EVENTS = 20

parser = argparse.ArgumentParser()
parser.add_argument("--nrows", type=int, default=None)
parser.add_argument("--tag", default="full")
args = parser.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)
lines = []

def log(msg):
    print(msg, flush=True)
    lines.append(msg)

cols = ["user_id", "timestamp", "artist_mbid", "artist_name",
        "track_mbid", "track_name"]
df = pd.read_csv(
    RAW, sep="\t", header=None, names=cols,
    usecols=["user_id", "timestamp", "artist_name", "track_name"],
    nrows=args.nrows, quoting=3, on_bad_lines="skip", dtype=str,
)
log(f"RAW: {len(df):,} events, {df.user_id.nunique():,} users")

# 1. drop rows with missing artist or track name, then normalize
df = df.dropna(subset=["artist_name", "track_name"])
df["artist_name"] = df["artist_name"].str.strip().str.lower()
df["track_name"] = df["track_name"].str.strip().str.lower()
df = df[(df["artist_name"] != "") & (df["track_name"] != "")]
log(f"After dropping missing/blank names: {len(df):,} events")

# 2. define a track = (artist, title)
df["item_key"] = df["artist_name"] + " ||| " + df["track_name"]
df["item_id"], _ = pd.factorize(df["item_key"])
df = df.drop(columns=["item_key"])
log(f"Distinct tracks before filtering: {df.item_id.nunique():,}")

# 3. drop rare tracks
counts = np.bincount(df["item_id"].to_numpy())
df = df[counts[df["item_id"].to_numpy()] >= MIN_ITEM_COUNT]
log(f"After min item count {MIN_ITEM_COUNT}: {len(df):,} events")

# 4. drop users with too few events
ucounts = df["user_id"].value_counts()
keep_users = ucounts[ucounts >= MIN_USER_EVENTS].index
df = df[df["user_id"].isin(keep_users)]
log(f"After min user events {MIN_USER_EVENTS}: {len(df):,} events")

# 5. parse time, make clean integer ids, sort oldest first
ts = pd.to_datetime(df["timestamp"], utc=True)
df["timestamp"] = ts.astype("int64") // 10**9   # unix seconds
df["user_idx"], _ = pd.factorize(df["user_id"])
df["artist_id"], _ = pd.factorize(df["artist_name"])
df["item_idx"], _ = pd.factorize(df["item_id"])

item_map = (df[["item_idx", "artist_id", "artist_name", "track_name"]]
            .drop_duplicates("item_idx").sort_values("item_idx"))

out = df[["user_idx", "item_idx", "artist_id", "timestamp"]].rename(
    columns={"user_idx": "user_id", "item_idx": "item_id"})
out = out.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)

out.to_parquet(f"{OUT_DIR}/interactions_{args.tag}.parquet", index=False)
item_map.to_parquet(f"{OUT_DIR}/items_{args.tag}.parquet", index=False)

# 6. final stats for the report
lens = out.groupby("user_id").size()
log("---- FINAL ----")
log(f"Users: {out.user_id.nunique():,}")
log(f"Tracks: {out.item_id.nunique():,}")
log(f"Artists: {out.artist_id.nunique():,}")
log(f"Events: {len(out):,}")
log(f"Events per user: min {lens.min():,}, median {int(lens.median()):,}, max {lens.max():,}")
log("Time range: " + str(pd.to_datetime(out.timestamp.min(), unit="s")) +
    " to " + str(pd.to_datetime(out.timestamp.max(), unit="s")))

with open(f"{OUT_DIR}/cleaning_stats_{args.tag}.txt", "w") as f:
    f.write("\n".join(lines))