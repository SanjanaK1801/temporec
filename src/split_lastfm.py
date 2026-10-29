import json
import numpy as np
import pandas as pd

PROC = "/scratch/zt1/project/msml612/shared/next_song_prediction/data/processed"
TRAIN_FRAC, VAL_FRAC = 0.80, 0.10
lines = []

def log(msg):
    print(msg, flush=True)
    lines.append(msg)

df = pd.read_parquet(f"{PROC}/interactions_full.parquet")
log(f"Loaded {len(df):,} events")

# drop the stray 2010 and 2013 timestamps
limit = int(pd.Timestamp("2010-01-01", tz="UTC").timestamp())
n_bad = int((df["timestamp"] >= limit).sum())
df = df[df["timestamp"] < limit].reset_index(drop=True)
log(f"Dropped {n_bad} events dated 2010 or later; {len(df):,} remain")

# replay flag: has this user played this track earlier?
# (data is sorted by user_id, timestamp)
df["is_replay"] = df.groupby(["user_id", "item_id"]).cumcount() > 0

# global time cutoffs
ts = df["timestamp"].to_numpy()
c1 = int(np.quantile(ts, TRAIN_FRAC))
c2 = int(np.quantile(ts, TRAIN_FRAC + VAL_FRAC))

def fmt(t):
    return str(pd.to_datetime(t, unit="s", utc=True))

log(f"Train/val cutoff: {fmt(c1)}")
log(f"Val/test cutoff:  {fmt(c2)}")

train = df[df["timestamp"] < c1]
val = df[(df["timestamp"] >= c1) & (df["timestamp"] < c2)]
test = df[df["timestamp"] >= c2]

train_items = set(train["item_id"].unique())

log("\n== Split table ==")
for name, part in [("train", train), ("val", val), ("test", test)]:
    unseen = (~part["item_id"].isin(train_items)).mean()
    log(f"{name}: {len(part):,} events ({len(part)/len(df):.1%}), "
        f"{part.user_id.nunique():,} users, "
        f"{part.item_id.nunique():,} tracks, "
        f"replay share {part.is_replay.mean():.1%}, "
        f"events with track not in train {unseen:.1%}")

log(f"\nTracks in train: {len(train_items):,} of "
    f"{df.item_id.nunique():,} total tracks")

cols = ["user_id", "item_id", "artist_id", "timestamp"]
train[cols].to_parquet(f"{PROC}/train.parquet", index=False)
val[cols].to_parquet(f"{PROC}/val.parquet", index=False)
test[cols].to_parquet(f"{PROC}/test.parquet", index=False)

with open(f"{PROC}/split_cutoffs.json", "w") as f:
    json.dump({"train_val_cutoff": c1, "val_test_cutoff": c2,
               "train_val_cutoff_utc": fmt(c1),
               "val_test_cutoff_utc": fmt(c2)}, f, indent=2)
with open(f"{PROC}/split_stats_full.txt", "w") as f:
    f.write("\n".join(lines))
log("\nDone.")