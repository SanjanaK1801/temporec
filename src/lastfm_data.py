import argparse
import numpy as np
import pandas as pd

DATA = ("/scratch/zt1/project/msml612/shared/next_song_prediction/"
        "data/processed_v2/lastfm")
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def load_sequences(data_dir=DATA):
    """Merge train/val/test into one time-ordered sequence per user.

    Catalog = tracks seen in train. Returns a dict of numpy arrays:
      item     integer track id (index into catalog)
      split    0 = train, 1 = val, 2 = test
      time     unix seconds
      replay   True if this user already played this track earlier
      starts/ends  boundaries of each user's events in the arrays
    """
    parts = []
    for name, code in SPLIT_CODE.items():
        df = pd.read_parquet(f"{data_dir}/{name}.parquet",
                             columns=["user_id", "item_id", "datetime"])
        df["split"] = np.int8(code)
        parts.append(df)
    catalog = pd.Index(parts[0]["item_id"].unique())
    df = pd.concat(parts, ignore_index=True)
    df["item"] = catalog.get_indexer(df["item_id"])
    assert (df["item"] >= 0).all(), "track outside the train catalog"
    df["user"] = pd.factorize(df["user_id"])[0]
    df = df.sort_values(["user", "datetime"], kind="stable").reset_index(drop=True)
    df["replay"] = df.groupby(["user", "item"]).cumcount() > 0

    user = df["user"].to_numpy()
    starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]])
    ends = np.r_[starts[1:], len(df)]
    return {
        "item": df["item"].to_numpy(np.int32),
        "split": df["split"].to_numpy(np.int8),
        "time": df["datetime"].astype("int64").to_numpy() // 10**9,
        "replay": df["replay"].to_numpy(),
        "starts": starts, "ends": ends,
        "n_items": len(catalog), "catalog": catalog,
    }


def sample_test_targets(seq, per_user=100, seed=0):
    """Pick up to `per_user` test listens per user (seeded).
    The history for a target is every event of that user before it,
    including train, validation and earlier test listens."""
    rng = np.random.default_rng(seed)
    out = []
    for s, e in zip(seq["starts"], seq["ends"]):
        pos = s + np.flatnonzero(seq["split"][s:e] == SPLIT_CODE["test"])
        pos = pos[pos > s]                      # needs at least 1 earlier event
        if len(pos) > per_user:
            pos = np.sort(rng.choice(pos, per_user, replace=False))
        out.append(pos)
    return np.concatenate(out) if out else np.array([], dtype=int)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--per-user", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seq = load_sequences(a.data)
    lens = seq["ends"] - seq["starts"]
    print(f"users {len(lens):,}  events {len(seq['item']):,}  tracks {seq['n_items']:,}")
    print(f"events per user: min {lens.min():,} median {int(np.median(lens)):,} max {lens.max():,}")
    for name, code in SPLIT_CODE.items():
        print(f"{name}: {(seq['split'] == code).sum():,} events")
    t = sample_test_targets(seq, a.per_user, a.seed)
    print(f"sampled test targets: {len(t):,} (up to {a.per_user} per user, seed {a.seed})")
    print(f"replay share among sampled targets: {seq['replay'][t].mean():.1%}")
    print("item_features columns:")
    print(pd.read_parquet(f"{a.data}/item_features.parquet").head(3))