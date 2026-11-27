"""Demo: one listener, their recent tracks, and what each method recommends next.

    python src/demo.py --kind new --seed 3
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from lastfm_data import DATA, SPLIT_CODE, load_sequences, sample_test_targets
from next_track_baselines import K, recent_topk


def show(catalog, ids, truth):
    for r, i in enumerate(ids, 1):
        mark = "   <-- the actual next track" if i == truth else ""
        print(f"  {r:>2}. {catalog[i]}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=["any", "replay", "new"], default="any")
    ap.add_argument("--shown", type=int, default=10)
    a = ap.parse_args()

    seq = load_sequences(a.data)
    items, times, catalog = seq["item"], seq["time"], seq["catalog"]
    targets = sample_test_targets(seq, 100, 0)
    if a.kind != "any":
        targets = targets[seq["replay"][targets] == (a.kind == "replay")]
    p = int(np.random.default_rng(a.seed).choice(targets))
    u = int(np.searchsorted(seq["starts"], p, side="right") - 1)
    s = int(seq["starts"][u])
    truth = int(items[p])

    print(f"Listener {u}: {p - s:,} earlier listens. Last {a.shown} before the request:")
    for i in range(max(s, p - a.shown), p):
        print(f"  {pd.to_datetime(times[i], unit='s')}  {catalog[items[i]]}")
    kind = "a REPLAY" if seq["replay"][p] else "a NEW track"
    print(f"\nRequest time: {pd.to_datetime(times[p], unit='s')} UTC")
    print(f"Actual next track: {catalog[truth]}  ({kind} for this listener)")

    counts = np.bincount(items[seq["split"] == SPLIT_CODE["train"]], minlength=seq["n_items"])
    top = np.argsort(-counts)[:K].tolist()
    print("\nMost popular top 10:")
    show(catalog, top, truth)
    print("\nRecently played top 10:")
    show(catalog, recent_topk(items, s, p), truth)

    if os.path.exists("results/next_track_baselines.json"):
        r = json.load(open("results/next_track_baselines.json"))
        print(f"\nBaselines on {r['targets']:,} sampled test listens:")
        for k in ("most_popular", "recently_played"):
            m = r[k]["all"]
            print(f"  {k:16} HR@10 {m['HR@10']:.4f}  NDCG@10 {m['NDCG@10']:.4f}")
    if os.path.exists("results/lastfm.csv"):
        df = pd.read_csv("results/lastfm.csv")
        cols = [c for c in ["model", "seed", "hr@10", "ndcg@10", "hr@10_replay", "hr@10_new", "min_per_epoch"] if c in df]
        print("\nTrained models:")
        print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()