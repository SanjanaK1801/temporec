import argparse
import json
import os
import numpy as np
from lastfm_data import load_sequences, sample_test_targets, SPLIT_CODE

K = 10


def recent_topk(items, s, p, k=K, max_back=20000):
    """The k most recently played distinct tracks before position p."""
    out, seen = [], set()
    i, lo = p - 1, max(s, p - max_back)
    while i >= lo and len(out) < k:
        it = int(items[i])
        if it not in seen:
            seen.add(it)
            out.append(it)
        i -= 1
    return out


def score(recs, truth, replay, n_items):
    n = len(truth)
    hit, ndcg = np.zeros(n), np.zeros(n)
    covered = set()
    for j, (r, t) in enumerate(zip(recs, truth)):
        covered.update(r)
        if t in r:
            hit[j] = 1.0
            ndcg[j] = 1.0 / np.log2(r.index(t) + 2)
    res = {"coverage": len(covered) / n_items}
    for name, m in [("all", np.ones(n, bool)), ("replay", replay), ("new", ~replay)]:
        res[name] = {"n": int(m.sum()),
                     "HR@10": float(hit[m].mean()) if m.any() else None,
                     "NDCG@10": float(ndcg[m].mean()) if m.any() else None}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-user", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seq = load_sequences()
    t = sample_test_targets(seq, a.per_user, a.seed)
    items = seq["item"]
    truth = items[t].tolist()
    replay = seq["replay"][t]
    user_start = seq["starts"][np.searchsorted(seq["starts"], t, side="right") - 1]

    counts = np.bincount(items[seq["split"] == SPLIT_CODE["train"]],
                         minlength=seq["n_items"])
    top = np.argsort(-counts)[:K].tolist()

    results = {
        "per_user": a.per_user, "seed": a.seed, "targets": int(len(t)),
        "most_popular": score([top] * len(t), truth, replay, seq["n_items"]),
        "recently_played": score(
            [recent_topk(items, int(s), int(p)) for s, p in zip(user_start, t)],
            truth, replay, seq["n_items"]),
    }
    print(json.dumps(results, indent=2))
    os.makedirs("results", exist_ok=True)
    with open("results/next_track_baselines.json", "w") as f:
        json.dump(results, f, indent=2)