import numpy as np
import pandas as pd

K = 10


def build_mini_inputs(seq, targets, hist_len=200):
    """One mini-user per sampled target.

    Returns (interactions, context):
      interactions: user_id (0..n-1), item_id, datetime, weight. The history is the
                    last `hist_len` events of the real user BEFORE the target
                    (train, validation and earlier test listens).
      context:      one row per mini-user with the target's own timestamp,
                    which is the request time for RecTools' recommend(context=...).
    """
    items, times, catalog = seq["item"], seq["time"], seq["catalog"]
    starts = seq["starts"]
    ustart = starts[np.searchsorted(starts, targets, side="right") - 1]
    pos_parts, uid_parts = [], []
    for j, (s, p) in enumerate(zip(ustart, targets)):
        pos = np.arange(max(int(s), int(p) - hist_len), int(p))
        pos_parts.append(pos)
        uid_parts.append(np.full(len(pos), j))
    pos = np.concatenate(pos_parts)
    inter = pd.DataFrame({
        "user_id": np.concatenate(uid_parts),
        "item_id": catalog[items[pos]],
        "datetime": pd.to_datetime(times[pos], unit="s"),
        "weight": 1.0,
    })
    context = pd.DataFrame({
        "user_id": np.arange(len(targets)),
        "datetime": pd.to_datetime(times[targets], unit="s"),
    })
    return inter, context


def metrics_from_ranks(rank, replay, n_covered, n_items):
    """rank: array, 0 = true track not in the top-k, else its 1-based rank."""
    hit = (rank > 0).astype(float)
    ndcg = np.where(rank > 0, 1.0 / np.log2(np.maximum(rank, 1) + 1), 0.0)
    res = {"coverage": n_covered / n_items}
    for name, m in [("all", np.ones(len(rank), bool)), ("replay", replay), ("new", ~replay)]:
        res[name] = {"n": int(m.sum()),
                     "HR@10": float(hit[m].mean()) if m.any() else None,
                     "NDCG@10": float(ndcg[m].mean()) if m.any() else None}
    return res


def score_reco(reco, truth_ids, replay, n_items, k=K):
    """reco: DataFrame with user_id (the mini-user number), item_id, rank.
    truth_ids: the true next track for each mini-user (index = mini-user number)."""
    reco = reco[reco["rank"] <= k]
    truth = pd.Series(np.asarray(truth_ids), name="truth").rename_axis("user_id").reset_index()
    hits = reco.merge(truth, on="user_id")
    hits = hits[hits["item_id"] == hits["truth"]]
    rank = np.zeros(len(truth_ids), dtype=int)
    rank[hits["user_id"].to_numpy()] = hits["rank"].to_numpy()
    return metrics_from_ranks(rank, np.asarray(replay), reco["item_id"].nunique(), n_items)