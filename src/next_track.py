"""Next-track evaluation on Last.fm, the same loop for every model (TR-08).

For each sampled test listen (lastfm_data.sample_test_targets: up to 100 per user, seeded), the
history is the listener's last `max_len` plays before it, from train, validation and earlier test
listens, with their timestamps plus the listen's own time as the request time. The scorer ranks
every catalog track, nothing is filtered (replays count), and the rank of the true track is kept.
Results: HR@10 and NDCG@10 over all sampled listens, over replays (tracks the listener played
before) and over discoveries; for TempoRec also the average gate value before each kind.

A scorer is `score_fn(x, unix_ts) -> (scores, gate or None)`: x [batch, max_len] item ids with 0
as padding, unix_ts [batch, max_len + 1] in seconds (last column = request time), scores
[batch, n_ids] numpy or torch. `to_ids` maps lastfm_data catalog positions to the scorer's ids
(0 = unknown to the scorer); ids below `n_skip` (padding and other special tokens) are never ranked.
"""
import typing as tp

import numpy as np
import pandas as pd
import torch

from temporec import score_next


def history_batches(seq: dict, targets: np.ndarray, to_ids: np.ndarray, max_len: int, batch_size: int):
    """Yield (x, unix_ts, true ids, replay flags) for the sampled target positions."""
    items, times = to_ids[seq["item"]], np.asarray(seq["time"], dtype=np.int64)
    user_start = seq["starts"][np.searchsorted(seq["starts"], targets, side="right") - 1]
    for lo in range(0, len(targets), batch_size):
        pos = targets[lo : lo + batch_size]
        x = np.zeros((len(pos), max_len), dtype=np.int64)
        ts = np.zeros((len(pos), max_len + 1), dtype=np.int64)
        for r, (p, s) in enumerate(zip(pos, user_start[lo : lo + batch_size])):
            first = max(s, p - 2 * max_len)  # a margin in case some history tracks are unknown to the scorer
            hist, hist_ts = items[first:p], times[first:p]
            keep = hist != 0
            hist, hist_ts = hist[keep][-max_len:], hist_ts[keep][-max_len:]
            n = len(hist)
            x[r, max_len - n :] = hist
            ts[r, max_len - n : max_len] = hist_ts
            ts[r, max_len] = times[p]
            ts[r, : max_len - n] = hist_ts[0] if n else times[p]  # padding times, as RecTools fills them
        yield x, ts, items[pos], np.asarray(seq["replay"])[pos]


def evaluate(score_fn: tp.Callable, seq: dict, targets: np.ndarray, to_ids: np.ndarray, max_len: int,
             n_skip: int = 1, k: int = 10, batch_size: int = 128) -> tp.Tuple[dict, pd.DataFrame]:
    """Rank every sampled test listen; return the summary and one row per target."""
    rows = []
    for x, ts, true, replay in history_batches(seq, targets, to_ids, max_len, batch_size):
        scores, gate = score_fn(x, ts)
        scores = torch.as_tensor(scores, dtype=torch.float32).clone()
        scores[:, :n_skip] = float("-inf")
        true_t = torch.as_tensor(true, device=scores.device)
        true_score = scores.gather(1, true_t.clamp(min=0).unsqueeze(1))
        rank = (scores > true_score).sum(dim=1).cpu().numpy() + 1
        rank = np.where(true == 0, np.inf, rank)  # a track the scorer has never seen can't be ranked
        rows.append(pd.DataFrame({
            "rank": rank, "replay": replay,
            "gate": np.nan if gate is None else torch.as_tensor(gate).float().cpu().numpy(),
        }))
    per_target = pd.concat(rows, ignore_index=True)
    return summarize(per_target, k), per_target


def summarize(per_target: pd.DataFrame, k: int = 10) -> dict:
    """HR@k and NDCG@k overall, for replays and for discoveries, plus the gate before each."""
    hit = per_target["rank"] <= k
    ndcg = np.where(hit, 1 / np.log2(per_target["rank"].clip(upper=1e9) + 1), 0.0)
    out = {}
    for name, mask in (("all", np.ones(len(per_target), bool)), ("replay", per_target["replay"].to_numpy()),
                       ("discovery", ~per_target["replay"].to_numpy())):
        out[f"{name}_n"] = int(mask.sum())
        out[f"{name}_hr@{k}"] = float(hit[mask].mean()) if mask.any() else float("nan")
        out[f"{name}_ndcg@{k}"] = float(ndcg[mask].mean()) if mask.any() else float("nan")
        if per_target["gate"].notna().any():
            out[f"{name}_gate"] = float(per_target.loc[mask, "gate"].mean()) if mask.any() else float("nan")
    out["unscorable"] = int(np.isinf(per_target["rank"]).sum())
    return out


def model_scorer(model, catalog: pd.Index, device: tp.Optional[str] = None):
    """(score_fn, to_ids, n_skip) for a fitted RecTools SASRec/HSTU/TempoRec model."""
    prep = model.data_preparator
    to_ids = pd.Series(catalog).map(prep.item_id_map.to_internal).fillna(0).astype(np.int64).to_numpy()
    backbone = model.lightning_model.torch_model
    device = device or next(backbone.parameters()).device
    backbone.eval()

    @torch.no_grad()
    def score_fn(x, ts):
        batch = {"x": torch.as_tensor(x, device=device), "unix_ts": torch.as_tensor(ts, device=device)}
        return score_next(backbone, batch)

    return score_fn, to_ids, prep.n_item_extra_tokens


def catalog_ids(n_items: int) -> np.ndarray:
    """Ids for scorers that work on catalog positions directly: position i -> i + 1 (0 = padding)."""
    return np.arange(1, n_items + 1, dtype=np.int64)


def popular_scorer(seq: dict):
    """Most popular: training play counts, the same for every listener."""
    counts = np.bincount(np.asarray(seq["item"])[np.asarray(seq["split"]) == 0] + 1,
                         minlength=seq["n_items"] + 1).astype(np.float32)
    return lambda x, ts: (np.broadcast_to(counts, (len(x), len(counts))), None)


def recent_scorer(n_items: int):
    """Recently played: the more recent the last play of a track in the history, the higher it ranks."""
    def score_fn(x, ts):
        scores = np.zeros((len(x), n_items + 1), dtype=np.float32)
        order = np.arange(1, x.shape[1] + 1, dtype=np.float32)
        for r in range(len(x)):
            scores[r, x[r]] = order  # for repeated tracks the latest position wins
        scores[:, 0] = 0
        return scores, None
    return score_fn
