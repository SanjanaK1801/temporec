"""Evaluation contract shared by every training script (TR-04, data and eval).

evaluate(reco, test, train) -> dict  is the only function training code needs. Do not change
its signature or the metric names without telling your partner.

reco   DataFrame with user_id, item_id, rank (1 = best), at most 10 rows per user
test   DataFrame with user_id, item_id (the plays to predict)
train  DataFrame with user_id, item_id (the history; used for the catalog)
"""
from pathlib import Path

import pandas as pd
from rectools import Columns
from rectools.metrics import NDCG, CatalogCoverage, HitRate, Recall, calc_metrics

METRICS = {
    "hr@10": HitRate(k=10),
    "recall@10": Recall(k=10),
    "ndcg@10": NDCG(k=10, divide_by_achievable=True),  # normalized to 0-1, as in the RecTools tutorial
    "coverage@10": CatalogCoverage(k=10, normalize=True),
}


def evaluate(reco: pd.DataFrame, test: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Ranking metrics at 10. A track played several times in the test window counts once per user."""
    test = test.drop_duplicates([Columns.User, Columns.Item])
    catalog = train[Columns.Item].unique()
    return calc_metrics(METRICS, reco=reco, interactions=test, prev_interactions=train, catalog=catalog)


def first_time_listens(test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Test plays of tracks the user never played in train (the 'discovery' target)."""
    seen = train[[Columns.User, Columns.Item]].drop_duplicates()
    merged = test.merge(seen, how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge")


def drop_seen(reco: pd.DataFrame, train: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Remove tracks the user played in train, re-rank, keep the top k (the 'discovery' ranking)."""
    seen = train[[Columns.User, Columns.Item]].drop_duplicates()
    new = reco.merge(seen, how="left", indicator=True)
    new = new[new["_merge"] == "left_only"].drop(columns="_merge")
    new = new.sort_values([Columns.User, Columns.Rank], kind="stable")
    new[Columns.Rank] = new.groupby(Columns.User).cumcount() + 1
    return new[new[Columns.Rank] <= k]


def save_result(path: str, row: dict) -> None:
    """Append one result row to a CSV, so every run adds to the same results table."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if p.exists():  # rows from different scripts can have different columns; align them
        new = pd.concat([pd.read_csv(p), new], ignore_index=True)
    new.to_csv(p, index=False)
