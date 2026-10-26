import pandas as pd
import pytest

from baselines import most_popular, recently_played
from evaluation import drop_seen, evaluate, first_time_listens

T = pd.Timestamp("2009-01-01")
TRAIN = pd.DataFrame({
    "user_id": ["u1", "u1", "u1", "u2", "u2"],
    "item_id": ["a", "b", "c", "a", "d"],
    "datetime": [T, T + pd.Timedelta(hours=1), T + pd.Timedelta(hours=2), T, T + pd.Timedelta(hours=1)],
})
TEST = pd.DataFrame({"user_id": ["u1", "u1", "u2"], "item_id": ["b", "b", "c"]})


def reco(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rank"])


def test_hit_rate_and_ndcg_by_hand():
    # u1's target b at rank 1 (hit, NDCG 1); u2's target c absent (miss, NDCG 0).
    r = reco([("u1", "b", 1), ("u1", "a", 2), ("u2", "a", 1), ("u2", "d", 2)])
    m = evaluate(r, TEST, TRAIN)
    assert m["hr@10"] == pytest.approx(0.5)
    assert m["ndcg@10"] == pytest.approx(0.5)


def test_repeated_test_plays_count_once():
    r = reco([("u1", "b", 1), ("u2", "c", 1)])
    assert evaluate(r, TEST, TRAIN)["recall@10"] == pytest.approx(1.0)


def test_drop_seen_reranks():
    r = reco([("u1", "a", 1), ("u1", "x", 2), ("u1", "b", 3), ("u1", "y", 4)])
    out = drop_seen(r, TRAIN, k=10)
    assert list(out["item_id"]) == ["x", "y"] and list(out["rank"]) == [1, 2]


def test_first_time_listens():
    assert list(first_time_listens(TEST, TRAIN)["item_id"]) == ["c"]


def test_baselines():
    pop = most_popular(TRAIN, ["u1", "u2"], k=1)
    assert list(pop["item_id"]) == ["a", "a"]
    recent = recently_played(TRAIN, ["u1"], k=2)
    assert list(recent["item_id"]) == ["c", "b"] and list(recent["rank"]) == [1, 2]
