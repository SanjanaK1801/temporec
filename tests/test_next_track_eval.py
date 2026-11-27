import numpy as np
import pandas as pd
from next_track_eval import build_mini_inputs, metrics_from_ranks, score_reco


def fake_seq():
    # user 0 = positions 0..3, user 1 = positions 4..6
    return {
        "item": np.array([0, 1, 2, 3, 1, 2, 0]),
        "time": np.array([10, 20, 30, 40, 50, 60, 70]),
        "starts": np.array([0, 4]),
        "catalog": pd.Index(["a", "b", "c", "d"]),
    }


def test_history_stops_before_target():
    inter, ctx = build_mini_inputs(fake_seq(), np.array([3, 6]))
    assert len(inter) == 5                                    # 3 + 2 history events
    assert inter[inter.user_id == 0].item_id.tolist() == ["a", "b", "c"]
    assert inter[inter.user_id == 1].item_id.tolist() == ["b", "c"]
    assert ctx.datetime.tolist() == [pd.Timestamp(40, unit="s"), pd.Timestamp(70, unit="s")]


def test_history_length_cap():
    inter, _ = build_mini_inputs(fake_seq(), np.array([3]), hist_len=2)
    assert inter.item_id.tolist() == ["b", "c"]


def test_metrics_by_hand():
    # ranks 1, miss, 3, miss -> HR 2/4, NDCG (1 + 1/log2(4)) / 4 = 0.375
    res = metrics_from_ranks(np.array([1, 0, 3, 0]), np.array([True, True, False, False]), 5, 10)
    assert res["all"]["HR@10"] == 0.5
    assert abs(res["all"]["NDCG@10"] - 0.375) < 1e-9
    assert res["replay"]["HR@10"] == 0.5 and abs(res["replay"]["NDCG@10"] - 0.5) < 1e-9
    assert res["new"]["HR@10"] == 0.5 and abs(res["new"]["NDCG@10"] - 0.25) < 1e-9
    assert res["coverage"] == 0.5


def test_score_reco():
    reco = pd.DataFrame({"user_id": [0, 0, 1, 1], "item_id": ["x", "y", "p", "q"], "rank": [1, 2, 1, 2]})
    res = score_reco(reco, ["y", "z"], [True, False], n_items=10)
    assert res["all"]["HR@10"] == 0.5 and abs(res["all"]["NDCG@10"] - 0.5 / np.log2(2) * np.log2(2) / np.log2(3)) < 1e-9