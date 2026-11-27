import math

import numpy as np
import pandas as pd
import pytest

from next_track import catalog_ids, evaluate, history_batches, summarize


def toy_seq():
    # Two listeners; catalog positions 0..4 (scorer ids 1..5); split 0 = train, 1 = val, 2 = test.
    #   listener 0, positions 0-6:  tracks 0 1 0 2 3 1 4
    #   listener 1, positions 7-10: tracks 2 2 3 4
    item = np.array([0, 1, 0, 2, 3, 1, 4, 2, 2, 3, 4])
    split = np.array([0, 0, 0, 1, 2, 2, 2, 0, 0, 1, 2])
    time = np.arange(len(item), dtype=np.int64) * 3600 + 1_200_000_000
    user = np.repeat([0, 1], [7, 4])
    replay = pd.DataFrame({"u": user, "i": item}).groupby(["u", "i"]).cumcount().to_numpy() > 0
    return {"item": item, "split": split, "time": time, "replay": replay,
            "starts": np.array([0, 7]), "ends": np.array([7, 11]), "n_items": 5}


TARGETS = np.array([4, 5, 6, 10])  # replays: only position 5 (listener 0 plays track 1 again)


def test_history_is_the_plays_just_before_the_target():
    seq = toy_seq()
    x, ts, true, replay = next(history_batches(seq, np.array([5, 10]), catalog_ids(5), max_len=3, batch_size=8))
    assert x.tolist() == [[1, 3, 4], [3, 3, 4]]  # nothing from the other listener or from the future
    assert ts[:, -1].tolist() == seq["time"][[5, 10]].tolist()  # request time = the listen's own time
    assert ts[0, :3].tolist() == seq["time"][2:5].tolist()
    assert true.tolist() == [2, 5] and replay.tolist() == [True, False]


def test_short_histories_are_left_padded():
    seq = toy_seq()
    x, ts, _, _ = next(history_batches(seq, np.array([8]), catalog_ids(5), max_len=4, batch_size=8))
    assert x.tolist() == [[0, 0, 0, 3]]
    assert ts[0, :3].tolist() == [seq["time"][7]] * 3  # padding repeats the first real time, as RecTools does


def test_hand_computed_metrics():
    per_target = pd.DataFrame({"rank": [1, 3, 11, np.inf], "replay": [True, False, True, False], "gate": np.nan})
    out = summarize(per_target, k=10)
    assert out["all_hr@10"] == 0.5
    assert out["all_ndcg@10"] == pytest.approx((1 + 1 / math.log2(4)) / 4)
    assert out["replay_hr@10"] == 0.5 and out["new_hr@10"] == 0.5
    assert out["unscorable"] == 1 and "all_gate" not in out


def fixed_scores(x, ts):
    # ids 0..5; id 4 always best, then 5, 1, 2, 3 (id 0 is padding and never ranked)
    return np.tile(np.array([9.0, 3.0, 2.0, 1.0, 5.0, 4.0]), (len(x), 1)), None


def test_ranks_come_from_the_scores():
    # targets: id 4, id 2, id 5, id 5 -> ranks 1, 4, 2, 2
    out, per_target = evaluate(fixed_scores, toy_seq(), TARGETS, catalog_ids(5), max_len=10, k=2)
    assert per_target["rank"].tolist() == [1, 4, 2, 2]
    assert out["all_hr@2"] == 0.75 and out["replay_hr@2"] == 0.0 and out["new_hr@2"] == 1.0
    assert per_target["gate"].isna().all()


def test_coverage_counts_distinct_top_k_tracks():
    out, _ = evaluate(fixed_scores, toy_seq(), TARGETS, catalog_ids(5), max_len=10, k=2)
    assert out["coverage@2"] == 2 / 5  # every listener gets ids 4 and 5


def test_gate_values_are_split_by_replay():
    def fake(x, ts):  # pretend the gate is high right after track 3 (id 4)
        return np.zeros((len(x), 6)), np.where(x[:, -1] == 4, 0.9, 0.1)

    out, _ = evaluate(fake, toy_seq(), TARGETS, catalog_ids(5), max_len=10)
    assert out["replay_gate"] == pytest.approx(0.9)  # target 5 comes right after track 3
    assert out["new_gate"] == pytest.approx((0.1 + 0.1 + 0.9) / 3)
