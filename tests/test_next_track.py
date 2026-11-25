import math

import numpy as np
import pandas as pd
import pytest

from next_track import (catalog_ids, evaluate, history_batches, popular_scorer, recent_scorer,
                        summarize)


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
    assert out["replay_hr@10"] == 0.5 and out["discovery_hr@10"] == 0.5
    assert out["unscorable"] == 1 and "all_gate" not in out


def test_recently_played_ranks_by_last_play():
    # target 5 (id 2) follows ids 1 2 1 3 4: last plays 4 > 3 > 1 > 2, so id 2 ranks 4th
    # target 10 (id 5) was never played by listener 1: ids 4 and 3 rank above it, ties at 0 don't count
    _, per_target = evaluate(recent_scorer(5), toy_seq(), TARGETS, catalog_ids(5), max_len=10, k=1)
    assert per_target["rank"].tolist() == [4, 4, 5, 3]


def test_most_popular_uses_train_counts_only():
    # train plays: track 0 twice, track 1 once, track 2 twice (listener 1); tracks 3 and 4 never
    out, per_target = evaluate(popular_scorer(toy_seq()), toy_seq(), TARGETS, catalog_ids(5), max_len=10, k=3)
    assert per_target["rank"].tolist() == [4, 3, 4, 4]
    assert out["all_hr@3"] == 0.25 and per_target["gate"].isna().all()


def test_gate_values_are_split_by_replay():
    def fake(x, ts):  # pretend the gate is high right after track 3 (id 4)
        return np.zeros((len(x), 6)), np.where(x[:, -1] == 4, 0.9, 0.1)

    out, _ = evaluate(fake, toy_seq(), TARGETS, catalog_ids(5), max_len=10)
    assert out["replay_gate"] == pytest.approx(0.9)  # target 5 comes right after track 3
    assert out["discovery_gate"] == pytest.approx((0.1 + 0.1 + 0.9) / 3)
