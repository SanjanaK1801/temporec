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


def test_save_result_lines_up_different_columns(tmp_path):
    from evaluation import save_result

    out = tmp_path / "results" / "lastfm.csv"
    save_result(out, {"model": "popular", "hr@10": 0.1})
    save_result(out, {"model": "hstu", "hr@10": 0.2, "epochs_run": 12})
    table = pd.read_csv(out)
    assert list(table["model"]) == ["popular", "hstu"]
    assert table["epochs_run"].isna().tolist() == [True, False]


def test_jobs_finishing_together_keep_every_row(tmp_path):
    # 8 processes x 5 rows written to the same CSV at the same time: none may get lost.
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    out = tmp_path / "lastfm.csv"
    code = (f"import sys; sys.path.insert(0, {str(src)!r}); from evaluation import save_result\n"
            f"for i in range(5): save_result({str(out)!r}, {{'job': int(sys.argv[1]), 'i': i}})")
    jobs = [subprocess.Popen([sys.executable, "-c", code, str(j)]) for j in range(8)]
    assert all(job.wait() == 0 for job in jobs)
    table = pd.read_csv(out)
    assert len(table) == 40
    assert sorted(zip(table["job"], table["i"])) == [(j, i) for j in range(8) for i in range(5)]
