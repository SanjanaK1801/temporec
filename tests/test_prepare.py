import json

import pandas as pd
import pytest

import prepare_lastfm
from prepare_lastfm import drop_after, drop_rare, drop_repeat_scrobbles

T = pd.Timestamp("2009-01-01")


def listens(rows):
    return pd.DataFrame(
        [(u, i, T + pd.Timedelta(seconds=s)) for u, i, s in rows], columns=["user_id", "item_id", "datetime"]
    )


def test_quick_repeat_of_the_same_track_is_dropped():
    df = listens([("u1", "a", 0), ("u1", "a", 10), ("u1", "a", 100), ("u1", "b", 105), ("u2", "b", 106)])
    kept = drop_repeat_scrobbles(df, seconds=30)
    assert list(kept["datetime"].dt.second) == [0, 40, 45, 46]  # only the play 10 s after the first goes


def test_rare_tracks_and_light_users_are_dropped():
    df = listens([("u1", "a", 0), ("u1", "a", 1), ("u1", "b", 2), ("u2", "a", 3), ("u2", "a", 4)])
    kept = drop_rare(df, min_item_plays=2, min_user_events=2)
    assert set(kept["item_id"]) == {"a"}  # b has a single play
    assert set(kept["user_id"]) == {"u1", "u2"}


def test_listens_after_the_last_date_are_dropped():
    df = pd.DataFrame({"user_id": ["u1"] * 3, "item_id": ["a"] * 3,
                       "datetime": pd.to_datetime(["2009-06-30 23:59", "2009-07-01 00:00", "2013-09-29 18:32"])})
    assert list(drop_after(df, "2009-06-30")["datetime"].astype(str)) == ["2009-06-30 23:59:00"]


@pytest.fixture
def raw_file(tmp_path):
    # 3 users x 6 listens of 2 tracks, in the raw Last.fm format (tab separated, no header).
    lines = [
        f"user_00000{u}\t2009-01-0{d}T10:00:00Z\t\tArtist {t}\t\tSong {t}"
        for u in (1, 2, 3) for d in range(1, 7) for t in ("A" if d % 2 else "B",)
    ]
    path = tmp_path / "raw.tsv"
    path.write_text("\n".join(lines) + "\n")
    return path


def run(raw_file, out, *extra):
    prepare_lastfm.main(["--raw", str(raw_file), "--out", str(out),
                         "--min-item-plays", "2", "--min-user-events", "2", *extra])


def test_full_run_writes_everything(raw_file, tmp_path):
    out = tmp_path / "out"
    run(raw_file, out)
    stats = json.loads((out / "stats.json").read_text())
    assert [s["step"] for s in stats["steps"]] == ["raw", "cleaned", "up to 2009-06-30", "deduplicated", "filtered"]
    assert stats["steps"][-1] == {"step": "filtered", "events": 18, "users": 3, "items": 2}
    interactions = pd.read_parquet(out / "interactions.parquet")
    assert list(interactions.columns) == ["user_id", "item_id", "datetime", "weight"]
    assert set(interactions["item_id"]) == {"artist a - song a", "artist b - song b"}
    features = pd.read_parquet(out / "item_features.parquet")
    assert set(features["value"]) == {"artist a", "artist b"}


def test_rerun_picks_up_from_the_checkpoint(raw_file, tmp_path, monkeypatch):
    out = tmp_path / "out"
    run(raw_file, out)
    first = pd.read_parquet(out / "interactions.parquet")

    # If the raw file were read again this would blow up; with the checkpoints it isn't needed.
    def no_reading(_):
        raise AssertionError("raw file read again")

    monkeypatch.setattr(prepare_lastfm, "load_raw", no_reading)
    (out / "stats.json").unlink()
    run(raw_file, out)
    pd.testing.assert_frame_equal(pd.read_parquet(out / "interactions.parquet"), first)
    assert json.loads((out / "stats.json").read_text())["steps"][-1]["events"] == 18

    with pytest.raises(AssertionError, match="raw file read again"):
        run(raw_file, out, "--fresh")
