import pandas as pd
import pytest

from checkpoint import StageCheckpoints, atomic_write, write_json

TABLE = pd.DataFrame({"user_id": ["u1", "u2"], "datetime": pd.to_datetime(["2009-01-01", "2009-01-02"])})


def test_failed_write_keeps_the_old_file(tmp_path):
    target = tmp_path / "stats.json"
    write_json({"events": 1}, target)

    def crash(tmp):
        tmp.write_text("half a fil")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        atomic_write(target, crash)
    assert target.read_text() == '{\n  "events": 1\n}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["stats.json"]  # no temp file left behind


def test_saved_step_comes_back(tmp_path):
    ckpt = StageCheckpoints(tmp_path)
    ckpt.save("cleaned", TABLE, {"min_plays": 5}, {"events": 2})
    table, info = ckpt.load("cleaned", {"min_plays": 5})
    pd.testing.assert_frame_equal(table, TABLE)
    assert info == {"events": 2}


def test_changed_settings_or_no_resume_means_recompute(tmp_path):
    StageCheckpoints(tmp_path).save("cleaned", TABLE, {"min_plays": 5})
    assert StageCheckpoints(tmp_path).load("cleaned", {"min_plays": 10}) is None
    assert StageCheckpoints(tmp_path, resume=False).load("cleaned", {"min_plays": 5}) is None
    assert StageCheckpoints(tmp_path).load("filtered", {"min_plays": 5}) is None
