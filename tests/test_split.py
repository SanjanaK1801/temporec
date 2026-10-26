import pandas as pd

from split_lastfm import time_split


def toy():
    t = pd.Timestamp("2009-01-01")
    rows = [
        ("u1", "a", t), ("u1", "b", t + pd.Timedelta(days=10)), ("u1", "a", t + pd.Timedelta(days=50)),
        ("u2", "b", t + pd.Timedelta(days=5)), ("u2", "new_track", t + pd.Timedelta(days=55)),
        ("u3_only_in_test", "a", t + pd.Timedelta(days=58)),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "datetime"])


def test_no_test_event_before_cutoff_and_no_train_event_after():
    train, test, cutoff = time_split(toy(), test_days=30)
    assert (train["datetime"] < cutoff).all()
    assert (test["datetime"] >= cutoff).all()


def test_unknown_users_and_items_dropped_from_test():
    train, test, _ = time_split(toy(), test_days=30)
    assert set(test["user_id"]) <= set(train["user_id"])
    assert set(test["item_id"]) <= set(train["item_id"])
    assert list(test["item_id"]) == ["a"]  # u1's replay; new_track and u3 are dropped
