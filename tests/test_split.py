import pandas as pd

from split_lastfm import first_time_share, time_split


def toy():
    # 20 listens one day apart, so the 80% and 90% points fall on day 15 and day 17.
    t = pd.Timestamp("2009-01-01")
    users = ["u1", "u2"] * 8 + ["u1", "u3_only_late", "u1", "u2"]
    items = ["a", "b"] * 8 + ["b", "a", "new_track", "a"]
    return pd.DataFrame({
        "user_id": users,
        "item_id": items,
        "datetime": [t + pd.Timedelta(days=d) for d in range(20)],
    })


def test_cutoffs_at_80_and_90_percent():
    _, _, _, (c1, c2) = time_split(toy(), val_frac=0.1, test_frac=0.1)
    assert c1 == pd.Timestamp("2009-01-16 04:48:00")  # np.quantile interpolates between day 15 and 16
    assert c2 == pd.Timestamp("2009-01-18 02:24:00")


def test_parts_dont_overlap_in_time():
    train, val, test, (c1, c2) = time_split(toy(), 0.1, 0.1)
    assert (train["datetime"] < c1).all()
    assert val["datetime"].between(c1, c2, inclusive="left").all()
    assert (test["datetime"] >= c2).all()
    assert len(train) == 16


def test_unknown_users_and_items_dropped():
    train, val, test, _ = time_split(toy(), 0.1, 0.1)
    assert list(val["item_id"]) == ["b"]  # u3 has no train listens, so only u1's listen stays
    assert set(test["user_id"]) <= set(train["user_id"])
    assert list(test["item_id"]) == ["a"]  # new_track never shows up in train


def test_first_time_share():
    history = pd.DataFrame({"user_id": ["u1"], "item_id": ["a"]})
    part = pd.DataFrame({"user_id": ["u1", "u1", "u2", "u1"], "item_id": ["a", "b", "a", "a"]})
    assert first_time_share(part, history) == 0.5
    assert first_time_share(part.iloc[:0], history) == 0.0
