"""The two simple baselines every model has to beat (TR-04, data and eval).

Both return user_id, item_id, rank, like the models do.
"""
import pandas as pd
from rectools import Columns


def most_popular(train: pd.DataFrame, users, k: int = 10) -> pd.DataFrame:
    """The k most-played tracks overall, the same list for every user."""
    top = train[Columns.Item].value_counts().head(k).index
    reco = pd.DataFrame(
        [(u, item, r + 1) for u in users for r, item in enumerate(top)],
        columns=[Columns.User, Columns.Item, Columns.Rank],
    )
    return reco


def recently_played(train: pd.DataFrame, users, k: int = 10) -> pd.DataFrame:
    """The k most recently played distinct tracks of each user (strong on replays, useless for discovery)."""
    last = (
        train[train[Columns.User].isin(users)]
        .sort_values(Columns.Datetime, ascending=False, kind="stable")
        .drop_duplicates([Columns.User, Columns.Item])
    )
    reco = last.groupby(Columns.User).head(k).copy()
    reco[Columns.Rank] = reco.groupby(Columns.User).cumcount() + 1
    return reco[[Columns.User, Columns.Item, Columns.Rank]].reset_index(drop=True)
