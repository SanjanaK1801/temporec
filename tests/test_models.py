import pandas as pd
import pytest
from rectools.models import BERT4RecModel, HSTUModel, SASRecModel
from rectools.models.nn.item_net import CatFeaturesItemNet, IdEmbeddingsItemNet

from models import BASE_CONFIG, build_model, chunk_histories, make_trainer_func


@pytest.fixture
def trainer_func(tmp_path):
    return make_trainer_func(str(tmp_path), "test_run", max_epochs=1, accelerator="cpu")


@pytest.mark.parametrize("name, cls", [
    ("sasrec", SASRecModel), ("bert4rec", BERT4RecModel), ("hstu", HSTUModel),
    ("hstu_large", HSTUModel), ("temporec", HSTUModel),
])
def test_build_model_class(name, cls, trainer_func):
    assert type(build_model(name, trainer_func)) is cls


def test_unknown_model_raises(trainer_func):
    with pytest.raises(ValueError):
        build_model("gru4rec", trainer_func)


def test_hstu_large_is_deeper(trainer_func):
    base, large = build_model("hstu", trainer_func), build_model("hstu_large", trainer_func)
    assert (base.n_blocks, base.n_heads) == (2, 1)
    assert (large.n_blocks, large.n_heads) == (8, 2)


def test_hstu_settings(trainer_func):
    m = build_model("hstu", trainer_func)
    assert m.relative_time_attention and m.relative_pos_attention
    assert m.lightning_module_kwargs == {"logits_t": 0.05}
    assert m.session_max_len == 200


def test_item_features_add_cat_item_net(trainer_func):
    assert build_model("hstu", trainer_func).item_net_block_types == (IdEmbeddingsItemNet,)
    m = build_model("temporec", trainer_func, use_item_features=True)
    assert m.item_net_block_types == (IdEmbeddingsItemNet, CatFeaturesItemNet)


def test_overrides_and_time_attention_ablation(trainer_func):
    m = build_model("hstu", trainer_func, session_max_len=50, relative_time_attention=False)
    assert m.session_max_len == 50
    assert not m.relative_time_attention
    assert "relative_time_attention" not in BASE_CONFIG  # the shared settings stay untouched


def history(user, n):
    return pd.DataFrame({
        "user_id": [user] * n,
        "item_id": [f"{user}_t{i}" for i in range(n)],
        "datetime": pd.date_range("2009-01-01", periods=n, freq="h"),
    })


def test_chunk_histories_newest_window_keeps_real_id():
    out = chunk_histories(history("u1", 5), chunk_len=2)
    assert list(out["user_id"]) == ["u1__c2", "u1__c1", "u1__c1", "u1", "u1"]
    assert list(out["item_id"]) == [f"u1_t{i}" for i in range(5)]  # order and listens unchanged


def test_chunk_histories_short_history_untouched():
    df = pd.concat([history("u1", 3), history("u2", 7)])
    out = chunk_histories(df, chunk_len=3)
    assert set(out.loc[out["item_id"].str.startswith("u1"), "user_id"]) == {"u1"}
    assert out["user_id"].value_counts().to_dict() == {"u2__c2": 1, "u1": 3, "u2": 3, "u2__c1": 3}


def test_chunk_histories_sorts_by_time_and_copies():
    df = history("u1", 4).iloc[::-1]  # newest first
    out = chunk_histories(df, chunk_len=2)
    assert list(out["item_id"]) == ["u1_t0", "u1_t1", "u1_t2", "u1_t3"]
    assert list(out["user_id"]) == ["u1__c1", "u1__c1", "u1", "u1"]
    assert set(df["user_id"]) == {"u1"}  # input frame not modified
