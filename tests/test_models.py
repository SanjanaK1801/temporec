from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from rectools.dataset import Dataset
from rectools.models import BERT4RecModel, HSTUModel, SASRecModel
from rectools.models.nn.item_net import CatFeaturesItemNet, IdEmbeddingsItemNet

from models import BASE_CONFIG, ValHitRate, build_model, chunk_histories, load_best_weights, make_trainer_func


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


def test_trainer_stops_and_checkpoints_on_val_hr(trainer_func):
    callbacks = trainer_func().callbacks
    stop = next(c for c in callbacks if isinstance(c, EarlyStopping))
    best = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    assert (stop.monitor, stop.mode, stop.patience) == ("val_hr@10", "max", 10)
    assert (best.monitor, best.mode, best.save_last) == ("val_hr@10", "max", True)


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


def test_load_best_weights_restores_highest_val_hr_epoch(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "user_id": np.repeat(np.arange(30), 8),
        "item_id": rng.integers(0, 20, 240),
        "weight": 1.0,
        "datetime": np.tile(pd.date_range("2009-01-01", periods=8, freq="h"), 30),
    })
    trainer_func = make_trainer_func(str(tmp_path), "tiny", max_epochs=4, patience=10, accelerator="cpu")
    model = build_model("sasrec", trainer_func, n_factors=8, batch_size=16, session_max_len=8, n_negatives=4)
    model.fit(Dataset.construct(df))

    best_epoch = load_best_weights(model)
    log = pd.read_csv(next(tmp_path.glob("tiny/version_*/metrics.csv"))).dropna(subset=["val_hr@10"])
    assert len(log) == 4 and log["val_hr@10"].between(0, 1).all()
    assert best_epoch == log.loc[log["val_hr@10"].idxmax(), "epoch"] + 1  # first of any ties, like ModelCheckpoint
    best = torch.load(model.fit_trainer.checkpoint_callback.best_model_path, weights_only=False)["state_dict"]
    assert all(torch.equal(v, best[k]) for k, v in model.lightning_model.state_dict().items())


class FakeModule:
    """Stands in for the Lightning module: item i scores `scores[i]` for every user."""

    def __init__(self, scores, n_extra_tokens=1):
        self.item_embs = torch.tensor(scores, dtype=torch.float).unsqueeze(1)  # [n_items, 1]
        self.torch_model = SimpleNamespace(
            encode_sessions=lambda batch, embs: torch.ones(len(batch["x"]), 1, 1),
            similarity_module=lambda sessions, embs: sessions @ embs.T,
        )
        self.data_preparator = SimpleNamespace(n_item_extra_tokens=n_extra_tokens)
        self.logged = {}

    def log(self, name, value, on_epoch):
        self.logged[name] = float(value)


def test_val_hit_rate_by_hand():
    # 12 items, item 0 = padding with the highest score; real items ranked 11, 10, ..., 1
    module = FakeModule([99.0] + list(range(1, 12)))
    batch = {
        "x": torch.tensor([[0, 11], [0, 5], [0, 0]]),  # histories (0 = padding)
        "y": torch.tensor([[2], [1], [2]]),
    }
    cb = ValHitRate(k=10, filter_viewed=True)
    cb.on_validation_batch_end(None, module, None, batch, 0)
    cb.on_validation_epoch_end(None, module)
    # user 0 saw item 11, so its top 10 is 10..1: hit on 2. user 1 saw 5: top 10 is 11..6, 4..1 -> hit on 1.
    # user 2 saw nothing: top 10 is 11..2 -> hit on 2. Padding is never recommended.
    assert module.logged["val_hr@10"] == 1.0

    cb = ValHitRate(k=10, filter_viewed=False)
    cb.on_validation_batch_end(None, module, None, batch, 0)
    cb.on_validation_epoch_end(None, module)
    assert module.logged["val_hr@10"] == pytest.approx(2 / 3)  # top 10 is 11..2 for everyone: user 1 misses
