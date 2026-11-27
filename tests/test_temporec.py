import math

import numpy as np
import pandas as pd
import pytest
import torch
from rectools.dataset import Dataset

from models import build_model, make_trainer_func
from temporec import (TempoRecBackbone, WindowNegativeSampler, cyclical_time_features, recency_weights,
                      score_next)

DAY, WEEK = 86400, 7 * 86400


def test_time_features_are_periodic():
    ts = torch.tensor([1_233_269_081, 1_108_339_207, 0])
    feats = cyclical_time_features(ts)
    assert feats.shape == (3, 4)
    assert torch.allclose(feats, cyclical_time_features(ts + WEEK), atol=1e-5)
    assert torch.allclose(feats[:, :2], cyclical_time_features(ts + DAY)[:, :2], atol=1e-5)  # hour of day
    assert not torch.allclose(feats[:, 2:], cyclical_time_features(ts + DAY)[:, 2:], atol=1e-3)  # day of week


def test_time_features_known_values():
    # 1970-01-01 00:00 UTC was a Thursday: hour angle 0, week angle 3/7 of a turn
    feats = cyclical_time_features(torch.tensor([0, 6 * 3600]))
    week = 2 * math.pi * 3 / 7
    assert torch.allclose(feats[0], torch.tensor([0.0, 1.0, math.sin(week), math.cos(week)]), atol=1e-6)
    assert torch.allclose(feats[1, :2], torch.tensor([1.0, 0.0]), atol=1e-6)  # 06:00 is a quarter day


def test_recency_weights_decay_and_padding():
    play = torch.tensor([0, 0, 3600 * 9, 3600 * 10])
    items = torch.tensor([0, 5, 5, 6])
    w = recency_weights(play, torch.tensor(3600 * 10), items, torch.tensor(0.5))
    assert w[0] == 0  # padding never counts
    assert torch.allclose(w[1:], torch.tensor([11.0, 2.0, 1.0]) ** -0.5)
    assert w[3] > w[2] > w[1]  # the more recent the play, the more it counts


def listens(n_users=30, n_events=24, n_items=20, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2008-01-01")
    rows = []
    for u in range(n_users):
        favourite = rng.integers(1, n_items + 1, 3)
        t = start + pd.Timedelta(hours=int(rng.integers(0, 500)))
        for _ in range(n_events):
            t += pd.Timedelta(minutes=int(rng.integers(3, 600)))
            item = favourite[rng.integers(0, 3)] if rng.random() < 0.6 else rng.integers(1, n_items + 1)
            rows.append((f"u{u}", f"t{item}", 1.0, t))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "weight", "datetime"])


def fit(tmp_path, time_embeddings, repeat_gate, epochs=2):
    trainer_func = make_trainer_func(str(tmp_path), f"t{int(time_embeddings)}g{int(repeat_gate)}", max_epochs=epochs,
                                     patience=10, accelerator="cpu", filter_viewed=False)
    model = build_model("hstu", trainer_func, time_embeddings=time_embeddings, repeat_gate=repeat_gate,
                        gate_window=5, n_factors=8, session_max_len=10, n_negatives=6, batch_size=16)
    model.fit(Dataset.construct(listens()))
    return model


@pytest.mark.parametrize("time_embeddings, repeat_gate", [(True, False), (False, True), (True, True)])
def test_training_works_for_each_combination(tmp_path, time_embeddings, repeat_gate):
    model = fit(tmp_path, time_embeddings, repeat_gate)
    backbone = model.lightning_model.torch_model
    assert isinstance(backbone, TempoRecBackbone)
    assert (backbone.time_embeddings, backbone.repeat_gate) == (time_embeddings, repeat_gate)
    log = pd.read_csv(next(tmp_path.glob("*/version_*/metrics.csv")))
    assert np.isfinite(log["train_loss"].dropna()).all() and len(log["train_loss"].dropna()) == 2
    assert log["val_hr@10"].dropna().between(0, 1).all()  # validation ran through the gate-aware scoring


def test_plain_hstu_has_no_tempo_rec_parts(tmp_path):
    trainer_func = make_trainer_func(str(tmp_path), "plain", max_epochs=1, accelerator="cpu")
    model = build_model("hstu", trainer_func, n_factors=8, session_max_len=10, n_negatives=6, batch_size=16)
    assert model.backbone_type is not TempoRecBackbone
    with pytest.raises(ValueError):
        build_model("sasrec", trainer_func, repeat_gate=True)


@pytest.fixture(scope="module")
def full_backbone(tmp_path_factory):
    model = fit(tmp_path_factory.mktemp("full"), True, True)
    backbone = model.lightning_model.torch_model.eval()
    with torch.no_grad():  # a clearly non-zero bonus, so the checks below are not trivially true
        backbone.log_scale.fill_(1.0)
        backbone.gate.bias.fill_(2.0)
    return backbone


def toy_batch(n_items, batch_size=4, max_len=10, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, n_items, (batch_size, max_len), generator=g)
    x[0, :3] = 0  # one listener with a short, padded history
    gaps = torch.randint(60, 20_000, (batch_size, max_len + 1), generator=g)
    ts = 1_200_000_000 + gaps.cumsum(dim=1)
    return {"x": x, "unix_ts": ts}


def test_inference_scores_equal_training_formula(full_backbone):
    n_items = full_backbone.item_model.get_all_embeddings().shape[0]
    batch = toy_batch(n_items)
    candidates = torch.cat([batch["x"][:, -6:], torch.randint(1, n_items, (4, 6))], dim=1)
    candidates = candidates.unsqueeze(1).expand(-1, batch["x"].shape[1], -1)  # same candidates at every position
    with torch.no_grad():
        train_logits = full_backbone(batch, candidates)[:, -1, :]
        scores, gate = score_next(full_backbone, batch)
    assert torch.allclose(scores.gather(1, candidates[:, -1, :]), train_logits, atol=1e-5)
    assert gate.shape == (4,) and ((gate > 0) & (gate < 1)).all()


def test_tracks_never_played_get_no_bonus(full_backbone):
    n_items = full_backbone.item_model.get_all_embeddings().shape[0]
    batch = toy_batch(n_items)
    with torch.no_grad():
        scores, _ = full_backbone.score_next(batch)
        state = full_backbone.encode_sessions(batch, full_backbone.item_model.get_all_embeddings())[:, -1:, :]
        plain = full_backbone.similarity_module(state, full_backbone.item_model.get_all_embeddings())[:, 0, :]
    window = batch["x"][:, -full_backbone.gate_window:]
    played = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, window, True)
    played[:, 0] = False  # padding
    assert torch.equal(scores[~played], plain[~played])
    assert (scores[played] > plain[played]).all()


def test_window_negatives_come_from_recent_plays_and_never_the_positive():
    torch.manual_seed(0)
    x = torch.tensor([[0, 0, 5, 6, 7, 5], [9, 9, 9, 9, 9, 9]])  # listener 1 only ever plays id 9
    y = torch.tensor([[0, 5, 6, 7, 5, 8], [9, 9, 9, 9, 9, 9]])
    sampler = WindowNegativeSampler(n_negatives=10, window=3, n_window=4)
    neg = sampler.get_negatives({"x": x, "y": y}, lowest_id=1, highest_id=9)  # uniform draws are 1..8
    assert neg.shape == (2, 6, 10)
    assert set(neg[0, -1, :4].tolist()) <= {5, 6, 7}  # last window: 6, 7, 5; the positive is 8
    assert (neg[1, :, :4] != 9).all()  # the only recent play is the positive itself: uniform draws instead
    assert (neg >= 1).all()  # never padding


def test_window_negatives_in_validation_use_the_last_window():
    torch.manual_seed(0)
    x = torch.tensor([[11, 12, 13, 14]])
    sampler = WindowNegativeSampler(n_negatives=6, window=2, n_window=6)
    neg = sampler.get_negatives({"x": x, "y": torch.tensor([[13]])}, lowest_id=1, highest_id=11,
                                session_len_limit=1)
    assert neg.shape == (1, 1, 6)
    values = neg[0, 0].tolist()
    assert 13 not in values  # in the window, but it is the true next track
    assert 14 in values and set(values) <= {14} | set(range(1, 11))


def test_build_model_with_window_negatives(tmp_path):
    trainer_func = make_trainer_func(str(tmp_path), "w", max_epochs=1, accelerator="cpu")
    model = build_model("hstu", trainer_func, repeat_gate=True, gate_window=7, window_negatives=3)
    assert model.negative_sampler_type is WindowNegativeSampler
    assert model.negative_sampler_kwargs == {"window": 7, "n_window": 3}
    with pytest.raises(ValueError):
        build_model("sasrec", trainer_func, window_negatives=3)
