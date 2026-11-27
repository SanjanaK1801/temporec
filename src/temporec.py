"""TempoRec's own components on top of RecTools' HSTU (TR-08, models and training).

Two parts, each switched on or off by a flag:

- Cyclical time embeddings. Each listen's hour of day and day of week become sin/cos features
  (4 numbers), are projected to the embedding size and added to the track's input embedding. The
  time of the item being predicted (the request time at inference) is added to the output of the
  HSTU layers the same way. HSTU on its own only sees time differences between listens.
- Repeat-or-explore gate. For every candidate track, its memory activation in the listener's last
  W plays, in the spirit of ACT-R base-level activation as used by PISA (Tran et al., RecSys 2024):
  the sum over its recent plays of (hours since that play + 1) ** -d, with the decay d learned.
  A gate g in [0, 1] comes from the listener's current hidden state, and
      score = similarity(state, track) + g * scale * log(1 + activation)
  with the scale learned. There is still one score per track, so RecTools' sampled-softmax training
  works unchanged. The bonus needs a sampled loss (sampled_softmax, bce, gbce), not full softmax.

Times are unix seconds in UTC. Batches must carry `unix_ts` (data_preparator_kwargs
{"add_unix_ts": True}), which has session_max_len + 1 columns: column k is the time of input item
k and column k + 1 the time of the item predicted at position k; at inference the last column is
the request time.

RecTools' recommend() ranks with a plain dot product, so it cannot add a per-listener bonus.
`score_next` scores the whole catalog for the next position instead (the bonus is scattered onto the
few tracks in the window); the next-track evaluation and the validation callback use it.

Window negatives: with uniform negatives over ~500K tracks, a negative is almost never one of the
listener's recent plays, so the bonus only ever helps the true track and the gate never learns to
switch off before a new track. WindowNegativeSampler draws some negatives from the listener's own
recent plays (never the true next track), so boosting a recent track can also be wrong in training.
"""
import math
import typing as tp

import torch
from rectools.models.nn.transformers.negative_sampler import CatalogUniformSampler
from rectools.models.nn.transformers.torch_backbone import TransformerTorchBackbone
from torch import nn

SECONDS_PER_DAY = 86400
EPOCH_WEEKDAY = 3  # 1970-01-01 was a Thursday; Monday = 0


def cyclical_time_features(unix_ts: torch.Tensor) -> torch.Tensor:
    """sin and cos of the hour of day and of the day of week: [...] int seconds -> [..., 4] floats."""
    unix_ts = unix_ts.long()  # whole seconds; float32 can't hold 2009-era timestamps to the second
    second_of_day = torch.remainder(unix_ts, SECONDS_PER_DAY).float()
    day_of_week = torch.remainder(torch.div(unix_ts, SECONDS_PER_DAY, rounding_mode="floor") + EPOCH_WEEKDAY, 7)
    day_angle = 2 * math.pi * second_of_day / SECONDS_PER_DAY
    week_angle = 2 * math.pi * (day_of_week.float() + second_of_day / SECONDS_PER_DAY) / 7
    return torch.stack([day_angle.sin(), day_angle.cos(), week_angle.sin(), week_angle.cos()], dim=-1)


def recency_weights(play_ts: torch.Tensor, request_ts: torch.Tensor, items: torch.Tensor, decay: torch.Tensor):
    """(hours since the play + 1) ** -decay for every play in a window; 0 for padding (item 0)."""
    hours = (request_ts - play_ts).clamp(min=0).float() / 3600
    return (hours + 1).pow(-decay) * (items != 0)


class WindowNegativeSampler(CatalogUniformSampler):
    """Uniform negatives, except the first `n_window` at each position, which are drawn from the
    listener's last `window` plays at that position. A draw that is padding or the true next track
    keeps its uniform negative instead."""

    def __init__(self, n_negatives: int, window: int = 50, n_window: int = 16, **kwargs: tp.Any) -> None:
        super().__init__(n_negatives=n_negatives, **kwargs)
        self.window, self.n_window = window, min(n_window, n_negatives)

    def get_negatives(self, batch_dict: tp.Dict, lowest_id: int, highest_id: int,
                      session_len_limit: tp.Optional[int] = None, **kwargs: tp.Any) -> torch.Tensor:
        negatives = super().get_negatives(batch_dict, lowest_id, highest_id, session_len_limit, **kwargs)
        if self.n_window == 0:
            return negatives
        x, y = batch_dict["x"], batch_dict["y"]  # x [batch, L]; y [batch, L] in training, [batch, 1] in validation
        window = min(self.window, x.shape[1])
        win_items = nn.functional.pad(x, (window - 1, 0)).unfold(1, window, 1)[:, -negatives.shape[1]:]
        pick = torch.randint(0, window, (*win_items.shape[:2], self.n_window))
        drawn = win_items.gather(-1, pick)  # [batch, positions, n_window]
        usable = (drawn >= lowest_id) & (drawn != y[:, -negatives.shape[1]:].unsqueeze(-1))
        negatives[..., : self.n_window] = torch.where(usable, drawn, negatives[..., : self.n_window])
        return negatives


class TempoRecBackbone(TransformerTorchBackbone):
    """RecTools transformer backbone with cyclical time embeddings and a repeat-or-explore gate."""

    def __init__(
        self,
        *args: tp.Any,
        n_factors: int,
        time_embeddings: bool = True,
        repeat_gate: bool = True,
        gate_window: int = 50,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.time_embeddings = time_embeddings
        self.repeat_gate = repeat_gate
        self.gate_window = gate_window
        if time_embeddings:
            self.input_time = nn.Linear(4, n_factors)
            self.target_time = nn.Linear(4, n_factors)
        if repeat_gate:
            self.gate = nn.Linear(n_factors, 1)
            self.log_scale = nn.Parameter(torch.zeros(()))
            self.raw_decay = nn.Parameter(torch.tensor(math.log(math.expm1(0.5))))  # softplus -> d = 0.5 at start

    @property
    def decay(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_decay)

    def encode_sessions(self, batch: tp.Dict[str, torch.Tensor], item_embs: torch.Tensor) -> torch.Tensor:
        """RecTools' encoder with the time of each listen added before the HSTU layers and the
        time of the predicted item added after them."""
        sessions = batch["x"]  # [batch, L]
        session_max_len = sessions.shape[1]
        timeline_mask = (sessions != 0).unsqueeze(-1)

        seqs = item_embs[sessions]
        seqs = self.pos_encoding_layer(seqs)
        if self.time_embeddings:
            seqs = seqs + self.input_time(cyclical_time_features(batch["unix_ts"][:, :-1]))
        seqs = self.emb_dropout(seqs)

        attn_mask, key_padding_mask = None, None
        if self.use_causal_attn:
            attn_mask = ~torch.tril(
                torch.ones((session_max_len, session_max_len), dtype=torch.bool, device=sessions.device)
            )
        if self.use_key_padding_mask:
            key_padding_mask = sessions == 0
            if attn_mask is not None:
                attn_mask = self._merge_masks(attn_mask, key_padding_mask, seqs)
                key_padding_mask = None

        seqs = self.transformer_layers(seqs, timeline_mask, attn_mask, key_padding_mask, batch=batch)
        if self.time_embeddings:
            seqs = seqs + self.target_time(cyclical_time_features(batch["unix_ts"][:, 1:])) * timeline_mask
        return seqs

    def gate_value(self, states: torch.Tensor) -> torch.Tensor:
        """g in [0, 1] from the hidden state: [..., n_factors] -> [..., 1]."""
        return torch.sigmoid(self.gate(states))

    def repeat_bonus(
        self, batch: tp.Dict[str, torch.Tensor], candidates: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        """g * scale * log(1 + activation) for each candidate at each position.

        candidates: [batch, L or 1, n_candidates]; states: [batch, L, n_factors] -> [batch, L, n_candidates].
        The window for position k is the plays x[k - W + 1 .. k], and the request time is the time
        of the item predicted at k. Memory is batch x L x n_candidates (one window slot at a time).
        """
        x, ts = batch["x"], batch["unix_ts"]
        window = min(self.gate_window, x.shape[1])
        win_items = nn.functional.pad(x, (window - 1, 0)).unfold(1, window, 1)  # [batch, L, W]
        win_ts = nn.functional.pad(ts[:, :-1], (window - 1, 0)).unfold(1, window, 1)
        weights = recency_weights(win_ts, ts[:, 1:].unsqueeze(-1), win_items, self.decay)  # [batch, L, W]

        activation = torch.zeros(
            (x.shape[0], x.shape[1], candidates.shape[-1]), device=x.device, dtype=weights.dtype
        )
        for j in range(window):
            activation = activation + (candidates == win_items[..., j : j + 1]) * weights[..., j : j + 1]
        return self.gate_value(states) * self.log_scale.exp() * torch.log1p(activation)

    def forward(
        self, batch: tp.Dict[str, torch.Tensor], candidate_item_ids: tp.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        item_embs = self.item_model.get_all_embeddings()
        session_embs = self.encode_sessions(batch, item_embs)
        logits = self.similarity_module(session_embs, item_embs, candidate_item_ids)
        if self.repeat_gate:
            if candidate_item_ids is None:
                raise ValueError("the repeat-or-explore gate needs a sampled loss (sampled_softmax, bce or gbce)")
            logits = logits + self.repeat_bonus(batch, candidate_item_ids, session_embs)
        return logits

    def score_next(
        self, batch: tp.Dict[str, torch.Tensor], item_embs: tp.Optional[torch.Tensor] = None
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Scores of every item for the next position, and the gate value (None without the gate).

        Uses the same formula as training at the last position: [batch, n_items], [batch] or None.
        """
        if item_embs is None:
            item_embs = self.item_model.get_all_embeddings()
        state = self.encode_sessions(batch, item_embs)[:, -1:, :]  # [batch, 1, n_factors]
        scores = self.similarity_module(state, item_embs)[:, 0, :]  # [batch, n_items]
        if not self.repeat_gate:
            return scores, None

        x, ts = batch["x"], batch["unix_ts"]
        window = min(self.gate_window, x.shape[1])
        win_items = x[:, -window:]
        weights = recency_weights(ts[:, -window - 1 : -1], ts[:, -1:], win_items, self.decay)
        activation = torch.zeros_like(scores).scatter_add_(1, win_items, weights)
        gate = self.gate_value(state[:, 0, :])  # [batch, 1]
        return scores + gate * self.log_scale.exp() * torch.log1p(activation), gate[:, 0]


def score_next(backbone: nn.Module, batch: tp.Dict[str, torch.Tensor], item_embs: tp.Optional[torch.Tensor] = None):
    """Next-position scores for any RecTools SASRec/HSTU-style backbone (TempoRec adds its bonus)."""
    if isinstance(backbone, TempoRecBackbone):
        return backbone.score_next(batch, item_embs)
    if item_embs is None:
        item_embs = backbone.item_model.get_all_embeddings()
    state = backbone.encode_sessions(batch, item_embs)[:, -1:, :]
    return backbone.similarity_module(state, item_embs)[:, 0, :], None
