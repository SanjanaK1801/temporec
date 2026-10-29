"""Model builders, history chunking and per-epoch logging (TR-05).

All transformer models come from RecTools (Apache 2.0). HSTU follows the RecTools HSTU tutorial,
which reproduces Zhai et al., ICML 2024:
https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_HSTU_tutorial.html
"""
import os

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rectools import Columns
from rectools.models import BERT4RecModel, HSTUModel, SASRecModel
from rectools.models.nn.item_net import CatFeaturesItemNet, IdEmbeddingsItemNet
from rectools.models.nn.transformers.utils import leave_one_out_mask

TRANSFORMERS = ("sasrec", "bert4rec", "hstu", "hstu_large", "temporec")

# Settings from the RecTools HSTU tutorial (they reproduce the HSTU paper on MovieLens-1M).
BASE_CONFIG = {
    "session_max_len": 200,
    "loss": "sampled_softmax",
    "n_negatives": 128,
    "use_pos_emb": True,
    "dropout_rate": 0.2,
    "n_factors": 50,
    "n_heads": 1,
    "n_blocks": 2,
    "lr": 0.001,
    "batch_size": 128,
    "get_val_mask_func": leave_one_out_mask,  # validation = each user's last training item
    "verbose": 1,
}


def pick_accelerator() -> str:
    if os.environ.get("TEMPOREC_CPU"):
        return "cpu"
    if torch.cuda.is_available():
        return "gpu"  # Zaratan A100
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon; set TEMPOREC_CPU=1 to force CPU if MPS errors
    return "cpu"


VAL_METRIC = "val_hr@10"


class ValHitRate(Callback):
    """HR@k on the validation targets after every epoch, logged as val_hr@10.

    Adapted from RecallCallback in the RecTools tutorials (Apache 2.0). The validation targets are
    the ones get_val_mask_func picks (each user's last training item), so with one target per user
    this is the same number as recall@k. filter_viewed drops items already in the user's history
    from the ranking, like recommend(filter_viewed=True); leave it off when replays count.

    The model only sees each user's last session_max_len items, so only those are filtered. For
    users with longer histories this can come out a little below recommend(), which filters the
    whole history (on MovieLens-1M after 1 epoch: 0.0348 vs 0.0404). Every model is measured the
    same way, and early stopping only compares epochs within one run.
    """

    def __init__(self, k: int = 10, filter_viewed: bool = True):
        self.k, self.filter_viewed = k, filter_viewed
        self.hits = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        model = pl_module.torch_model
        sessions = model.encode_sessions(batch, pl_module.item_embs)[:, -1, :]
        scores = model.similarity_module(sessions, pl_module.item_embs)  # [batch, n_items]
        if self.filter_viewed:
            scores = scores.scatter(1, batch["x"], float("-inf"))
        scores[:, :pl_module.data_preparator.n_item_extra_tokens] = float("-inf")  # padding (and BERT4Rec's mask)
        top = scores.topk(self.k).indices  # [batch, k]
        targets = batch["y"]  # [batch, n_targets], 0 = no target
        hit = (top.unsqueeze(-1) == targets.unsqueeze(1)).any(dim=(1, 2))
        self.hits.append(hit.float())

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.hits:
            pl_module.log(VAL_METRIC, torch.cat(self.hits).mean(), on_epoch=True)
        self.hits.clear()


def make_trainer_func(log_dir: str, run_name: str, max_epochs: int, patience: int = 10, accelerator: str = None,
                      filter_viewed: bool = True):
    """Return a function that builds a Lightning Trainer.

    CSVLogger writes train_loss, val_loss and val_hr@10 for every epoch to
    <log_dir>/<run_name>/version_*/metrics.csv (used by plot_curves.py).
    Training stops when val_hr@10 has not improved for `patience` epochs (the proposal's rule).
    The checkpoints/ folder next to metrics.csv keeps the weights from the epoch with the highest
    val_hr@10 (best.ckpt) and from the latest epoch (last.ckpt); load_best_weights() swaps the
    best ones back in. filter_viewed goes to ValHitRate.
    """
    accelerator = accelerator or pick_accelerator()

    def get_trainer() -> Trainer:
        return Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=max_epochs,
            min_epochs=min(5, max_epochs),
            deterministic=accelerator != "mps",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=CSVLogger(log_dir, name=run_name),
            callbacks=[
                ValHitRate(k=10, filter_viewed=filter_viewed),
                EarlyStopping(monitor=VAL_METRIC, mode="max", patience=patience),
                ModelCheckpoint(monitor=VAL_METRIC, mode="max", filename="best", save_top_k=1, save_last=True),
            ],
        )

    return get_trainer


def load_best_weights(model) -> int:
    """After fit, replace the last epoch's weights with the best epoch's. Returns that epoch (1-based)."""
    path = model.fit_trainer.checkpoint_callback.best_model_path
    model.load_weights_from_checkpoint(path)
    return torch.load(path, weights_only=False)["epoch"] + 1


def build_model(name: str, trainer_func, use_item_features: bool = False, **overrides):
    """Build one of: sasrec, bert4rec, hstu, hstu_large, temporec.

    temporec = HSTU whose item embedding also includes categorical item features
    (artist on Last.fm, genre on MovieLens). Pass relative_time_attention=False for the ablation.
    """
    if name not in TRANSFORMERS:
        raise ValueError(f"unknown model {name}")
    cfg = {**BASE_CONFIG, **overrides, "get_trainer_func": trainer_func}
    time_attention = cfg.pop("relative_time_attention", True)
    cfg["item_net_block_types"] = (
        (IdEmbeddingsItemNet, CatFeaturesItemNet) if use_item_features else (IdEmbeddingsItemNet,)
    )
    if name == "sasrec":
        return SASRecModel(**cfg)
    if name == "bert4rec":
        return BERT4RecModel(**cfg)
    if name == "hstu_large":
        cfg.update(n_blocks=8, n_heads=2)
    cfg["lightning_module_kwargs"] = {"logits_t": 0.05}  # as in the official HSTU code
    return HSTUModel(relative_time_attention=time_attention, relative_pos_attention=True, **cfg)


def chunk_histories(df: pd.DataFrame, chunk_len: int) -> pd.DataFrame:
    """Split long user histories into windows of `chunk_len` events.

    Transformers train only on each user's last `session_max_len` events, and Last.fm 1K users
    have thousands. The newest window keeps the real user ID (we recommend for it); older windows
    get pseudo-user IDs like "user_000001__c3" and are used only for training.
    """
    df = df.sort_values([Columns.User, Columns.Datetime], kind="stable").copy()
    chunk = df.groupby(Columns.User).cumcount(ascending=False) // chunk_len
    user = df[Columns.User].astype(str)
    df[Columns.User] = user.where(chunk == 0, user + "__c" + chunk.astype(str))
    return df
