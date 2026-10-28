"""Model builders, history chunking and per-epoch logging (TR-05, models and training).

All transformer models come from RecTools (Apache 2.0). HSTU follows the RecTools HSTU tutorial,
which reproduces Zhai et al., ICML 2024:
https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_HSTU_tutorial.html
"""
import os

import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
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


def make_trainer_func(log_dir: str, run_name: str, max_epochs: int, patience: int = 5, accelerator: str = None):
    """Return a function that builds a Lightning Trainer.

    CSVLogger writes train_loss and val_loss for every epoch to
    <log_dir>/<run_name>/version_*/metrics.csv (used by plot_curves.py).
    Training stops when val_loss has not improved for `patience` epochs.
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
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=patience)],
        )

    return get_trainer


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
