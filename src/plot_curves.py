"""Per-epoch training and validation loss plots for every run in logs/lightning (TR-09, data and eval).

Usage:
    python src/plot_curves.py --logs logs/lightning --out figures
Makes one figure per run plus a combined validation-loss figure per dataset prefix.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_run(run_dir: Path) -> pd.DataFrame:
    """Read the newest version's metrics.csv and collapse it to one row per epoch."""
    versions = sorted(run_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
    df = pd.read_csv(versions[-1] / "metrics.csv")
    loss_cols = [c for c in df.columns if "loss" in c]
    return df.groupby("epoch")[loss_cols].mean().reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs/lightning")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = {d.name: load_run(d) for d in sorted(Path(args.logs).iterdir()) if d.is_dir() and any(d.glob("version_*"))}
    for name, df in runs.items():
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for col in df.columns.drop("epoch"):
            ax.plot(df["epoch"] + 1, df[col], marker="o", ms=3, label=col)
        ax.set(xlabel="Epoch", ylabel="Loss", title=name)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"curve_{name}.png", dpi=150)
        plt.close(fig)

    for prefix in sorted({n.split("_")[0] for n in runs}):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for name, df in runs.items():
            if name.startswith(prefix) and "val_loss" in df:
                ax.plot(df["epoch"] + 1, df["val_loss"], label=name.removeprefix(prefix + "_"))
        ax.set(xlabel="Epoch", ylabel="Validation loss", title=f"{prefix}: validation loss by epoch")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"val_loss_{prefix}.png", dpi=150)
        plt.close(fig)
    print(f"wrote {len(runs)} run figures to {out}")


if __name__ == "__main__":
    main()
