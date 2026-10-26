"""Dataset statistics table and figures for the report (TR-03, data and eval).

Usage:
    python src/dataset_figures.py --data data/processed/lastfm --out figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/lastfm")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path(args.data, "stats.json").read_text())
    print("## Data preparation steps\n")
    print(pd.DataFrame(stats["steps"]).to_markdown(index=False))
    print(f"\nDate range: {stats['first_event']} to {stats['last_event']}")
    print(f"Artists: {stats['artists']:,}; median events per user: {stats['events_per_user_median']:,.0f}; "
          f"repeat share: {stats['repeat_share']:.1%}")

    df = pd.read_parquet(Path(args.data, "interactions.parquet"))

    # Events per user (log scale: a few heavy listeners dominate).
    fig, ax = plt.subplots(figsize=(5, 3.2))
    df.groupby("user_id").size().plot.hist(bins=50, log=True, ax=ax)
    ax.set(xlabel="Listening events per user", ylabel="Users (log scale)")
    fig.tight_layout()
    fig.savefig(out / "events_per_user.png", dpi=150)
    plt.close(fig)

    # Plays per track, sorted: the long tail of the catalog.
    fig, ax = plt.subplots(figsize=(5, 3.2))
    counts = df["item_id"].value_counts().to_numpy()
    ax.loglog(range(1, len(counts) + 1), counts)
    ax.set(xlabel="Track rank by popularity", ylabel="Plays")
    fig.tight_layout()
    fig.savefig(out / "track_popularity.png", dpi=150)
    plt.close(fig)

    # Listening by hour of day (UTC): the daily rhythm that motivates time-aware attention.
    fig, ax = plt.subplots(figsize=(5, 3.2))
    df["datetime"].dt.hour.value_counts().sort_index().plot.bar(ax=ax)
    ax.set(xlabel="Hour of day (UTC)", ylabel="Listening events")
    fig.tight_layout()
    fig.savefig(out / "plays_by_hour.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote figures to {out}")


if __name__ == "__main__":
    main()
