"""Dataset statistics table and figures for the report (TR-03, data and eval).

Usage:
    python src/dataset_figures.py --data data/processed/lastfm --out figures

Prints the statistics table and also saves it as stats_table.md next to stats.json. Each figure
is drawn on its own, so if one of them breaks the others are still saved; the script lists what
failed and exits with an error at the end.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from checkpoint import atomic_write


def stats_table(stats: dict) -> str:
    lines = [
        "## Data preparation steps",
        "",
        pd.DataFrame(stats["steps"]).to_markdown(index=False),
        "",
        f"Date range: {stats['first_event']} to {stats['last_event']}",
        f"Artists: {stats['artists']:,}; median events per user: {stats['events_per_user_median']:,.0f}; "
        f"repeat share: {stats['repeat_share']:.1%}",
    ]
    return "\n".join(lines) + "\n"


def events_per_user(df, ax):
    # Log scale on the y axis: a handful of very heavy listeners would flatten everything else.
    df.groupby("user_id").size().plot.hist(bins=50, log=True, ax=ax)
    ax.set(xlabel="Listening events per user", ylabel="Users (log scale)")


def track_popularity(df, ax):
    # Tracks sorted by play count: the long tail of the catalog.
    counts = df["item_id"].value_counts().to_numpy()
    ax.loglog(range(1, len(counts) + 1), counts)
    ax.set(xlabel="Track rank by popularity", ylabel="Plays")


def plays_by_hour(df, ax):
    # The daily rhythm of listening, which is part of why time-aware attention might help.
    df["datetime"].dt.hour.value_counts().sort_index().plot.bar(ax=ax)
    ax.set(xlabel="Hour of day (UTC)", ylabel="Listening events")


FIGURES = {
    "events_per_user.png": events_per_user,
    "track_popularity.png": track_popularity,
    "plays_by_hour.png": plays_by_hour,
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/lastfm")
    p.add_argument("--out", default="figures")
    args = p.parse_args(argv)
    data, out = Path(args.data), Path(args.out)

    table = stats_table(json.loads((data / "stats.json").read_text()))
    print(table)
    atomic_write(data / "stats_table.md", lambda tmp: Path(tmp).write_text(table))

    df = pd.read_parquet(data / "interactions.parquet")
    failed = []
    for name, draw in FIGURES.items():
        fig, ax = plt.subplots(figsize=(5, 3.2))
        try:
            draw(df, ax)
            fig.tight_layout()
            atomic_write(out / name, lambda tmp: fig.savefig(tmp, dpi=150, format="png"))
            print(f"saved {out / name}")
        except Exception:
            traceback.print_exc()
            failed.append(name)
        finally:
            plt.close(fig)

    if failed:
        sys.exit(f"could not draw: {', '.join(failed)}")


if __name__ == "__main__":
    main()
