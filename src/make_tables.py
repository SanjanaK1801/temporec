"""Average results over training seeds and print Markdown tables for the report (TR-09).

Usage:
    python src/make_tables.py > results/tables.md
"""
from pathlib import Path

import pandas as pd

PUBLISHED_ML1M = pd.DataFrame([
    {"model": "HSTU (paper)", "hr@10": 0.3097, "ndcg@10": 0.1720},
    {"model": "HSTU-large (paper)", "hr@10": 0.3294, "ndcg@10": 0.1893},
    {"model": "HSTU (RecTools reproduction)", "hr@10": 0.3226, "ndcg@10": 0.1880},
    {"model": "HSTU-large (RecTools reproduction)", "hr@10": 0.3642, "ndcg@10": 0.2164},
    {"model": "SASRec (RecTools)", "hr@10": 0.2877, "ndcg@10": 0.1622},
])
METRICS = ["hr@10", "recall@10", "ndcg@10", "coverage@10"]
CONFIG_KEYS = ["max_len", "blocks", "artist", "time_attention"]


def summarize(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Mean ± standard deviation over seeds for each configuration."""
    cols = [m for m in METRICS if m in df]
    g = df.groupby(keys, dropna=False)[cols]
    table = g.mean().round(4).astype(str) + " ± " + g.std().fillna(0).round(4).astype(str)
    table["seeds"] = g.size()
    return table.reset_index()


def main() -> None:
    if Path("results/movielens.csv").exists():
        ml = pd.read_csv("results/movielens.csv")
        print("## MovieLens-1M: published results\n")
        print(PUBLISHED_ML1M.to_markdown(index=False))
        print("\n## MovieLens-1M: our runs\n")
        print(summarize(ml, ["model"]).to_markdown(index=False))
        print()
    if Path("results/lastfm.csv").exists():
        lf = pd.read_csv("results/lastfm.csv")
        lf = lf[lf["sample_frac"] == 1.0]
        keys = ["eval", "model"] + [k for k in CONFIG_KEYS if k in lf]
        lf[keys] = lf[keys].astype(object).where(lf[keys].notna(), "default")
        for ev in ("all", "discovery"):
            print(f"## Last.fm 1K: {ev}\n")
            print(summarize(lf[lf["eval"] == ev], keys).drop(columns="eval").to_markdown(index=False))
            print()


if __name__ == "__main__":
    main()
