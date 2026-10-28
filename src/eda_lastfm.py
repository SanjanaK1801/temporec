import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display on the cluster
import matplotlib.pyplot as plt

PROC = "/scratch/zt1/project/msml612/shared/next_song_prediction/data/processed"
FIG = f"{PROC}/figures"
os.makedirs(FIG, exist_ok=True)
lines = []

def log(msg):
    print(msg, flush=True)
    lines.append(msg)

df = pd.read_parquet(f"{PROC}/interactions_full.parquet")
log(f"Loaded {len(df):,} events, {df.user_id.nunique():,} users, "
    f"{df.item_id.nunique():,} tracks")

# ---------- 1. Timestamp check ----------
dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)
log("\n== Events per year ==")
log(dt.dt.year.value_counts().sort_index().to_string())
cutoff = pd.Timestamp("2009-06-01", tz="UTC")
log(f"\nEvents after 2009-06-01: {(dt > cutoff).sum():,}")
log(f"Users with any event after 2009-06-01: {df.user_id[dt > cutoff].nunique():,}")

month = dt.dt.to_period("M").value_counts().sort_index()
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.bar(month.index.to_timestamp(), month.values, width=25)
ax.set_xlabel("Month"); ax.set_ylabel("Number of events")
ax.set_title("Listening events per month")
fig.tight_layout(); fig.savefig(f"{FIG}/events_per_month.png", dpi=150); plt.close(fig)

# ---------- 2. Sequence length per user ----------
lens = df.groupby("user_id").size()
log("\n== Events per user ==")
log(lens.describe().to_string())
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(np.log10(lens.values), bins=30)
ax.set_xlabel("log10(events per user)"); ax.set_ylabel("Number of users")
ax.set_title("Listening history length per user")
fig.tight_layout(); fig.savefig(f"{FIG}/sequence_lengths.png", dpi=150); plt.close(fig)

# ---------- 3. Track popularity ----------
pop = df["item_id"].value_counts()
total = pop.sum()
log("\n== Popularity ==")
log(f"Most played track: {pop.iloc[0]:,} plays")
for pct in [0.001, 0.01, 0.1]:
    k = max(1, int(len(pop) * pct))
    log(f"Top {pct*100:g}% of tracks ({k:,}) cover {pop.iloc[:k].sum()/total:.1%} of events")
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(np.arange(1, len(pop) + 1), pop.values)
ax.set_xlabel("Track rank"); ax.set_ylabel("Number of plays")
ax.set_title("Track popularity (log-log)")
fig.tight_layout(); fig.savefig(f"{FIG}/popularity_curve.png", dpi=150); plt.close(fig)

# ---------- 4. Replay rate ----------
# data is sorted by (user_id, timestamp), so cumcount > 0 means
# "this user already played this track earlier"
is_replay = df.groupby(["user_id", "item_id"]).cumcount() > 0
log("\n== Replay rate ==")
log(f"Overall: {is_replay.mean():.1%} of events replay an earlier track")
per_user = is_replay.groupby(df["user_id"]).mean()
log(f"Per user: median {per_user.median():.1%}, "
    f"min {per_user.min():.1%}, max {per_user.max():.1%}")
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(per_user.values, bins=30)
ax.set_xlabel("Replay rate"); ax.set_ylabel("Number of users")
ax.set_title("Share of events that are replays, per user")
fig.tight_layout(); fig.savefig(f"{FIG}/replay_rate.png", dpi=150); plt.close(fig)

# ---------- 5. Hour of day (UTC) ----------
hours = dt.dt.hour.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(hours.index, hours.values)
ax.set_xlabel("Hour of day (UTC)"); ax.set_ylabel("Number of events")
ax.set_title("Listening by hour of day")
fig.tight_layout(); fig.savefig(f"{FIG}/hour_of_day.png", dpi=150); plt.close(fig)

with open(f"{PROC}/eda_stats_full.txt", "w") as f:
    f.write("\n".join(lines))
log("\nDone. Figures saved to " + FIG)