import pandas as pd

RAW = "/scratch/zt1/project/msml612/shared/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"

cols = ["user_id", "timestamp", "artist_mbid", "artist_name", "track_mbid", "track_name"]


df = pd.read_csv(
    RAW,
    sep="\t",
    header=None,
    names=cols,
    nrows=100_000,
    quoting=3,
    on_bad_lines="skip",
    dtype=str,
)

print("Shape:", df.shape)
print(df.head())
print("\nMissing values per column:")
print(df.isna().sum())
print("\nUnique values per column:")
print(df.nunique())

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
print("\nTime range:", df["timestamp"].min(), "to", df["timestamp"].max())