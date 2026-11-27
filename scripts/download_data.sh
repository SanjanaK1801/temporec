#!/bin/bash
# Downloads Last.fm 1K and MovieLens-1M into data/raw and checks the Last.fm checksum.
# Last.fm 1K is licensed for non-commercial research (Last.fm must be referenced), so it is not in the repo.
# Needs wget, tar, unzip and md5sum (Linux; on Windows use WSL or Git Bash).
set -e
LASTFM_URL="${LASTFM_URL:-https://zenodo.org/records/6090214/files/lastfm-dataset-1K.tar.gz?download=1}"
LASTFM_MD5="a79a6808f54f73354789a9fb02cb1e41"

mkdir -p data/raw && cd data/raw

wget -O lastfm-dataset-1K.tar.gz "$LASTFM_URL"
echo "$LASTFM_MD5  lastfm-dataset-1K.tar.gz" | md5sum -c -
tar -xzf lastfm-dataset-1K.tar.gz

wget -O ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -o ml-1m.zip

echo "Done. Last.fm: data/raw/lastfm-dataset-1K, MovieLens: data/raw/ml-1m"