# Zaratan runs (Fatima)

| Job | ID | Commit | Result |
|---|---|---|---|
| prepare_v2 (cleaning) | 23655700 | 7f8e687 | 17,406,337 events, 983 users, 528,287 tracks |
| split_v2 | (job ID) | 98dffac | train 13,925,068 / val 1,513,590 / test 1,269,816 |
| check_gpu (TR-02) | 23650349 | 2b4537e | queued, estimated start 2026-11-03 |

Output folder: /scratch/zt1/project/msml612/shared/next_song_prediction/data/processed_v2/lastfm

Environment: miniforge, conda env `temporec` (Python 3.11), `pip install -r requirements.txt "pyarrow<26"`. Tests: 12 passing on 98dffac.

## Data Split Details

### split.json
```json
{
  "val_cutoff": "2008-10-10 17:56:30",
  "test_cutoff": "2009-01-29 22:44:39",
  "val_frac": 0.1,
  "test_frac": 0.1,
  "train_events": 13925068,
  "val_events": 1513590,
  "test_events": 1269816,
  "train_users": 900,
  "val_users": 698,
  "test_users": 784,
  "val_first_time_share": 0.3436260810391189,
  "test_first_time_share": 0.3014413111820925
}
```

### stats.json
```json
{
  "steps": [
    {"step": "raw", "events": 19150868, "users": 992, "items": null},
    {"step": "cleaned", "events": 19150659, "users": 992, "items": 1500621},
    {"step": "up to 2009-06-30", "events": 19150657, "users": 992, "items": 1500621},
    {"step": "deduplicated", "events": 19137103, "users": 992, "items": 1500621},
    {"step": "filtered", "events": 17406337, "users": 983, "items": 528287}
  ],
  "first_event": "2005-02-14 00:00:07",
  "last_event": "2009-06-19 21:31:16",
  "artists": 61003,
  "events_per_user_median": 10183.0,
  "repeat_share": 0.800601355701662
}
```