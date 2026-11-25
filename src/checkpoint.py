"""Save work as we go, so a crash near the end doesn't cost us the whole run (TR-01, shared).

Two small tools:

- write_parquet / write_json / atomic_write: write to a temporary file next to the target and
  then rename it. The rename happens in one step, so if the process dies halfway through we are
  left with the old file (or no file), never a truncated one that only breaks when someone
  tries to read it.

- StageCheckpoints: a script with several slow steps saves its table after each one. The next
  run starts from the last step that finished, as long as the settings that produced it are the
  same. Change a setting and the affected steps are simply recomputed.
"""
import json
import os
from pathlib import Path

import pandas as pd


def atomic_write(path, write) -> None:
    """Call write(tmp_path), then move the result onto `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # only still there if write() failed


def write_parquet(df: pd.DataFrame, path) -> None:
    atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))


def write_json(obj, path) -> None:
    atomic_write(path, lambda tmp: Path(tmp).write_text(json.dumps(obj, indent=2) + "\n"))


class StageCheckpoints:
    """One parquet file per finished step, plus a small JSON with the settings that made it.

    `settings` should be plain JSON values (str, int, float, bool, lists, dicts), so they compare
    equal after a round trip through the JSON file. Pass resume=False to ignore anything saved
    earlier (new checkpoints are still written).
    """

    def __init__(self, folder, resume: bool = True):
        self.folder = Path(folder)
        self.resume = resume

    def _files(self, step: str):
        return self.folder / f"{step}.parquet", self.folder / f"{step}.json"

    def load(self, step: str, settings: dict):
        """Return (table, info) saved for this step, or None if there is nothing usable."""
        if not self.resume:
            return None
        table_file, meta_file = self._files(step)
        if not (table_file.exists() and meta_file.exists()):
            return None
        meta = json.loads(meta_file.read_text())
        if meta.get("settings") != settings:
            return None
        return pd.read_parquet(table_file), meta.get("info", {})

    def save(self, step: str, table: pd.DataFrame, settings: dict, info: dict = None) -> None:
        table_file, meta_file = self._files(step)
        # The JSON goes last: a step only counts as saved once both files are there.
        write_parquet(table, table_file)
        write_json({"settings": settings, "info": info or {}}, meta_file)
