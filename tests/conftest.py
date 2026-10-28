import sys
from pathlib import Path

# Scripts in src/ import each other as top-level modules (python src/train_lastfm.py), so tests do the same.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
