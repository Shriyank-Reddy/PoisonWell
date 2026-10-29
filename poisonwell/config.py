"""Paths and settings in one place so beginners know where to tweak things."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# "tfidf" (default, offline, fast) or "minilm" (sentence-transformers all-MiniLM-L6-v2).
EMBEDDER = os.environ.get("POISONWELL_EMBEDDER", "tfidf")

TOP_K = 3                # how many chunks the retriever returns
BLOCK_THRESHOLD = 0.5    # DefenseLayer blocks a chunk when its risk score reaches this
STRONG_SIGNAL = 0.8      # ...or when any single detector is at least this sure
