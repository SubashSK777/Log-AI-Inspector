"""
utils.py
--------
Log preprocessing utilities.

Pipeline:
  raw lines  →  cleaned tokens  →  integer sequences (window-based)

Token encoding uses a deterministic hash to keep things self-contained
(no training / vocabulary file needed for the MVP).
"""

import re
from typing import List, Tuple


# ── constants ──────────────────────────────────────────────────────────────
VOCAB_SIZE   = 500    # matches LogModel(vocab_size=500)
WINDOW_SIZE  = 10     # how many log lines per sequence


# ── helpers ────────────────────────────────────────────────────────────────

def _clean_line(line: str) -> str:
    """Strip timestamps, IPs and numeric IDs so tokens are more meaningful."""
    line = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.,\d]*Z?', '', line)  # timestamps
    line = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', line)        # IPv4
    line = re.sub(r'blk_[\w-]+',  '<BLK>',  line)                                 # block IDs
    line = re.sub(r'\b\d+\b',     '<NUM>',  line)                                 # plain numbers
    return line.strip()


def _encode_line(line: str) -> int:
    """Map a single log line to an integer token in [1, VOCAB_SIZE-1]."""
    cleaned = _clean_line(line)
    # deterministic, stable hash (avoids Python hash randomisation)
    h = 0
    for ch in cleaned:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (h % (VOCAB_SIZE - 1)) + 1   # keep 0 for padding


# ── public API ─────────────────────────────────────────────────────────────

def preprocess_logs(
    logs: List[str],
    window: int = WINDOW_SIZE
) -> Tuple[List[List[int]], List[List[str]]]:
    """
    Convert a list of raw log strings into fixed-length integer sequences.

    Returns
    -------
    sequences : list of list[int]   — encoded token IDs per window
    raw_windows: list of list[str]  — original lines per window
    """
    # Drop blank lines
    non_empty = [l for l in logs if l.strip()]

    sequences   = []
    raw_windows = []

    for i in range(0, len(non_empty), window):
        chunk = non_empty[i : i + window]
        if not chunk:
            continue

        # Pad short final window so every sequence is the same length
        while len(chunk) < window:
            chunk.append("")

        encoded     = [_encode_line(line) for line in chunk]
        sequences.append(encoded)
        raw_windows.append(chunk)

    return sequences, raw_windows
