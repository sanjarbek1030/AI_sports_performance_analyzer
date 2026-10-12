"""Small, dependency-light helpers used across modules."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

MS_TO_KMH = 3.6


def ms_to_kmh(value):
    return value * MS_TO_KMH


def kmh_to_ms(value):
    return value / MS_TO_KMH


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``MM:SS.ff`` (minutes may exceed 59), e.g. ``00:13.42``."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"


def find_runs(mask: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index pairs of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_length]


def merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge runs separated by at most ``max_gap`` samples."""
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator == 0 else numerator / denominator


def clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def wrap_angle_deg(angle: np.ndarray) -> np.ndarray:
    """Wrap angle differences to [-180, 180)."""
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0
