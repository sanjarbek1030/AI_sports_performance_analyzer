"""Ball trajectory in field coordinates, smoothed, with velocity.

Problem solved : events (pass / shot / goal) need ball speed and direction in
                 metres, not pixels.
Receives       : per-frame ``BallState`` (with ``field_xy``) from the analysis pass.
Produces       : ``BallKinematics`` arrays aligned with the analysed frames:
                 position, velocity, speed, and an ``observed`` mask separating
                 real detections from motion-predicted positions.
Caveat         : ground-plane homography is wrong for an airborne ball, so
                 speeds of lofted balls are over/under-estimated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BallConfig
from ..models import FrameData
from ..utils import find_runs
from .movement import smooth_series

_MIN_SEGMENT = 5
_MAX_GAP_SAMPLES = 2


@dataclass
class BallKinematics:
    frames: np.ndarray       # frame index per sample
    times: np.ndarray        # seconds
    xy: np.ndarray           # (N,2) smoothed field metres, NaN = unknown
    raw_xy: np.ndarray       # (N,2) unsmoothed
    velocity: np.ndarray     # (N,2) m/s
    speed: np.ndarray        # (N,) m/s, NaN = unknown/implausible
    observed: np.ndarray     # bool: a real detection backs this sample
    confidence: np.ndarray   # detector/tracker confidence

    def __len__(self) -> int:
        return len(self.times)

    @property
    def coverage(self) -> float:
        """Fraction of analysed frames with a genuinely detected ball."""
        return float(self.observed.mean()) if len(self) else 0.0

    def window(self, t_start: float, t_end: float) -> np.ndarray:
        """Indices with t_start <= t <= t_end."""
        lo = np.searchsorted(self.times, t_start, side="left")
        hi = np.searchsorted(self.times, t_end, side="right")
        return np.arange(lo, hi)

    def index_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.times, t), 0, max(len(self) - 1, 0)))


def build_ball_kinematics(frames: list[FrameData], cfg: BallConfig, effective_fps: float) -> BallKinematics:
    n = len(frames)
    dt = 1.0 / effective_fps
    times = np.array([f.timestamp for f in frames], dtype=float)
    raw = np.full((n, 2), np.nan)
    observed = np.zeros(n, dtype=bool)
    conf = np.zeros(n)
    for i, f in enumerate(frames):
        if f.ball is not None and f.ball.field_xy is not None:
            raw[i] = f.ball.field_xy
            observed[i] = not f.ball.predicted
            conf[i] = f.ball.confidence

    smooth = raw.copy()
    velocity = np.full((n, 2), np.nan)
    window = max(int(round(cfg.smoothing_window_s / dt)) | 1, 3)
    valid = np.isfinite(raw).all(axis=1)
    # segments of consecutive valid samples (small holes are tolerated by the runs finder)
    for s, e in find_runs(valid, _MIN_SEGMENT):
        smooth[s:e + 1] = smooth_series(raw[s:e + 1], "savgol", window, 2)
        velocity[s:e + 1] = np.gradient(smooth[s:e + 1], dt, axis=0)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    bad = speed > cfg.max_plausible_speed_ms
    speed[bad] = np.nan
    velocity[bad] = np.nan
    return BallKinematics(
        frames=np.array([f.frame for f in frames]), times=times, xy=smooth, raw_xy=raw,
        velocity=velocity, speed=speed, observed=observed, confidence=conf,
    )
