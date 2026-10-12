"""MovementAnalyzer: distance, speed, acceleration, sprints, direction changes.

Problem solved : raw foot positions jitter by 0.3-1 m from frame to frame.
                 Differentiating them directly yields absurd speeds
                 (1 m jitter at 25 FPS = 90 km/h).
Receives       : PlayerTrack with raw field positions (metres) + timestamps.
Produces       : smoothed positions, speed (km/h), acceleration (m/s^2),
                 total distance, sprints, acceleration/deceleration/stop/turn events.
Method         : 1. split the track at gaps > max_gap_s (never infer motion across them)
                 2. reject teleporting points (implied speed > plausible maximum)
                 3. resample to a uniform grid (interpolating short gaps)
                 4. smooth (Savitzky-Golay / moving average), THEN differentiate
                 5. discard physically impossible speeds/accelerations (NaN, not clipped)
                 6. max speed = max of a median-filtered speed series
All calculations use FIELD coordinates (metres), never pixels.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter

from ..config import MovementConfig
from ..models import MovementEvent, PlayerTrack, SprintRecord
from ..utils import MS_TO_KMH, find_runs, kmh_to_ms, merge_runs, wrap_angle_deg

logger = logging.getLogger(__name__)
_MIN_SEGMENT_SECONDS = 0.5
_MIN_SEGMENT_SAMPLES = 5
_MAX_RESYNC_REJECTIONS = 5
_OUTLIER_SPEED_TOLERANCE = 1.25


# ----------------------------------------------------------------------------
# Pure helper functions (unit-tested directly)
# ----------------------------------------------------------------------------
def reject_position_outliers(t: np.ndarray, xy: np.ndarray, max_speed_ms: float) -> np.ndarray:
    """Boolean keep-mask: drop points whose implied speed from the last accepted
    point exceeds the human limit. After ``_MAX_RESYNC_REJECTIONS`` consecutive
    rejections the tracker is assumed to have genuinely moved (e.g. ID switch)
    and the point is accepted to re-synchronise."""
    keep = np.ones(len(t), dtype=bool)
    last, rejected = 0, 0
    for i in range(1, len(t)):
        dt = max(t[i] - t[last], 1e-6)
        speed = np.linalg.norm(xy[i] - xy[last]) / dt
        if speed > max_speed_ms * _OUTLIER_SPEED_TOLERANCE and rejected < _MAX_RESYNC_REJECTIONS:
            keep[i] = False
            rejected += 1
        else:
            last, rejected = i, 0
    return keep


def smooth_series(values: np.ndarray, method: str, window: int, polyorder: int) -> np.ndarray:
    """Smooth along axis 0. ``window`` is forced odd and clipped to the series length."""
    if method == "none" or len(values) < 3:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    window = max(window | 1, 3)
    if method == "savgol":
        order = min(polyorder, window - 1)
        return savgol_filter(values, window, order, axis=0, mode="interp")
    if method == "moving_average":
        return uniform_filter1d(values, size=window, axis=0, mode="nearest")
    raise ValueError(f"Unknown smoothing method: {method}")


def moving_average_nan(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average that ignores NaNs."""
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    num = uniform_filter1d(filled, size=window, mode="nearest")
    den = uniform_filter1d(valid.astype(float), size=window, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 0.5] = np.nan
    return out


@dataclass
class _SegmentResult:
    grid_t: np.ndarray
    smoothed_xy: np.ndarray
    speed_ms: np.ndarray
    accel: np.ndarray
    heading_deg: np.ndarray
    step_distance: np.ndarray   # distance from previous grid sample


class MovementAnalyzer:
    def __init__(self, cfg: MovementConfig, effective_fps: float, video_fps: float) -> None:
        self.cfg = cfg
        self.dt = 1.0 / effective_fps
        self.video_fps = video_fps
        self.max_speed_ms = kmh_to_ms(cfg.max_plausible_speed_kmh)

    def _samples(self, seconds: float) -> int:
        return max(1, int(round(seconds / self.dt)))

    # --------------------------------------------------------------- public
    def analyze_all(self, tracks: dict[int, PlayerTrack]) -> None:
        for track in tracks.values():
            self.analyze_track(track)

    def analyze_track(self, track: PlayerTrack) -> None:
        obs = [o for o in track.observations if o.field_xy is not None and o.on_pitch]
        track.speeds = np.full(len(track.observations), np.nan)
        track.accelerations = np.full(len(track.observations), np.nan)
        track.distances = np.zeros(len(track.observations))
        if len(obs) < _MIN_SEGMENT_SAMPLES:
            return
        index_of = {id(o): i for i, o in enumerate(track.observations)}
        t = np.array([o.timestamp for o in obs])
        xy = np.array([o.field_xy for o in obs], dtype=float)

        gap_limit = max(self.cfg.max_gap_s, 3.0 * self.dt)
        cut_points = np.flatnonzero(np.diff(t) > gap_limit) + 1
        bounds = np.concatenate(([0], cut_points, [len(t)]))

        total_distance = total_time = 0.0
        speed_chunks: list[np.ndarray] = []
        sprint_candidates: list[SprintRecord] = []
        zone_time: dict[str, float] = {"walking": 0, "jogging": 0, "running": 0, "high_speed": 0, "sprint": 0}
        zone_dist = dict.fromkeys(zone_time, 0.0)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            seg_obs, seg_t, seg_xy = obs[lo:hi], t[lo:hi], xy[lo:hi]
            if hi - lo < _MIN_SEGMENT_SAMPLES or seg_t[-1] - seg_t[0] < _MIN_SEGMENT_SECONDS:
                continue
            keep = reject_position_outliers(seg_t, seg_xy, self.max_speed_ms)
            for o, k in zip(seg_obs, keep):
                o.is_outlier = not bool(k)
            if keep.sum() < _MIN_SEGMENT_SAMPLES:
                continue
            res = self._process_segment(seg_t[keep], seg_xy[keep])
            self._write_back(track, index_of, seg_obs, seg_t, res)
            valid_step = np.isfinite(res.speed_ms)
            total_distance += float(np.sum(res.step_distance[valid_step]))
            total_time += float(valid_step.sum() * self.dt)
            speed_chunks.append(res.speed_ms)
            self._zones(res, zone_time, zone_dist)
            sprint_candidates += self._detect_sprints(res, seg_t[keep])
            track.movement_events += self._movement_events(res)

        if not speed_chunks:
            return
        all_speed = np.concatenate(speed_chunks)
        track.movement_available = True
        track.total_distance_m = total_distance
        track.sprints = sprint_candidates
        track.sprint_count = len(sprint_candidates)
        track.average_speed = (total_distance / total_time * MS_TO_KMH) if total_time > 0 else None
        track.maximum_speed = self._robust_max_kmh(speed_chunks)
        track.extra_metrics.update({
            "time_in_zone_s": {k: round(v, 2) for k, v in zone_time.items()},
            "distance_in_zone_m": {k: round(v, 2) for k, v in zone_dist.items()},
            "high_intensity_distance_m": round(zone_dist["high_speed"] + zone_dist["sprint"], 2),
            "analysed_time_s": round(total_time, 2),
            "outlier_points_rejected": int(sum(o.is_outlier for o in obs)),
            "rejected_speed_samples": int(np.sum(~np.isfinite(all_speed))),
        })

    # ------------------------------------------------------------- internals
    def _process_segment(self, t: np.ndarray, xy: np.ndarray) -> _SegmentResult:
        grid_t = np.arange(t[0], t[-1] + 1e-9, self.dt)
        gx = np.interp(grid_t, t, xy[:, 0])
        gy = np.interp(grid_t, t, xy[:, 1])
        grid = np.stack([gx, gy], axis=1)
        smoothed = smooth_series(grid, self.cfg.smoothing_method,
                                 self._samples(self.cfg.smoothing_window_s), self.cfg.savgol_polyorder)
        vel = np.gradient(smoothed, self.dt, axis=0)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        speed[speed > self.max_speed_ms] = np.nan  # impossible: reject, never clip
        heading = np.degrees(np.arctan2(vel[:, 1], vel[:, 0]))
        speed_smooth = moving_average_nan(speed, self._samples(self.cfg.accel_smoothing_window_s) | 1)
        accel = np.gradient(speed_smooth, self.dt)
        accel[np.abs(accel) > self.cfg.max_plausible_accel_ms2] = np.nan
        step = np.concatenate(([0.0], np.hypot(*np.diff(smoothed, axis=0).T)))
        return _SegmentResult(grid_t, smoothed, speed, accel, heading, step)

    @staticmethod
    def _write_back(track, index_of, seg_obs, seg_t, res: _SegmentResult) -> None:
        sx = np.interp(seg_t, res.grid_t, res.smoothed_xy[:, 0])
        sy = np.interp(seg_t, res.grid_t, res.smoothed_xy[:, 1])
        sp = np.interp(seg_t, res.grid_t, res.speed_ms)
        ac = np.interp(seg_t, res.grid_t, res.accel)
        step = np.interp(seg_t, res.grid_t, np.cumsum(res.step_distance))
        previous_cum = 0.0
        for o, x, y, s, a, cum in zip(seg_obs, sx, sy, sp, ac, step):
            i = index_of[id(o)]
            o.smoothed_xy = (float(x), float(y))
            o.speed_kmh = None if not np.isfinite(s) else float(s * MS_TO_KMH)
            o.accel_ms2 = None if not np.isfinite(a) else float(a)
            track.speeds[i] = np.nan if o.speed_kmh is None else o.speed_kmh
            track.accelerations[i] = np.nan if o.accel_ms2 is None else o.accel_ms2
            track.distances[i] = cum - previous_cum
            previous_cum = cum

    def _robust_max_kmh(self, speed_chunks: list[np.ndarray]) -> float | None:
        window = self._samples(self.cfg.max_speed_median_window_s) | 1
        best = np.nan
        for chunk in speed_chunks:
            filled = np.where(np.isfinite(chunk), chunk, 0.0)  # NaN (rejected) cannot raise the max
            med = median_filter(filled, size=window, mode="nearest")
            if med.size:
                best = np.nanmax([best, med.max()])
        return None if not np.isfinite(best) or best <= 0 else float(best * MS_TO_KMH)

    def _zones(self, res: _SegmentResult, zone_time: dict, zone_dist: dict) -> None:
        c = self.cfg
        kmh = res.speed_ms * MS_TO_KMH
        edges = [("walking", c.walking_max_speed), ("jogging", c.jogging_max_speed),
                 ("running", c.running_max_speed), ("high_speed", c.sprint_threshold), ("sprint", math.inf)]
        lower = 0.0
        for name, upper in edges:
            mask = np.isfinite(kmh) & (kmh >= lower) & (kmh < upper)
            zone_time[name] += float(mask.sum() * self.dt)
            zone_dist[name] += float(res.step_distance[mask].sum())
            lower = upper

    def _detect_sprints(self, res: _SegmentResult, obs_times: np.ndarray) -> list[SprintRecord]:
        """Sprint = speed >= threshold for >= min duration (single frames never count)."""
        c = self.cfg
        kmh = res.speed_ms * MS_TO_KMH
        mask = np.isfinite(kmh) & (kmh >= c.sprint_threshold)
        runs = merge_runs(find_runs(mask), self._samples(c.sprint_merge_gap_s) - 1)
        min_len = self._samples(c.min_sprint_duration)
        sprints: list[SprintRecord] = []
        for start, end in runs:
            if end - start + 1 < min_len:
                continue
            duration = (end - start + 1) * self.dt
            in_run = (obs_times >= res.grid_t[start]) & (obs_times <= res.grid_t[end])
            expected = max(1.0, duration * self._expected_obs_rate())
            sprints.append(SprintRecord(
                start_time=float(res.grid_t[start]), end_time=float(res.grid_t[end] + self.dt),
                start_frame=int(round(res.grid_t[start] * self.video_fps)),
                end_frame=int(round(res.grid_t[end] * self.video_fps)),
                duration_s=float(duration), max_speed_kmh=float(np.nanmax(kmh[start:end + 1])),
                distance_m=float(res.step_distance[start:end + 1].sum()),
                observed_fraction=float(min(1.0, in_run.sum() / expected)),
            ))
        return sprints

    def _expected_obs_rate(self) -> float:
        return 1.0 / self.dt

    def _movement_events(self, res: _SegmentResult) -> list[MovementEvent]:
        c = self.cfg
        events: list[MovementEvent] = []
        min_len = self._samples(c.min_accel_duration_s)
        acc = np.where(np.isfinite(res.accel), res.accel, 0.0)
        for kind, mask in (("explosive_acceleration", acc >= c.explosive_accel_ms2),
                           ("rapid_deceleration", acc <= c.rapid_decel_ms2)):
            for s, e in find_runs(mask, min_len):
                value = float(np.max(np.abs(acc[s:e + 1])))
                events.append(MovementEvent(kind, float(res.grid_t[s]), float(res.grid_t[e]), value))
        kmh = res.speed_ms * MS_TO_KMH
        slow = np.isfinite(kmh) & (kmh <= c.stop_speed_kmh)
        lookback = self._samples(1.5)
        for s, e in find_runs(slow, min_len):
            prior = kmh[max(0, s - lookback):s]
            if prior.size and np.nanmax(prior) >= c.stop_prior_speed_kmh:
                events.append(MovementEvent("stop", float(res.grid_t[s]), float(res.grid_t[e]), float(np.nanmax(prior))))
        lag = self._samples(c.direction_change_window_s)
        if len(kmh) > lag:
            delta = np.abs(wrap_angle_deg(res.heading_deg[lag:] - res.heading_deg[:-lag]))
            fast = (np.nan_to_num(kmh[lag:]) >= c.direction_change_min_speed_kmh) & (np.nan_to_num(kmh[:-lag]) >= c.direction_change_min_speed_kmh)
            for s, e in find_runs((delta >= c.direction_change_deg) & fast):
                events.append(MovementEvent("direction_change", float(res.grid_t[s]), float(res.grid_t[e + lag]), float(delta[s:e + 1].max())))
        return events
