"""BallTracker: temporally consistent ball trajectory.

Problem solved : raw ball detections flicker (misses, false positives such as
                 white boots or line marks). Players tolerate that; a ball
                 trajectory cannot.
Receives       : candidate ``BallDetection`` list per frame (may be empty).
Produces       : one ``BallState`` per frame (or None while no ball is known)
                 with position, velocity, direction, confidence and a
                 ``predicted`` flag.
Method         : constant-velocity Kalman filter + distance gating.
                 * a candidate is accepted only if it lies inside a gate around
                   the predicted position (rejects impossible jumps);
                 * a new track must be confirmed by N consistent detections;
                 * on a miss the filter coasts for ``max_coast_s`` with decaying
                   confidence, and the trajectory is NOT reset.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ..config import BallConfig
from ..models import BallDetection, BallState

logger = logging.getLogger(__name__)
_MIN_DT = 1e-3
_OUT_OF_FRAME_MARGIN = 0.05


class _ConstantVelocityKalman:
    """State [x, y, vx, vy]; position measurements only."""

    def __init__(self, xy, velocity, accel_noise: float, meas_noise: float) -> None:
        self.x = np.array([xy[0], xy[1], velocity[0], velocity[1]], dtype=float)
        self.P = np.diag([meas_noise ** 2] * 2 + [(accel_noise * 0.1) ** 2] * 2)
        self.q = accel_noise
        self.R = np.eye(2) * meas_noise ** 2
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    def predict(self, dt: float) -> None:
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        q2 = self.q ** 2
        block = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * q2
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = block
        Q[np.ix_([1, 3], [1, 3])] = block
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z) -> None:
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


class BallTracker:
    NONE, TENTATIVE, TRACKING = "none", "tentative", "tracking"

    def __init__(self, cfg: BallConfig, frame_size: tuple[int, int]) -> None:
        self.cfg = cfg
        self.width, self.height = frame_size
        self._gate_min = cfg.gate_min_frac_width * self.width
        self._max_speed_px = cfg.max_speed_width_per_s * self.width
        self._accel_noise = cfg.process_accel_width_per_s2 * self.width
        self.state = self.NONE
        self._kf: Optional[_ConstantVelocityKalman] = None
        self._last_time = 0.0
        self._last_obs_time = 0.0
        self._last_conf = 0.0
        self._last_bbox_size = (8.0, 8.0)
        self._tentative: Optional[tuple[BallDetection, int]] = None
        self.lost_events = 0

    # ------------------------------------------------------------------
    def update(self, candidates: list[BallDetection], frame: int, timestamp: float) -> Optional[BallState]:
        if self.state == self.NONE:
            return self._start_tentative(candidates, timestamp)
        if self.state == self.TENTATIVE:
            return self._confirm_tentative(candidates, timestamp)
        return self._track(candidates, frame, timestamp)

    # ---------------------------------------------------------- acquisition
    def _start_tentative(self, candidates, timestamp) -> None:
        good = [c for c in candidates if c.confidence >= self.cfg.init_min_confidence]
        if good:
            self._tentative = (max(good, key=lambda c: c.confidence), 1)
            self.state = self.TENTATIVE
        return None

    def _confirm_tentative(self, candidates, timestamp) -> Optional[BallState]:
        prev, count = self._tentative
        dt = max(timestamp - prev.timestamp, _MIN_DT)
        reach = self._gate_min + self._max_speed_px * dt
        near = [c for c in candidates if math.dist(c.center, prev.center) <= reach]
        if not near:
            self.state, self._tentative = self.NONE, None
            return self._start_tentative(candidates, timestamp)
        best = max(near, key=lambda c: c.confidence)
        count += 1
        if count < self.cfg.init_confirm_detections:
            self._tentative = (best, count)
            return None
        velocity = ((best.center[0] - prev.center[0]) / dt, (best.center[1] - prev.center[1]) / dt)
        self._kf = _ConstantVelocityKalman(best.center, velocity, self._accel_noise, self.cfg.measurement_noise_px)
        self.state, self._tentative = self.TRACKING, None
        self._last_time = self._last_obs_time = timestamp
        return self._observed_state(best, best.frame, timestamp)

    # ------------------------------------------------------------- tracking
    def _track(self, candidates, frame, timestamp) -> Optional[BallState]:
        dt = max(timestamp - self._last_time, _MIN_DT)
        self._kf.predict(dt)
        self._last_time = timestamp
        predicted = self._kf.position
        gate = self._gate_min + self._max_speed_px * max(timestamp - self._last_obs_time, _MIN_DT)

        best, best_score = None, 0.0
        for cand in candidates:
            dist = math.dist(cand.center, predicted)
            if dist > gate:
                continue  # impossible jump -> false detection
            score = cand.confidence * (1.0 - 0.7 * dist / gate)
            if score > best_score:
                best, best_score = cand, score
        if best is not None:
            self._kf.update(best.center)
            self._last_obs_time = timestamp
            return self._observed_state(best, frame, timestamp)

        # Miss: coast on the motion model, do not reset the trajectory.
        missed = timestamp - self._last_obs_time
        x, y = predicted
        out_of_frame = (x < -_OUT_OF_FRAME_MARGIN * self.width or x > (1 + _OUT_OF_FRAME_MARGIN) * self.width
                        or y < -_OUT_OF_FRAME_MARGIN * self.height or y > (1 + _OUT_OF_FRAME_MARGIN) * self.height)
        if missed > self.cfg.max_coast_s or out_of_frame:
            logger.warning("Ball lost at frame %d (t=%.2fs): no consistent detection for %.2fs.", frame, timestamp, missed)
            self.state, self._kf = self.NONE, None
            self.lost_events += 1
            return None
        frames_missed = max(1, int(round(missed / dt)))
        conf = self._last_conf * (self.cfg.coast_confidence_decay ** frames_missed)
        return self._build_state(frame, timestamp, conf, True, None)

    # --------------------------------------------------------------- helpers
    def _observed_state(self, det: BallDetection, frame: int, timestamp: float) -> BallState:
        self._last_conf = det.confidence
        self._last_bbox_size = (det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1])
        return self._build_state(frame, timestamp, det.confidence, False, det)

    def _build_state(self, frame, timestamp, conf, predicted, raw: Optional[BallDetection]) -> BallState:
        x, y = self._kf.position
        vx, vy = self._kf.velocity
        w, h = self._last_bbox_size
        bbox = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        if raw is not None:  # keep the true box of a real detection
            bbox = raw.bbox
        return BallState(
            frame=frame, timestamp=timestamp, center=(x, y), bbox=bbox, confidence=float(conf),
            predicted=predicted, velocity_px=(vx, vy),
            direction_deg=math.degrees(math.atan2(vy, vx)) if (vx or vy) else 0.0, raw_detection=raw,
        )
