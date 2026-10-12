"""FieldMapper: image coordinates -> field coordinates (metres).

image coordinates -> perspective transformation (homography) -> field coordinates

Only ground-plane points are mapped correctly: use a player's feet
(bottom-centre of the box) and the ball's contact point. An airborne ball is
projected onto the pitch at the wrong place (documented limitation).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .calibration import CalibrationProvider
from .football_field import FieldModel


class FieldMapper:
    def __init__(self, field: FieldModel, provider: CalibrationProvider) -> None:
        self.field = field
        self.provider = provider
        self._cache: tuple[int, Optional[np.ndarray]] = (-1, None)

    @property
    def is_calibrated(self) -> bool:
        try:
            return self.provider.homography(0, None) is not None
        except NotImplementedError:
            return False

    def _matrix(self, frame_index: int, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if self._cache[0] != frame_index:
            self._cache = (frame_index, self.provider.homography(frame_index, frame))
        return self._cache[1]

    def to_field(self, points_xy, frame_index: int = 0, frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Map (N,2) image points -> (N,2) field metres; None if uncalibrated; NaN if non-finite."""
        matrix = self._matrix(frame_index, frame)
        if matrix is None:
            return None
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
        out[~np.isfinite(out).all(axis=1)] = np.nan
        return out

    def to_field_point(self, x: float, y: float, frame_index: int = 0) -> Optional[tuple[float, float]]:
        mapped = self.to_field([[x, y]], frame_index)
        if mapped is None or np.isnan(mapped).any():
            return None
        return float(mapped[0, 0]), float(mapped[0, 1])

    def to_image(self, field_points, frame_index: int = 0) -> Optional[np.ndarray]:
        """Inverse mapping (used to draw field geometry on the video in debug mode)."""
        matrix = self._matrix(frame_index, None)
        if matrix is None:
            return None
        pts = np.asarray(field_points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(matrix)).reshape(-1, 2)

    def on_pitch(self, xy: Optional[tuple[float, float]], margin: float) -> bool:
        return xy is not None and np.isfinite(xy).all() and self.field.contains(xy[0], xy[1], margin)
