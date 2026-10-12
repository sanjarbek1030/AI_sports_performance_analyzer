"""Camera/field calibration providers (isolated from the analytics).

Problem solved : image pixels are not metres. A homography maps the image
                 plane of the pitch to field coordinates.
Receives       : >= 4 image points and the matching field points (metres) -
                 either explicit coordinates or named pitch landmarks.
Produces       : a 3x3 homography per frame via ``CalibrationProvider.homography``.
Extensible     : ``ManualCalibration`` is a static camera. Replace it with an
                 automatic provider (field-line / keypoint detection, or
                 per-frame tracking for panning cameras) by subclassing
                 ``CalibrationProvider``; nothing else in the pipeline changes.
Validation     : point count, duplicates, collinearity, reprojection error and
                 matrix conditioning are all checked -> ``CalibrationError``.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ..exceptions import CalibrationError
from .football_field import FieldModel

logger = logging.getLogger(__name__)
_MIN_POINTS = 4
_MAX_CONDITION_NUMBER = 1e9
_MIN_SPREAD_RATIO = 1e-3


class CalibrationProvider(ABC):
    @abstractmethod
    def homography(self, frame_index: int, frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Image->field homography valid for this frame (None if unknown)."""

    @property
    def image_points(self) -> Optional[np.ndarray]:
        """Calibration points in image space (drawn in debug mode)."""
        return None


class NullCalibration(CalibrationProvider):
    """No calibration configured: field-based analytics are disabled."""

    def homography(self, frame_index, frame=None):
        return None


class AutomaticCalibration(CalibrationProvider):
    """Placeholder for automatic keypoint/field-line calibration (not implemented)."""

    def homography(self, frame_index, frame=None):
        raise NotImplementedError(
            "Automatic calibration is a planned extension: detect pitch keypoints in `frame` and call "
            "compute_homography(image_pts, field_pts, ...)."
        )


def compute_homography(
    image_points: Sequence[Sequence[float]], field_points: Sequence[Sequence[float]], max_error_m: float
) -> np.ndarray:
    """Validated homography image->field. Raises ``CalibrationError`` when unusable."""
    src = np.asarray(image_points, dtype=np.float64)
    dst = np.asarray(field_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape:
        raise CalibrationError("Image and field points must be equal-length lists of [x, y] pairs.")
    if len(src) < _MIN_POINTS:
        raise CalibrationError(f"At least {_MIN_POINTS} point pairs are required, got {len(src)}.")
    for name, pts in (("image", src), ("field", dst)):
        if len({tuple(np.round(p, 3)) for p in pts}) < len(pts):
            raise CalibrationError(f"Duplicate {name} calibration points.")
        centered = pts - pts.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        if singular[0] == 0 or singular[-1] / singular[0] < _MIN_SPREAD_RATIO:
            raise CalibrationError(f"The {name} calibration points are (nearly) collinear.")
    method = 0 if len(src) == _MIN_POINTS else cv2.RANSAC
    matrix, _ = cv2.findHomography(src, dst, method, ransacReprojThreshold=max_error_m)
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise CalibrationError("Homography estimation failed.")
    if np.linalg.cond(matrix) > _MAX_CONDITION_NUMBER:
        raise CalibrationError("Homography is degenerate (ill-conditioned).")
    mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(mapped - dst, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    if rmse > max_error_m:
        raise CalibrationError(
            f"Calibration reprojection RMSE {rmse:.2f} m exceeds {max_error_m:.2f} m "
            "(check that point pairs correspond; worst pair index "
            f"{int(errors.argmax())})."
        )
    logger.info("Calibration OK: %d points, reprojection RMSE %.3f m", len(src), rmse)
    return matrix


class ManualCalibration(CalibrationProvider):
    """Static homography from user-clicked points (fixed or tactical camera)."""

    def __init__(self, image_points, field_points, max_error_m: float) -> None:
        self._image_points = np.asarray(image_points, dtype=np.float64)
        self._matrix = compute_homography(image_points, field_points, max_error_m)

    @classmethod
    def from_json(cls, path: str | Path, field: FieldModel, video_size: tuple[int, int], max_error_m: float):
        """Load ``{"image_points": [...], "field_points"|"landmarks": [...], "frame_size": [w, h]}``."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CalibrationError(f"Calibration file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"Calibration file is not valid JSON: {exc}") from exc
        if "image_points" not in data:
            raise CalibrationError("Calibration file needs 'image_points'.")
        if "field_points" in data:
            field_points = data["field_points"]
        elif "landmarks" in data:
            known = field.landmarks()
            missing = [n for n in data["landmarks"] if n not in known]
            if missing:
                raise CalibrationError(f"Unknown landmark names: {missing}. Valid: {sorted(known)}")
            field_points = [known[n] for n in data["landmarks"]]
        else:
            raise CalibrationError("Calibration file needs 'field_points' or 'landmarks'.")
        image_points = np.asarray(data["image_points"], dtype=np.float64)
        if "frame_size" in data and tuple(data["frame_size"]) != tuple(video_size):
            sx, sy = video_size[0] / data["frame_size"][0], video_size[1] / data["frame_size"][1]
            logger.warning("Calibration was made at %s but video is %s: scaling points.", data["frame_size"], video_size)
            image_points = image_points * np.array([sx, sy])
        return cls(image_points, field_points, max_error_m)

    def homography(self, frame_index, frame=None):
        return self._matrix

    @property
    def image_points(self) -> np.ndarray:
        return self._image_points
