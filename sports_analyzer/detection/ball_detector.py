"""BallDetector: dedicated small-object ball detection.

Problem solved : the ball is tiny (often < 15 px), blurred and frequently
                 occluded, so it needs its own inference settings: native
                 resolution, larger ``imgsz`` and a *low* confidence floor.
Receives       : full-resolution BGR frames.
Produces       : up to ``max_ball_candidates`` ``BallDetection`` per frame.
Interacts with: BallTracker chooses among the candidates using temporal
                 consistency; this class deliberately does NOT pick one.
Shape filters reject obvious false positives (huge or very non-square boxes).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..config import DetectionConfig
from ..exceptions import ModelInferenceError, ModelLoadError
from ..models import BallDetection
from .yolo_loader import load_yolo, match_class_ids

logger = logging.getLogger(__name__)
_MAX_ASPECT = 2.5


class BallDetector:
    def __init__(self, det_cfg: DetectionConfig, device: str) -> None:
        self.cfg = det_cfg
        self.device = device
        self.model = load_yolo(det_cfg.ball_model or det_cfg.player_model, device)
        self._class_ids = match_class_ids(self.model.names, det_cfg.ball_class_names)
        if not self._class_ids:
            raise ModelLoadError(
                f"Ball model has no class named {det_cfg.ball_class_names}. "
                f"Model classes: {list(self.model.names.values())}"
            )
        self._half = det_cfg.half_precision and device.startswith("cuda")

    def detect_batch(
        self, frames: Sequence[np.ndarray], frame_indices: Sequence[int], timestamps: Sequence[float]
    ) -> list[list[BallDetection]]:
        if not frames:
            return []
        try:
            results = self.model.predict(
                source=list(frames), conf=self.cfg.ball_confidence_threshold, iou=self.cfg.iou_threshold,
                imgsz=self.cfg.ball_imgsz, device=self.device, classes=self._class_ids,
                half=self._half, verbose=False,
            )
        except Exception as exc:
            raise ModelInferenceError(f"Ball detection failed: {exc}") from exc
        return [
            self._parse(result, frame.shape[1], idx, ts)
            for result, frame, idx, ts in zip(results, frames, frame_indices, timestamps)
        ]

    def _parse(self, result, frame_width: int, frame_idx: int, timestamp: float) -> list[BallDetection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        candidates: list[BallDetection] = []
        for (x1, y1, x2, y2), conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            w, h = float(x2 - x1), float(y2 - y1)
            if max(w, h) > self.cfg.ball_max_size_frac * frame_width:
                continue
            if max(w, h) < self.cfg.ball_min_size_px or max(w, h) / max(min(w, h), 1e-6) > _MAX_ASPECT:
                continue
            candidates.append(BallDetection(frame_idx, timestamp, (float(x1), float(y1), float(x2), float(y2)), float(conf)))
        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return candidates[: self.cfg.max_ball_candidates]
