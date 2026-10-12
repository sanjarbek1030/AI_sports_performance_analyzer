"""PlayerDetector: YOLO person/player detection.

Problem solved : find every player (and optionally goalkeeper / referee) per frame.
Receives       : BGR frames (batched), frame indices, timestamps.
Produces       : ``Detection`` objects: bbox, confidence, class, frame, timestamp.
Interacts with: PlayerTracker consumes the detections. Boxes are always
                 returned in *original-frame* pixels even if inference ran at
                 a reduced ``process_width``.

The ground position of a player is the bottom-centre of the box
(``Detection.ground_point``), never the box centre.
"""
from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from ..config import DetectionConfig, VideoConfig
from ..exceptions import ModelInferenceError, ModelLoadError
from ..models import Detection, DetectionRole
from .yolo_loader import load_yolo, match_class_ids

logger = logging.getLogger(__name__)
_STRIDE_MULTIPLE = 32


class PlayerDetector:
    def __init__(self, det_cfg: DetectionConfig, video_cfg: VideoConfig, device: str) -> None:
        self.cfg = det_cfg
        self.video_cfg = video_cfg
        self.device = device
        self.model = load_yolo(det_cfg.player_model, device)
        self._role_by_class = self._build_role_map()
        self._class_ids = sorted(self._role_by_class)
        self._half = det_cfg.half_precision and device.startswith("cuda")

    def _build_role_map(self) -> dict[int, str]:
        names = self.model.names
        roles: dict[int, str] = {}
        for role, wanted in (
            (DetectionRole.PLAYER, self.cfg.player_class_names),
            (DetectionRole.GOALKEEPER, self.cfg.goalkeeper_class_names),
            (DetectionRole.REFEREE, self.cfg.referee_class_names),
        ):
            for class_id in match_class_ids(names, wanted):
                roles[class_id] = role
        if not roles:
            raise ModelLoadError(
                f"Model has none of the configured player classes. Model classes: {list(names.values())}"
            )
        return roles

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        pw, ph = self.video_cfg.process_width, self.video_cfg.process_height
        if pw and ph:
            return pw, ph
        if pw:
            return pw, max(1, round(pw * height / width))
        if ph:
            return max(1, round(ph * width / height)), ph
        return width, height

    def detect_batch(
        self, frames: Sequence[np.ndarray], frame_indices: Sequence[int], timestamps: Sequence[float]
    ) -> list[list[Detection]]:
        if not frames:
            return []
        prepared, scales = [], []
        for frame in frames:
            h, w = frame.shape[:2]
            tw, th = self._target_size(w, h)
            if (tw, th) != (w, h):
                prepared.append(cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA))
            else:
                prepared.append(frame)
            scales.append((w / tw, h / th))
        if self.video_cfg.process_width or self.video_cfg.process_height:
            longest = max(prepared[0].shape[:2])
            imgsz = int(np.ceil(longest / _STRIDE_MULTIPLE) * _STRIDE_MULTIPLE)
        else:
            imgsz = self.cfg.player_imgsz
        try:
            results = self.model.predict(
                source=prepared, conf=self.cfg.low_confidence_threshold, iou=self.cfg.iou_threshold,
                imgsz=imgsz, device=self.device, classes=self._class_ids, half=self._half, verbose=False,
            )
        except Exception as exc:
            raise ModelInferenceError(f"Player detection failed: {exc}") from exc

        output: list[list[Detection]] = []
        for result, (sx, sy), frame, idx, ts in zip(results, scales, frames, frame_indices, timestamps):
            output.append(self._parse(result, sx, sy, frame.shape, idx, ts))
        return output

    def _parse(self, result, sx: float, sy: float, shape, frame_idx: int, timestamp: float) -> list[Detection]:
        height, width = shape[:2]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        detections: list[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            x1, x2 = np.clip([x1 * sx, x2 * sx], 0, width - 1)
            y1, y2 = np.clip([y1 * sy, y2 * sy], 0, height - 1)
            if min(x2 - x1, y2 - y1) < self.cfg.min_player_box_px:
                continue
            detections.append(Detection(
                frame=frame_idx, timestamp=timestamp, bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(c), class_id=int(k), class_name=str(self.model.names[int(k)]),
                role=self._role_by_class.get(int(k), DetectionRole.PLAYER),
            ))
        return detections
