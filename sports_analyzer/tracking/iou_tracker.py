"""Built-in two-stage IoU tracker (ByteTrack-style, no external dependency).

Used as (a) automatic fallback when the Ultralytics trackers cannot start and
(b) a deterministic tracker for unit tests. Association = Hungarian matching on
(1 - IoU) with a constant-velocity box prediction; high-confidence detections
are matched first, then low-confidence ones are used to keep existing tracks
alive through partial occlusion.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..models import Detection

_LOW_STAGE_MAX_COST = 0.5
_VELOCITY_BLEND = 0.3


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for (N,4) and (M,4) xyxy arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    velocity: np.ndarray            # per-frame box shift
    hits: int = 1
    frames_since_update: int = 0

    def predict(self) -> np.ndarray:
        return self.bbox + self.velocity * (self.frames_since_update + 1)


class IoUTracker:
    def __init__(self, high_threshold: float, low_threshold: float, new_track_threshold: float,
                 match_threshold: float, max_age_frames: int, min_hits: int) -> None:
        self.high = high_threshold
        self.low = low_threshold
        self.new_thr = new_track_threshold
        self.match_thr = match_threshold
        self.max_age = max_age_frames
        self.min_hits = min_hits
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Associate detections with tracks; return detections that carry a ``track_id``."""
        high_idx = [i for i, d in enumerate(detections) if d.confidence >= self.high]
        low_idx = [i for i, d in enumerate(detections) if self.low <= d.confidence < self.high]
        boxes = np.array([d.bbox for d in detections], dtype=float).reshape(-1, 4)
        predictions = np.array([t.predict() for t in self._tracks], dtype=float).reshape(-1, 4)

        matched_t, matched_d = set(), set()
        pairs = self._associate(list(range(len(self._tracks))), high_idx, predictions, boxes, self.match_thr)
        for ti, di in pairs:
            matched_t.add(ti); matched_d.add(di)
        remaining = [i for i in range(len(self._tracks)) if i not in matched_t and self._tracks[i].hits >= self.min_hits]
        for ti, di in self._associate(remaining, low_idx, predictions, boxes, _LOW_STAGE_MAX_COST):
            pairs.append((ti, di)); matched_t.add(ti); matched_d.add(di)

        output: list[Detection] = []
        for ti, di in pairs:
            track = self._tracks[ti]
            new_box = boxes[di]
            gap = track.frames_since_update + 1
            track.velocity = (1 - _VELOCITY_BLEND) * track.velocity + _VELOCITY_BLEND * (new_box - track.bbox) / gap
            track.bbox, track.hits, track.frames_since_update = new_box, track.hits + 1, 0
            if track.hits >= self.min_hits:
                output.append(replace(detections[di], track_id=track.track_id))

        for i, track in enumerate(self._tracks):
            if i not in matched_t:
                track.frames_since_update += 1
        for di in high_idx:
            if di not in matched_d and detections[di].confidence >= self.new_thr:
                self._tracks.append(_Track(self._next_id, boxes[di].copy(), np.zeros(4)))
                if self.min_hits <= 1:
                    output.append(replace(detections[di], track_id=self._next_id))
                self._next_id += 1
        self._tracks = [
            t for t in self._tracks
            if t.frames_since_update <= (self.max_age if t.hits >= self.min_hits else 1)
        ]
        return output

    @staticmethod
    def _associate(track_idx, det_idx, predictions, boxes, max_cost) -> list[tuple[int, int]]:
        if not track_idx or not det_idx:
            return []
        cost = 1.0 - iou_matrix(predictions[track_idx], boxes[det_idx])
        rows, cols = linear_sum_assignment(cost)
        return [(track_idx[r], det_idx[c]) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]
