"""PlayerTracker: persistent multi-object tracking (ByteTrack / BoT-SORT).

Problem solved : give the same player the same id from frame to frame, and
                 survive short occlusions (the tracker keeps lost tracks
                 alive for ``track_buffer_s`` seconds).
Receives       : ``Detection`` list (incl. low-confidence boxes) + the frame.
Produces       : the same detections annotated with ``track_id``.
Interacts with: PlayerIdentityManager (long-term re-linking) runs after this.

The Ultralytics ``BYTETracker`` / ``BOTSORT`` classes are driven directly with
our own detections (no second inference). If they cannot be created or fail on
the first frame (API differences between versions), we fall back to the
built-in ``IoUTracker`` and log a warning instead of crashing.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional

import numpy as np

from ..config import Config
from ..models import Detection
from .iou_tracker import IoUTracker, iou_matrix

logger = logging.getLogger(__name__)

_TRACKER_CLASSES = {"bytetrack": "BYTETracker", "botsort": "BOTSORT"}


class PlayerTracker:
    def __init__(self, config: Config, fps: float) -> None:
        self.cfg = config
        self.fps = fps
        det, trk = config.detection, config.tracking
        self._buffer_frames = max(1, int(round(trk.track_buffer_s * fps / config.video.analysis_stride)))
        self._ultra = None
        self._backend_name = "iou"
        if trk.tracker in _TRACKER_CLASSES:
            self._ultra = self._build_ultralytics(trk.tracker)
        if self._ultra is None:
            self._fallback: Optional[IoUTracker] = IoUTracker(
                det.confidence_threshold, det.low_confidence_threshold, trk.new_track_threshold,
                trk.match_threshold, self._buffer_frames, trk.min_track_hits,
            )
        else:
            self._fallback = None
        logger.info("Player tracker backend: %s (buffer=%d analysed frames)", self._backend_name, self._buffer_frames)

    # ------------------------------------------------------------ backends
    def _build_ultralytics(self, name: str):
        try:
            import ultralytics.trackers as ut
            from ultralytics.utils import IterableSimpleNamespace

            args = self._tracker_args(name)
            tracker_cls = getattr(ut, _TRACKER_CLASSES[name])
            tracker = tracker_cls(IterableSimpleNamespace(**args), frame_rate=max(1, int(round(self.fps / self.cfg.video.analysis_stride))))
            self._backend_name = name
            return tracker
        except Exception as exc:  # version differences, missing package ...
            logger.warning("Could not start Ultralytics %s tracker (%s). Using built-in IoU tracker.", name, exc)
            return None

    def _tracker_args(self, name: str) -> dict:
        det, trk = self.cfg.detection, self.cfg.tracking
        args = {
            "tracker_type": name,
            "track_high_thresh": det.confidence_threshold,
            "track_low_thresh": det.low_confidence_threshold,
            "new_track_thresh": trk.new_track_threshold,
            "track_buffer": self._buffer_frames,
            "match_thresh": trk.match_threshold,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.25,
            "with_reid": False,
            "model": "auto",
        }
        # Prefer the defaults shipped with the installed Ultralytics, then override ours.
        try:
            import ultralytics, yaml
            from pathlib import Path

            path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{name}.yaml"
            shipped = yaml.safe_load(path.read_text())
            shipped.update({k: v for k, v in args.items() if k in shipped})
            return shipped
        except Exception:
            return args

    # -------------------------------------------------------------- update
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Detection]:
        if self._ultra is not None:
            try:
                return self._update_ultralytics(detections, frame)
            except Exception as exc:
                logger.warning("Ultralytics tracker failed (%s); switching to built-in IoU tracker.", exc)
                self._ultra = None
                det, trk = self.cfg.detection, self.cfg.tracking
                self._fallback = IoUTracker(
                    det.confidence_threshold, det.low_confidence_threshold, trk.new_track_threshold,
                    trk.match_threshold, self._buffer_frames, trk.min_track_hits,
                )
                self._backend_name = "iou"
        assert self._fallback is not None
        return self._fallback.update(detections)

    def _update_ultralytics(self, detections: list[Detection], frame: np.ndarray) -> list[Detection]:
        if detections:
            xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
            conf = np.array([d.confidence for d in detections], dtype=np.float32)
            cls = np.array([d.class_id for d in detections], dtype=np.float32)
        else:
            xyxy = np.zeros((0, 4), np.float32); conf = np.zeros(0, np.float32); cls = np.zeros(0, np.float32)
        xywh = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                         xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1) if len(xyxy) else xyxy
        adapter = SimpleNamespace(conf=conf, xywh=xywh, xyxy=xyxy, cls=cls)
        tracks = self._ultra.update(adapter, frame)
        output: list[Detection] = []
        rows = np.asarray(tracks)
        if rows.size == 0:
            return output
        for row in rows.reshape(len(rows), -1):
            track_id = int(row[4])
            idx = int(row[7]) if len(row) >= 8 else -1
            if 0 <= idx < len(detections):
                source = detections[idx]
            else:  # index missing: match by best IoU
                ious = iou_matrix(np.array([row[:4]]), xyxy)
                if ious.size == 0 or ious.max() < 0.3:
                    continue
                source = detections[int(ious.argmax())]
            output.append(replace(source, track_id=track_id))
        return output
