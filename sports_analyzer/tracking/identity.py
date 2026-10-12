"""PlayerIdentityManager: keeps ONE stable player id across tracker id changes.

Problem solved : ByteTrack/BoT-SORT forget a track after ``track_buffer_s``.
                 A player hidden longer comes back with a NEW tracker id,
                 which would fragment statistics.
Receives       : tracked detections + jersey colour feature per detection.
Produces       : a stable ``player_id`` per detection.
Method         : a new tracker id is linked to a recently-lost identity only
                 if ALL hold: lost < relink_max_gap_s ago, the reappearance
                 point is physically reachable (distance in player-heights),
                 jersey colour is similar. Otherwise a new id is issued.
Limits         : no appearance embeddings (ReID) yet - see README "Future".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import TrackingConfig
from ..models import Detection, Point

logger = logging.getLogger(__name__)
_COLOR_EMA = 0.8


@dataclass
class _LastSeen:
    frame: int
    timestamp: float
    ground: Point
    box_height: float
    color: Optional[np.ndarray]
    role: str


class PlayerIdentityManager:
    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self._alias: dict[int, int] = {}
        self._last_seen: dict[int, _LastSeen] = {}
        self._next_id = 1
        self.relink_count = 0

    def knows(self, tracker_id: Optional[int]) -> bool:
        """True if this tracker id already maps to a player (no colour feature needed)."""
        return tracker_id in self._alias

    def assign(
        self, detections: list[Detection], features: list[Optional[np.ndarray]], timestamp: float
    ) -> list[int]:
        """Return a stable player id for each detection (same order)."""
        ids: list[Optional[int]] = [None] * len(detections)
        new_positions: list[int] = []
        for i, det in enumerate(detections):
            if det.track_id in self._alias:
                ids[i] = self._alias[det.track_id]
            else:
                new_positions.append(i)

        active = {pid for pid in ids if pid is not None}
        if new_positions:
            links = self._relink(detections, features, new_positions, active, timestamp) if self.cfg.relink_enabled else {}
            for i in new_positions:
                if i in links:
                    ids[i] = links[i]
                    self.relink_count += 1
                    logger.debug("Re-linked tracker id %s -> player %d", detections[i].track_id, links[i])
                else:
                    ids[i] = self._next_id
                    self._next_id += 1
                self._alias[detections[i].track_id] = ids[i]

        for det, feat, pid in zip(detections, features, ids):
            self._remember(pid, det, feat, timestamp)
        return [int(pid) for pid in ids]

    # ------------------------------------------------------------------
    def _relink(self, detections, features, new_positions, active, timestamp) -> dict[int, int]:
        candidates = []
        for pid, seen in self._last_seen.items():
            gap = timestamp - seen.timestamp
            if pid in active or gap <= 0 or gap > self.cfg.relink_max_gap_s:
                continue
            candidates.append((pid, seen, gap))
        pairs = []
        for i in new_positions:
            det, feat = detections[i], features[i]
            gx, gy = det.ground_point
            for pid, seen, gap in candidates:
                if seen.role != det.role:
                    continue
                allowed = seen.box_height * (self.cfg.relink_base_heights + self.cfg.relink_heights_per_s * gap)
                dist = float(np.hypot(gx - seen.ground[0], gy - seen.ground[1]))
                if dist > allowed:
                    continue
                if feat is not None and seen.color is not None:
                    if float(np.linalg.norm(feat - seen.color)) > self.cfg.relink_max_color_distance:
                        continue
                pairs.append((dist / allowed, i, pid))
        links: dict[int, int] = {}
        used: set[int] = set()
        for _, i, pid in sorted(pairs):  # greedy, best (closest) first, one-to-one
            if i not in links and pid not in used:
                links[i] = pid
                used.add(pid)
        return links

    def _remember(self, pid: int, det: Detection, feat: Optional[np.ndarray], timestamp: float) -> None:
        previous = self._last_seen.get(pid)
        color = previous.color if previous else None
        if feat is not None:
            color = feat.copy() if color is None else _COLOR_EMA * color + (1 - _COLOR_EMA) * feat
        self._last_seen[pid] = _LastSeen(det.frame, timestamp, det.ground_point, det.height, color, det.role)
