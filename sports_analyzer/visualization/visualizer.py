"""Visualizer: draws the analytics onto the original frames (render pass).

Shows (clean mode): player boxes with persistent id / team / speed,
possession marker, player and ball trails, ball marker, match clock, live
possession bar, recent-event banner and the mini-map.
Debug mode adds detector confidence, tracker id, field coordinates, ball-player
distance, event confidence, raw ball detection and calibration points.

Everything drawn comes from the precomputed ``AnalysisResult``; the Visualizer
never recomputes statistics. Overlay sizes scale with the frame height so
4K and 480p both stay readable.
"""
from __future__ import annotations

import bisect
from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np

from ..config import Config
from ..models import (DetectionRole, Event, EventType, FrameData, PlayerObservation, PlayerTrack,
                      TEAM_REFEREE, team_label)
from ..utils import format_timestamp
from .minimap import MiniMap

_REFERENCE_HEIGHT = 720.0
_UNKNOWN_COLOR = (170, 170, 170)
_REFEREE_COLOR = (0, 215, 255)
_POSSESSION_COLOR = (0, 255, 255)
_BALL_COLOR = (255, 255, 255)
_HUD_EVENTS = {EventType.PASS, EventType.SHOT, EventType.GOAL, EventType.TACKLE_CANDIDATE,
               EventType.INTERCEPTION_CANDIDATE, EventType.OUT_OF_BOUNDS}
_MAX_EVENTS_SHOWN = 2
_PANEL_ALPHA = 0.55
_TRAIL_ALPHA = 0.6


class Visualizer:
    def __init__(self, config: Config, frames: list[FrameData], tracks: dict[int, PlayerTrack], events: list[Event],
                 possession_states, spells, field, mapper, frame_size: tuple[int, int], stride: int) -> None:
        self.cfg, self.vis = config, config.visualization
        self.frames, self.tracks, self.stride = frames, tracks, stride
        self.events = [e for e in events if e.event_type in _HUD_EVENTS]
        self._event_times = [e.timestamp for e in self.events]
        self.states = possession_states
        self.mapper = mapper
        self.width, self.height = frame_size
        self.scale = self.height / _REFERENCE_HEIGHT
        self.minimap: Optional[MiniMap] = None
        if self.vis.show_minimap and field is not None and mapper is not None and mapper.is_calibrated:
            self.minimap = MiniMap(field, int(self.vis.minimap_width_frac * self.width), self.vis.team_colors_bgr,
                                   self.vis.minimap_show_ids)
        self._cum = self._cumulative_possession(spells, len(frames))
        self._player_trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.vis.trail_length))
        self._ball_trail: deque = deque(maxlen=self.vis.ball_trail_length)
        self._last_pos = -1
        self._hide_off_pitch = mapper is not None and mapper.is_calibrated

    # -------------------------------------------------------------- public
    def render(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        if not self.frames:
            return frame
        pos = min(frame_index // self.stride, len(self.frames) - 1)
        fd = self.frames[pos]
        if pos != self._last_pos:
            self._update_trails(fd)
            self._last_pos = pos
        out = frame
        if self.vis.show_trails:
            out = self._draw_trails(out, fd)
        for obs in fd.players:
            if self._hide_off_pitch and not obs.on_pitch:
                continue
            self._draw_player(out, obs, pos)
        self._draw_ball(out, fd)
        self._draw_hud(out, fd, pos)
        if self.minimap is not None:
            self._draw_minimap(out, fd)
        if self.cfg.debug_mode:
            self._draw_debug(out, fd, pos)
        return out

    # ------------------------------------------------------------- helpers
    def _team_color(self, team_id: int) -> tuple[int, int, int]:
        if team_id == TEAM_REFEREE:
            return _REFEREE_COLOR
        if 0 <= team_id < len(self.vis.team_colors_bgr):
            return tuple(int(c) for c in self.vis.team_colors_bgr[team_id])
        return _UNKNOWN_COLOR

    def _px(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    def _cumulative_possession(self, spells, n_frames) -> np.ndarray:
        cum = np.zeros((2, n_frames))
        if not n_frames or not self.frames:
            return cum
        times = np.array([fd.timestamp for fd in self.frames])
        for s in spells:
            if s.team_id in (0, 1):
                lo, hi = np.searchsorted(times, s.start_time), np.searchsorted(times, s.end_time, side="right")
                cum[s.team_id, lo:hi] = 1
        return np.cumsum(cum, axis=1)

    def _update_trails(self, fd: FrameData) -> None:
        for obs in fd.players:
            self._player_trails[obs.player_id].append((int(obs.ground_image[0]), int(obs.ground_image[1])))
        if fd.ball is not None:
            self._ball_trail.append((int(fd.ball.center[0]), int(fd.ball.center[1])))

    def _text(self, img, text, org, color=(255, 255, 255), size=0.5, thickness=1, bg=(0, 0, 0)) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        size, thickness = size * self.scale, max(1, int(round(thickness * self.scale)))
        (w, h), base = cv2.getTextSize(text, font, size, thickness)
        x, y = int(org[0]), int(org[1])
        pad = self._px(2)
        cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg, -1)
        cv2.putText(img, text, (x, y), font, size, color, thickness, cv2.LINE_AA)

    # --------------------------------------------------------------- trails
    def _draw_trails(self, frame: np.ndarray, fd: FrameData) -> np.ndarray:
        layer = frame.copy()
        thickness = self._px(2)
        for obs in fd.players:
            pts = list(self._player_trails[obs.player_id])
            color = self._team_color(obs.team_id)
            for a, b in zip(pts[:-1], pts[1:]):
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) < self.width * 0.15:  # skip ID-jump segments
                    cv2.line(layer, a, b, color, thickness, cv2.LINE_AA)
        pts = list(self._ball_trail)
        for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            fade = (k + 1) / max(len(pts) - 1, 1)
            cv2.line(layer, a, b, tuple(int(c * fade) for c in _BALL_COLOR), self._px(2), cv2.LINE_AA)
        return cv2.addWeighted(layer, _TRAIL_ALPHA, frame, 1 - _TRAIL_ALPHA, 0)

    # -------------------------------------------------------------- players
    def _draw_player(self, img: np.ndarray, obs: PlayerObservation, pos: int) -> None:
        color = self._team_color(obs.team_id)
        x1, y1, x2, y2 = (int(v) for v in obs.bbox)
        if self.cfg.debug_mode:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, self._px(1))
        cx, gy = int(obs.ground_image[0]), int(obs.ground_image[1])
        axes = (max(4, (x2 - x1) // 2), max(2, (x2 - x1) // 5))
        cv2.ellipse(img, (cx, gy), axes, 0, -45, 235, color, self._px(2), cv2.LINE_AA)
        label = f"#{obs.player_id}"
        if obs.role == DetectionRole.REFEREE:
            label = "REF"
        elif obs.speed_kmh is not None:
            label += f" {obs.speed_kmh:.0f}km/h"
        if self.cfg.debug_mode:
            label += f" c{obs.confidence:.2f} t{obs.tracker_id}"
        self._text(img, label, (x1, max(y1 - self._px(6), self._px(14))), (255, 255, 255), 0.45, 1, color)
        state = self.states[pos] if pos < len(self.states) else None
        if state is not None and state.holder_id == obs.player_id:
            tip = (cx, max(y1 - self._px(22), self._px(6)))
            s = self._px(7)
            pts = np.array([[tip[0], tip[1]], [tip[0] - s, tip[1] - s * 2], [tip[0] + s, tip[1] - s * 2]], np.int32)
            cv2.fillPoly(img, [pts], _POSSESSION_COLOR, cv2.LINE_AA)

    # ----------------------------------------------------------------- ball
    def _draw_ball(self, img: np.ndarray, fd: FrameData) -> None:
        ball = fd.ball
        if ball is None:
            return
        center = (int(ball.center[0]), int(ball.center[1]))
        radius = self._px(6)
        if ball.predicted:
            cv2.circle(img, center, radius, _BALL_COLOR, self._px(1), cv2.LINE_AA)
        else:
            cv2.circle(img, center, radius, _BALL_COLOR, -1, cv2.LINE_AA)
            cv2.circle(img, center, radius, (0, 0, 0), self._px(1), cv2.LINE_AA)
        tip = (center[0], center[1] - radius - self._px(8))
        pts = np.array([[tip[0], tip[1] + self._px(6)], [tip[0] - self._px(5), tip[1]], [tip[0] + self._px(5), tip[1]]], np.int32)
        cv2.fillPoly(img, [pts], _BALL_COLOR, cv2.LINE_AA)

    # ------------------------------------------------------------------ HUD
    def _draw_hud(self, img: np.ndarray, fd: FrameData, pos: int) -> None:
        clock = format_timestamp(fd.timestamp + self.vis.match_clock_offset_s)
        self._text(img, f"TIME {clock}", (self._px(12), self._px(26)), (255, 255, 255), 0.6, 2, (30, 30, 30))
        a, b = self._cum[0, pos], self._cum[1, pos]
        total = a + b
        names = self.vis.team_names
        bar_w, bar_h = int(self.width * 0.3), self._px(16)
        x0, y0 = (self.width - bar_w) // 2, self._px(10)
        cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (40, 40, 40), -1)
        if total > 0:
            split = x0 + int(bar_w * a / total)
            cv2.rectangle(img, (x0, y0), (split, y0 + bar_h), self._team_color(0), -1)
            cv2.rectangle(img, (split, y0), (x0 + bar_w, y0 + bar_h), self._team_color(1), -1)
            label = f"{names[0]} {100 * a / total:.0f}%  |  {100 * b / total:.0f}% {names[1]}"
        else:
            label = "possession: n/a"
        self._text(img, label, (x0, y0 + bar_h + self._px(18)), (255, 255, 255), 0.45, 1, (30, 30, 30))
        now = fd.timestamp
        hi = bisect.bisect_right(self._event_times, now)
        shown = [e for e in self.events[:hi] if now - e.timestamp <= self.vis.event_display_s][-_MAX_EVENTS_SHOWN:]
        for k, e in enumerate(reversed(shown)):
            self._text(img, self._event_text(e), (self._px(12), self.height - self._px(14) - k * self._px(26)),
                       (255, 255, 255), 0.55, 1, self._team_color(e.team_id) if e.team_id is not None else (60, 60, 60))

    def _event_text(self, e: Event) -> str:
        who = f"#{e.player_id}" if e.player_id is not None else "?"
        text = e.event_type.replace("_", " ").upper()
        if e.event_type == EventType.PASS:
            text += f" {who}->#{e.metadata.get('to_player')}" + ("" if e.metadata.get("successful") else " (incomplete)")
        elif e.event_type == EventType.SHOT:
            text += f" {who} [{e.metadata.get('outcome')}]"
        else:
            text += f" {who}"
        if e.team_id is not None and e.team_id >= 0:
            text += f" {team_label(e.team_id, self.vis.team_names)}"
        suffix = f" ({e.confidence:.2f}" + (", inferred)" if self.cfg.debug_mode else ")")
        return text + suffix

    def _draw_minimap(self, img: np.ndarray, fd: FrameData) -> None:
        picture = self.minimap.render(fd.players, fd.ball.field_xy if fd.ball else None)
        h, w = picture.shape[:2]
        margin = self._px(10)
        x0, y0 = self.width - w - margin, self.height - h - margin
        if x0 < 0 or y0 < 0:
            return
        roi = img[y0:y0 + h, x0:x0 + w]
        img[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(picture, 0.85, roi, 0.15, 0)

    # ---------------------------------------------------------------- debug
    def _draw_debug(self, img: np.ndarray, fd: FrameData, pos: int) -> None:
        if self.mapper is not None and self.mapper.provider.image_points is not None:
            for k, (x, y) in enumerate(self.mapper.provider.image_points):
                cv2.circle(img, (int(x), int(y)), self._px(6), (0, 0, 255), -1)
                self._text(img, f"C{k}", (int(x) + 8, int(y) - 8), (0, 0, 255), 0.45)
        for o in fd.players:
            if o.field_xy is not None:
                self._text(img, f"({o.field_xy[0]:.1f},{o.field_xy[1]:.1f})m",
                           (int(o.bbox[0]), int(o.bbox[3]) + self._px(14)), (200, 255, 200), 0.38)
        if fd.ball is not None and fd.ball.raw_detection is not None:
            x1, y1, x2, y2 = (int(v) for v in fd.ball.raw_detection.bbox)
            cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), self._px(1))
            self._text(img, f"ball {fd.ball.raw_detection.confidence:.2f}", (x2 + 6, y1), (0, 0, 255), 0.4)
        state = self.states[pos] if pos < len(self.states) else None
        if state is not None:
            dist = "n/a" if state.nearest_distance is None else f"{state.nearest_distance:.1f}m"
            self._text(img, f"state={state.state.value} holder={state.holder_id} nearest={state.nearest_id} d={dist}",
                       (self._px(12), self._px(52)), (0, 255, 255), 0.45)
