"""MiniMap: top-down tactical view in the same field coordinates as the analytics."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ..field.football_field import FieldModel
from ..models import PlayerObservation

_MARGIN_PX = 8
_PLAYER_RADIUS = 5
_BALL_RADIUS = 4
_UNKNOWN_COLOR = (170, 170, 170)


class MiniMap:
    def __init__(self, field: FieldModel, width_px: int, team_colors: Sequence[Sequence[int]], show_ids: bool) -> None:
        self.field = field
        self.ppm = (width_px - 2 * _MARGIN_PX) / field.length
        self.template = field.draw_template(self.ppm, _MARGIN_PX)
        self.colors = [tuple(int(c) for c in col) for col in team_colors]
        self.show_ids = show_ids

    def _to_px(self, xy) -> tuple[int, int]:
        return int(_MARGIN_PX + xy[0] * self.ppm), int(_MARGIN_PX + xy[1] * self.ppm)

    def render(self, players: list[PlayerObservation], ball_xy: Optional[tuple[float, float]]) -> np.ndarray:
        image = self.template.copy()
        for o in players:
            if o.smoothed_xy is None or not o.on_pitch or o.team_id < -1:
                continue
            color = self.colors[o.team_id] if 0 <= o.team_id < len(self.colors) else _UNKNOWN_COLOR
            center = self._to_px(o.smoothed_xy)
            cv2.circle(image, center, _PLAYER_RADIUS, color, -1, cv2.LINE_AA)
            cv2.circle(image, center, _PLAYER_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)
            if self.show_ids:
                cv2.putText(image, str(o.player_id), (center[0] + 6, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, (255, 255, 255), 1, cv2.LINE_AA)
        if ball_xy is not None and np.isfinite(ball_xy).all() and self.field.contains(*ball_xy, 5.0):
            center = self._to_px(ball_xy)
            cv2.circle(image, center, _BALL_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, center, _BALL_RADIUS, (0, 0, 0), 1, cv2.LINE_AA)
        return image
