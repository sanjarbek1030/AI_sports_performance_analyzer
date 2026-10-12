"""Pitch geometry: landmarks for calibration, goal geometry, drawing helpers.

``FieldModel`` is the sport-agnostic interface (other sports implement it);
``FootballField`` is the standard association-football pitch.
Coordinates: x along the length, y along the width, metres, origin = top-left
corner of the pitch as seen from the main camera.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DISTANCE = 11.0
CENTER_CIRCLE_RADIUS = 9.15
_LINE_COLOR = (255, 255, 255)
_PITCH_COLOR = (60, 120, 50)


class FieldModel(ABC):
    length: float
    width: float

    @abstractmethod
    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named reference points usable for calibration."""

    @abstractmethod
    def draw_template(self, pixels_per_meter: float, margin_px: int = 10) -> np.ndarray:
        """Top-down BGR picture of the pitch."""

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin


class FootballField(FieldModel):
    def __init__(self, length: float = 105.0, width: float = 68.0, goal_width: float = 7.32) -> None:
        self.length, self.width, self.goal_width = float(length), float(width), float(goal_width)

    # ---------------------------------------------------------- goal geometry
    def goal_mouth(self, side: int) -> tuple[float, float, float]:
        """(goal_line_x, y_min, y_max) for the goal at x=0 (side=-1) or x=L (side=+1)."""
        cy = self.width / 2.0
        gx = self.length if side > 0 else 0.0
        return gx, cy - self.goal_width / 2.0, cy + self.goal_width / 2.0

    # ------------------------------------------------------------- landmarks
    def landmarks(self) -> dict[str, tuple[float, float]]:
        L, W, cy, cx = self.length, self.width, self.width / 2.0, self.length / 2.0
        pa_y = PENALTY_AREA_WIDTH / 2.0
        ga_y = GOAL_AREA_WIDTH / 2.0
        pts = {
            "corner_top_left": (0, 0), "corner_top_right": (L, 0),
            "corner_bottom_left": (0, W), "corner_bottom_right": (L, W),
            "halfway_top": (cx, 0), "halfway_bottom": (cx, W), "center_spot": (cx, cy),
            "center_circle_left": (cx - CENTER_CIRCLE_RADIUS, cy),
            "center_circle_right": (cx + CENTER_CIRCLE_RADIUS, cy),
            "center_circle_top": (cx, cy - CENTER_CIRCLE_RADIUS),
            "center_circle_bottom": (cx, cy + CENTER_CIRCLE_RADIUS),
        }
        for side, x0, sign in (("left", 0.0, 1), ("right", L, -1)):
            pts.update({
                f"{side}_penalty_box_top_outer": (x0, cy - pa_y),
                f"{side}_penalty_box_top_inner": (x0 + sign * PENALTY_AREA_DEPTH, cy - pa_y),
                f"{side}_penalty_box_bottom_outer": (x0, cy + pa_y),
                f"{side}_penalty_box_bottom_inner": (x0 + sign * PENALTY_AREA_DEPTH, cy + pa_y),
                f"{side}_goal_area_top_outer": (x0, cy - ga_y),
                f"{side}_goal_area_top_inner": (x0 + sign * GOAL_AREA_DEPTH, cy - ga_y),
                f"{side}_goal_area_bottom_outer": (x0, cy + ga_y),
                f"{side}_goal_area_bottom_inner": (x0 + sign * GOAL_AREA_DEPTH, cy + ga_y),
                f"{side}_penalty_spot": (x0 + sign * PENALTY_SPOT_DISTANCE, cy),
                f"{side}_goal_post_top": (x0, cy - self.goal_width / 2),
                f"{side}_goal_post_bottom": (x0, cy + self.goal_width / 2),
            })
        return {k: (float(v[0]), float(v[1])) for k, v in pts.items()}

    # --------------------------------------------------------------- drawing
    def line_segments(self) -> list[list[tuple[float, float]]]:
        """Polylines of the pitch markings (used by OpenCV and matplotlib drawers)."""
        L, W, cx, cy = self.length, self.width, self.length / 2, self.width / 2
        pa, ga = PENALTY_AREA_WIDTH / 2, GOAL_AREA_WIDTH / 2
        lines = [[(0, 0), (L, 0), (L, W), (0, W), (0, 0)], [(cx, 0), (cx, W)]]
        for x0, s in ((0.0, 1), (L, -1)):
            lines.append([(x0, cy - pa), (x0 + s * PENALTY_AREA_DEPTH, cy - pa),
                          (x0 + s * PENALTY_AREA_DEPTH, cy + pa), (x0, cy + pa)])
            lines.append([(x0, cy - ga), (x0 + s * GOAL_AREA_DEPTH, cy - ga),
                          (x0 + s * GOAL_AREA_DEPTH, cy + ga), (x0, cy + ga)])
        angles = np.linspace(0, 2 * np.pi, 73)
        lines.append([(cx + CENTER_CIRCLE_RADIUS * np.cos(a), cy + CENTER_CIRCLE_RADIUS * np.sin(a)) for a in angles])
        return lines

    def draw_template(self, pixels_per_meter: float, margin_px: int = 10) -> np.ndarray:
        w = int(self.length * pixels_per_meter) + 2 * margin_px
        h = int(self.width * pixels_per_meter) + 2 * margin_px
        image = np.full((h, w, 3), _PITCH_COLOR, dtype=np.uint8)
        for line in self.line_segments():
            pts = np.array([[margin_px + x * pixels_per_meter, margin_px + y * pixels_per_meter] for x, y in line], np.int32)
            cv2.polylines(image, [pts], False, _LINE_COLOR, 1, cv2.LINE_AA)
        return image

    def draw_matplotlib(self, ax) -> None:
        for line in self.line_segments():
            xs, ys = zip(*line)
            ax.plot(xs, ys, color="white", linewidth=1.0, alpha=0.9)
        ax.set_xlim(0, self.length)
        ax.set_ylim(self.width, 0)  # y grows downward, like the camera view
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
