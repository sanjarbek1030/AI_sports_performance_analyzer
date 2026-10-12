"""Fake detectors + a synthetic top-down match video for end-to-end testing.

Scale: 10 px per metre, so the pitch (105 x 68 m) is a 1050 x 680 image and the
true homography is a pure scaling. Only the YOLO detectors are replaced; video
I/O, tracking, team classification, calibration, analysis, rendering and
reporting are the real code.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sports_analyzer.models import BallDetection, Detection

PX_PER_M = 10
WIDTH, HEIGHT = 1050, 680
FPS = 25.0
DURATION_S = 12.0
TEAM_COLORS = {0: (235, 235, 235), 1: (200, 60, 30)}   # white vs blue (BGR)
PLAYER_W, PLAYER_H = 24, 44


def _stand(x, y):
    return lambda t: (x, y)


def _runner(x0, y, speed):
    return lambda t: (x0 + speed * min(max(t - 2.0, 0.0), 3.0) + 0.0 * t, y)  # sprints from t=2s to t=5s


def players():
    """id -> (team, position fn in metres). Team 0 attacks +x."""
    p = {1: (0, _stand(60, 34)), 2: (0, _stand(75, 34)), 3: (0, _runner(30, 15, 8.0)), 4: (0, _stand(40, 50)),
         5: (0, _stand(45, 20)), 6: (1, _stand(90, 45)), 7: (1, _stand(85, 25)), 8: (1, _stand(95, 34)),
         9: (1, _stand(55, 60)), 10: (1, _stand(65, 8))}
    return p


def ball_position(t):
    """Possession by 1, pass to 2, shot at 5.0 s that goes in, then stays in the net."""
    if t < 2.0:
        return (60.6, 34.0)
    if t < 3.0:
        return (60.6 + 15.0 * (t - 2.0), 34.0)
    if t < 5.0:
        return (75.6, 34.0)
    if t < 6.4:
        return (75.6 + (105.6 - 75.6) * (t - 5.0) / 1.4, 34.0 + 1.0 * (t - 5.0) / 1.4)
    return (106.6, 35.0)


def render_frame(t: float) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (60, 130, 50), np.uint8)
    cv2.line(frame, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), (255, 255, 255), 2)
    for pid, (team, fn) in players().items():
        x, y = fn(t)
        gx, gy = int(x * PX_PER_M), int(y * PX_PER_M)
        cv2.rectangle(frame, (gx - PLAYER_W // 2, gy - PLAYER_H), (gx + PLAYER_W // 2, gy), TEAM_COLORS[team], -1)
    bx, by = ball_position(t)
    cv2.circle(frame, (int(bx * PX_PER_M), int(by * PX_PER_M) - 4), 4, (255, 255, 255), -1)
    return frame


def write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for i in range(int(DURATION_S * FPS)):
        writer.write(render_frame(i / FPS))
    writer.release()


def write_calibration(path: Path) -> None:
    names = ["corner_top_left", "corner_top_right", "corner_bottom_right", "corner_bottom_left", "center_spot"]
    pts = [[0, 0], [1050, 0], [1050, 680], [0, 680], [525, 340]]
    path.write_text(json.dumps({"frame_size": [WIDTH, HEIGHT], "image_points": pts, "landmarks": names}))


class FakePlayerDetector:
    """Drop-in for PlayerDetector: returns ground-truth boxes (shuffled order)."""

    def __init__(self, det_cfg, video_cfg, device):
        self.rng = np.random.default_rng(0)

    def detect_batch(self, frames, indices, timestamps):
        out = []
        for idx, ts in zip(indices, timestamps):
            dets = []
            for pid, (team, fn) in players().items():
                x, y = fn(ts)
                gx, gy = x * PX_PER_M, y * PX_PER_M
                dets.append(Detection(idx, ts, (gx - PLAYER_W / 2, gy - PLAYER_H, gx + PLAYER_W / 2, gy), 0.9, 0, "person"))
            self.rng.shuffle(dets)
            out.append(dets)
        return out


class FakeBallDetector:
    def __init__(self, det_cfg, device):
        pass

    def detect_batch(self, frames, indices, timestamps):
        out = []
        for idx, ts in zip(indices, timestamps):
            bx, by = ball_position(ts)
            cx, cy = bx * PX_PER_M, by * PX_PER_M - 4
            out.append([BallDetection(idx, ts, (cx - 4, cy - 4, cx + 4, cy + 4), 0.8)])
        return out
