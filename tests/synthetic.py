"""Synthetic match builder used by the tests (no video, no models).

Players and the ball follow functions of time in FIELD metres; the builder
creates the same ``FrameData`` objects the perception pass would produce.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from sports_analyzer.analytics.ball_kinematics import build_ball_kinematics
from sports_analyzer.analytics.possession import run_possession
from sports_analyzer.analytics.tracks import group_observations_into_tracks
from sports_analyzer.config import Config
from sports_analyzer.events.base import EventContext, EventEngine
from sports_analyzer.models import BallState, FrameData, PlayerObservation
from sports_analyzer.sports import get_sport_analyzer

FPS = 25.0


def build_frames(players: dict[int, tuple[int, Callable[[float], tuple[float, float]]]],
                 ball_fn: Callable[[float], Optional[tuple[float, float]]], duration: float,
                 fps: float = FPS) -> list[FrameData]:
    frames = []
    for i in range(int(duration * fps)):
        t = i / fps
        fd = FrameData(frame=i, timestamp=t)
        for pid, (team, fn) in players.items():
            xy = fn(t)
            fd.players.append(PlayerObservation(
                player_id=pid, tracker_id=pid, frame=i, timestamp=t, bbox=(0, 0, 10, 20), confidence=0.9,
                role="player", ground_image=(0, 0), team_id=team, field_xy=xy, on_pitch=True, smoothed_xy=xy))
        b = ball_fn(t)
        if b is not None:
            fd.ball = BallState(frame=i, timestamp=t, center=(0, 0), bbox=(0, 0, 4, 4), confidence=0.8,
                                predicted=False, field_xy=b)
        frames.append(fd)
    return frames


def analyse(frames, cfg: Optional[Config] = None, fps: float = FPS):
    cfg = cfg or Config()
    sport = get_sport_analyzer(cfg)
    tracks = group_observations_into_tracks(frames)
    for track in tracks.values():
        track.team_id = track.observations[0].team_id
        track.team_confidence = 1.0
    ball = build_ball_kinematics(frames, cfg.ball, fps)
    possession = run_possession(frames, ball, cfg.possession, fps)
    ctx = EventContext(cfg, sport.create_field(), fps, frames, tracks, possession.spells, ball, sport.attack_direction)
    events = EventEngine(sport.create_event_detectors()).run(ctx)
    return ctx, possession, events


def stand(x: float, y: float):
    return lambda t: (x, y)


def piecewise_ball(segments):
    """segments: list of (t_start, t_end, fn(t)->(x,y)|None); outside segments -> None."""
    def ball(t):
        for t0, t1, fn in segments:
            if t0 <= t < t1:
                return fn(t)
        return None
    return ball


def linear(p0, p1, t0, t1):
    return lambda t: (p0[0] + (p1[0] - p0[0]) * (t - t0) / (t1 - t0), p0[1] + (p1[1] - p0[1]) * (t - t0) / (t1 - t0))


def at(p):
    return lambda t: p
