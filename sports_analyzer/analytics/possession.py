"""PossessionAnalyzer: who controls the ball, with temporal confirmation.

Problem solved : "nearest player to the ball" flickers every frame and is
                 wrong while the ball is passed between players or in flight.
State machine  : FREE -> POSSESSION_CANDIDATE -> POSSESSION -> RELEASE -> FREE
  * FREE        : no one in control.
  * CANDIDATE   : a player is within ``possession_radius`` of the ball; needs
                  ``confirm_s`` of consecutive frames to become POSSESSION
                  (a single frame never counts).
  * POSSESSION  : holder keeps possession while within the (larger) release
                  radius; short ball/player dropouts are tolerated; a rival who
                  is clearly closer for ``confirm_s`` takes over ("challenged").
  * RELEASE     : one-frame transition state in which the finished
                  ``PossessionSpell`` is emitted; the next frame is FREE.
Receives       : per-frame ball field position + player field positions.
Produces       : possession spells (player, team, start, end, duration, confidence)
                 and a per-frame state record (for overlays / debugging).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ..config import PossessionConfig
from ..models import FrameData, PossessionSpell, Point, TEAM_REFEREE, TEAM_UNKNOWN
from .ball_kinematics import BallKinematics

logger = logging.getLogger(__name__)


class PossessionState(str, Enum):
    FREE = "FREE"
    CANDIDATE = "POSSESSION_CANDIDATE"
    POSSESSION = "POSSESSION"
    RELEASE = "RELEASE"


@dataclass(slots=True)
class PossessionFrameState:
    frame: int
    timestamp: float
    state: PossessionState
    holder_id: Optional[int]
    holder_team: Optional[int]
    nearest_id: Optional[int]
    nearest_distance: Optional[float]


@dataclass
class PlayerPoint:
    player_id: int
    team_id: int
    xy: Point


@dataclass
class _SpellAccumulator:
    player_id: int
    team_id: int
    start_frame: int
    start_time: float
    start_ball: Optional[Point]
    frames: int = 0
    good_frames: int = 0
    norm_dist_sum: float = 0.0
    last_in_frame: int = 0
    last_in_time: float = 0.0
    last_in_ball: Optional[Point] = None


class PossessionAnalyzer:
    def __init__(self, cfg: PossessionConfig, effective_fps: float) -> None:
        self.cfg = cfg
        self.dt = 1.0 / effective_fps
        self.confirm_frames = max(2, int(round(cfg.confirm_s * effective_fps)))
        self.release_frames = max(1, int(round(cfg.release_s * effective_fps)))
        self.max_missing_frames = max(1, int(round(cfg.max_missing_s * effective_fps)))
        self.state = PossessionState.FREE
        self.spells: list[PossessionSpell] = []
        self._candidate: Optional[int] = None
        self._cand_count = 0
        self._cand_start: tuple[int, float, Optional[Point]] = (0, 0.0, None)
        self._cand_team = TEAM_UNKNOWN
        self._acc: Optional[_SpellAccumulator] = None
        self._missing = 0
        self._out_count = 0
        self._challenger: Optional[int] = None
        self._challenger_count = 0
        self._carry: Optional[tuple[int, int, tuple[int, float, Optional[Point]]]] = None

    # -------------------------------------------------------------- step
    def step(self, frame: int, t: float, ball_xy: Optional[Point], players: list[PlayerPoint]) -> PossessionFrameState:
        """Advance the state machine by one analysed frame."""
        players = [p for p in players if p.team_id != TEAM_REFEREE]
        nearest, dist = self._nearest(ball_xy, players)
        near = nearest is not None and dist <= self.cfg.possession_radius

        if self.state == PossessionState.RELEASE:
            self.state = PossessionState.FREE
            if self._carry is not None:  # a challenger had already been confirmed
                self._candidate, self._cand_count, self._cand_start = self._carry
                self._cand_team = next((p.team_id for p in players if p.player_id == self._candidate), TEAM_UNKNOWN)
                self._carry = None
                self.state = PossessionState.CANDIDATE

        if self.state == PossessionState.FREE:
            if near:
                self._begin_candidate(nearest, frame, t, ball_xy, players)
        elif self.state == PossessionState.CANDIDATE:
            self._step_candidate(nearest, near, frame, t, ball_xy, players)
        elif self.state == PossessionState.POSSESSION:
            self._step_possession(nearest, dist, frame, t, ball_xy, players)

        holder = self._acc.player_id if (self._acc and self.state == PossessionState.POSSESSION) else None
        holder_team = self._acc.team_id if holder is not None else None
        return PossessionFrameState(frame, t, self.state, holder, holder_team, nearest, dist if nearest is not None else None)

    # ------------------------------------------------------------ states
    def _begin_candidate(self, pid, frame, t, ball_xy, players) -> None:
        self.state = PossessionState.CANDIDATE
        self._candidate, self._cand_count = pid, 1
        self._cand_start = (frame, t, ball_xy)
        self._cand_team = next(p.team_id for p in players if p.player_id == pid)
        self._missing = 0

    def _step_candidate(self, nearest, near, frame, t, ball_xy, players) -> None:
        if ball_xy is None:
            self._missing += 1
            if self._missing > self.max_missing_frames:
                self.state = PossessionState.FREE
            return
        self._missing = 0
        if near and nearest == self._candidate:
            self._cand_count += 1
            if self._cand_count >= self.confirm_frames:
                self._start_possession(frame, t, ball_xy)
        elif near:
            self._begin_candidate(nearest, frame, t, ball_xy, players)
        else:
            self.state = PossessionState.FREE

    def _start_possession(self, frame, t, ball_xy) -> None:
        start_frame, start_time, start_ball = self._cand_start
        self._acc = _SpellAccumulator(self._candidate, self._cand_team, start_frame, start_time, start_ball,
                                      frames=self._cand_count, good_frames=self._cand_count,
                                      norm_dist_sum=0.5 * self._cand_count,
                                      last_in_frame=frame, last_in_time=t, last_in_ball=ball_xy)
        self.state = PossessionState.POSSESSION
        self._out_count = self._missing = self._challenger_count = 0
        self._challenger = None

    def _step_possession(self, nearest, nearest_dist, frame, t, ball_xy, players) -> None:
        acc = self._acc
        holder = next((p for p in players if p.player_id == acc.player_id), None)
        if ball_xy is None or holder is None:
            self._missing += 1
            if self._missing > self.max_missing_frames:
                self._release("ball_or_player_lost")
            return
        self._missing = 0
        d_holder = math.dist(ball_xy, holder.xy)
        release_radius = self.cfg.possession_radius * self.cfg.release_radius_factor
        acc.frames += 1
        if d_holder <= release_radius:
            self._out_count = 0
            acc.good_frames += 1
            acc.norm_dist_sum += min(1.0, d_holder / release_radius)
            acc.last_in_frame, acc.last_in_time, acc.last_in_ball = frame, t, ball_xy
            self._check_challenger(nearest, nearest_dist, d_holder, frame, t, ball_xy)
        else:
            self._out_count += 1
            acc.norm_dist_sum += 1.0
            if self._out_count >= self.release_frames:
                self._release("ball_left")

    def _check_challenger(self, nearest, nearest_dist, d_holder, frame, t, ball_xy) -> None:
        acc = self._acc
        rival = (nearest is not None and nearest != acc.player_id
                 and nearest_dist <= self.cfg.possession_radius
                 and nearest_dist + self.cfg.challenger_margin_m < d_holder)
        if rival and nearest == self._challenger:
            self._challenger_count += 1
        elif rival:
            self._challenger, self._challenger_count = nearest, 1
            self._challenger_start = (frame, t, ball_xy)
        else:
            self._challenger, self._challenger_count = None, 0
        if self._challenger is not None and self._challenger_count >= self.confirm_frames:
            self._carry = (self._challenger, self._challenger_count, self._challenger_start)
            self._release("challenged")

    def _release(self, reason: str) -> None:
        acc = self._acc
        end_frame, end_time, end_ball = acc.last_in_frame, acc.last_in_time, acc.last_in_ball
        frames = max(acc.frames, 1)
        quality = 1.0 - 0.5 * (acc.norm_dist_sum / frames)
        confidence = float(np.clip((acc.good_frames / frames) * quality, 0.0, 1.0))
        self.spells.append(PossessionSpell(
            player_id=acc.player_id, team_id=acc.team_id, start_frame=acc.start_frame, end_frame=end_frame,
            start_time=acc.start_time, end_time=end_time,
            duration=max(end_time - acc.start_time, 0.0) + self.dt, confidence=confidence,
            start_ball_xy=acc.start_ball, end_ball_xy=end_ball, end_reason=reason,
        ))
        self._acc, self.state = None, PossessionState.RELEASE
        self._out_count = self._missing = self._challenger_count = 0
        self._challenger = None

    def finalize(self) -> list[PossessionSpell]:
        if self._acc is not None:
            self._release("end_of_video")
        return self.spells

    @staticmethod
    def _nearest(ball_xy, players) -> tuple[Optional[int], float]:
        if ball_xy is None or not players:
            return None, math.inf
        dists = [math.dist(ball_xy, p.xy) for p in players]
        i = int(np.argmin(dists))
        return players[i].player_id, dists[i]


@dataclass
class PossessionResult:
    spells: list[PossessionSpell]
    frame_states: list[PossessionFrameState]

    def team_time(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for s in self.spells:
            out[s.team_id] = out.get(s.team_id, 0.0) + s.duration
        return out


def run_possession(frames: list[FrameData], ball: BallKinematics, cfg: PossessionConfig,
                   effective_fps: float) -> PossessionResult:
    """Run the state machine over the whole analysed timeline."""
    analyzer = PossessionAnalyzer(cfg, effective_fps)
    max_coast_frames = max(1, int(round(cfg.max_ball_coast_s * effective_fps)))
    states: list[PossessionFrameState] = []
    coast = 0
    for i, fd in enumerate(frames):
        ball_xy: Optional[Point] = None
        if np.isfinite(ball.xy[i]).all():
            coast = 0 if ball.observed[i] else coast + 1
            if ball.observed[i] or coast <= max_coast_frames:
                ball_xy = (float(ball.xy[i, 0]), float(ball.xy[i, 1]))
        players = [PlayerPoint(o.player_id, o.team_id, o.smoothed_xy) for o in fd.players
                   if o.smoothed_xy is not None and o.on_pitch]
        states.append(analyzer.step(fd.frame, fd.timestamp, ball_xy, players))
    spells = analyzer.finalize()
    return PossessionResult(spells, states)
