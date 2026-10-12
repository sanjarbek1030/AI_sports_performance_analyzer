"""Generic event engine.

Problem solved : keep sport-specific rules out of the pipeline. An
                 ``EventDetector`` looks at the shared ``EventContext``
                 (tracks, possession spells, ball kinematics ...) and returns
                 ``Event`` objects. ``EventEngine`` runs detectors in order;
                 later detectors may read events produced by earlier ones
                 (e.g. goals attach to shots, assists to passes).
A failing detector is logged and skipped - it never aborts the analysis.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..analytics.ball_kinematics import BallKinematics
from ..config import Config
from ..field.football_field import FootballField
from ..models import Event, FrameData, PlayerObservation, PlayerTrack, PossessionSpell

logger = logging.getLogger(__name__)


@dataclass
class EventContext:
    config: Config
    field: FootballField
    effective_fps: float
    frames: list[FrameData]
    tracks: dict[int, PlayerTrack]
    spells: list[PossessionSpell]
    ball: BallKinematics
    attack_direction: Callable[[int, float], Optional[int]]   # (team_id, t) -> +1 / -1 / None
    events: list[Event] = field(default_factory=list)          # filled progressively
    _obs_index: dict[tuple[int, int], PlayerObservation] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for frame in self.frames:
            for obs in frame.players:
                self._obs_index[(obs.player_id, obs.frame)] = obs

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.effective_fps

    def player_xy(self, player_id: int, frame: int):
        obs = self._obs_index.get((player_id, frame))
        return obs.smoothed_xy if obs is not None else None

    def team_confidence(self, player_id: Optional[int]) -> float:
        track = self.tracks.get(player_id) if player_id is not None else None
        return track.team_confidence if track is not None else 0.0

    def events_of(self, event_type: str) -> list[Event]:
        return [e for e in self.events if e.event_type == event_type]


class EventDetector(ABC):
    name: str = "detector"

    @abstractmethod
    def detect(self, ctx: EventContext) -> list[Event]:
        """Return the events found in ``ctx``."""


class EventEngine:
    def __init__(self, detectors: list[EventDetector]) -> None:
        self.detectors = list(detectors)

    def register(self, detector: EventDetector) -> None:
        self.detectors.append(detector)

    def run(self, ctx: EventContext) -> list[Event]:
        for detector in self.detectors:
            try:
                found = detector.detect(ctx)
            except Exception:  # keep the pipeline alive; the failure is logged with traceback
                logger.exception("Event detector '%s' failed and was skipped.", detector.name)
                continue
            ctx.events.extend(found)
            logger.info("Detector %-22s -> %d event(s)", detector.name, len(found))
        ctx.events.sort(key=lambda e: (e.timestamp, e.event_type))
        return ctx.events
