"""Football (soccer) analyzer: field, attack directions, event detectors."""
from __future__ import annotations

from typing import Optional

from ..events.base import EventDetector
from ..events.football_events import (GoalDetector, OutOfBoundsDetector, PassDetector, PossessionChangeDetector,
                                      ShotDetector, SprintEventDetector, TackleInterceptionDetector)
from ..field.football_field import FootballField
from .base import SportsAnalyzer


class FootballAnalyzer(SportsAnalyzer):
    name = "football"

    def create_field(self) -> FootballField:
        f = self.config.field
        return FootballField(f.field_length, f.field_width, f.goal_width)

    def create_event_detectors(self) -> list[EventDetector]:
        # Order matters: shots -> passes (exclude shot spells) -> goals (link shot/assist) -> the rest.
        return [PossessionChangeDetector(), ShotDetector(), PassDetector(), GoalDetector(),
                TackleInterceptionDetector(), OutOfBoundsDetector(), SprintEventDetector()]

    def attack_direction(self, team_id: int, timestamp: float) -> Optional[int]:
        if team_id not in (0, 1):
            return None
        direction = 1 if self.config.field.team_a_attacks_positive_x else -1
        switch = self.config.field.attack_direction_switch_s
        if switch is not None and timestamp >= switch:
            direction = -direction
        return direction if team_id == 0 else -direction
