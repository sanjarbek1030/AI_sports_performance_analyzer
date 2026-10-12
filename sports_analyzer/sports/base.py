"""Sport abstraction: everything sport-specific lives behind this interface.

The detection / tracking / movement layers are sport-agnostic. A new sport
(basketball, tennis ...) implements ``SportsAnalyzer`` with its own field
model, attack-direction rules and event detectors, and registers itself in
``sports/__init__.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..config import Config
from ..events.base import EventDetector
from ..field.football_field import FieldModel


class SportsAnalyzer(ABC):
    name: str = "generic"

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def create_field(self) -> FieldModel:
        """Field/court geometry used for calibration, mini-map and heatmaps."""

    @abstractmethod
    def create_event_detectors(self) -> list[EventDetector]:
        """Ordered detectors (later ones may use events of earlier ones)."""

    @abstractmethod
    def attack_direction(self, team_id: int, timestamp: float) -> Optional[int]:
        """+1 / -1 = direction of increasing/decreasing x that the team attacks."""
