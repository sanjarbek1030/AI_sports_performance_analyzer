"""Shared data structures passed between pipeline components.

Coordinate conventions
----------------------
* Image coordinates: pixels, origin top-left, y grows downward.
* Field coordinates: metres. x runs along the pitch length (0..L), y along the
  width (0..W). y grows toward the camera-side touchline, i.e. a top-down map
  drawn with y downward looks like the main camera view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2
Point = tuple[float, float]

TEAM_UNKNOWN = -1
TEAM_REFEREE = -2


def team_label(team_id: Optional[int], names: Optional[list[str]] = None) -> str:
    names = names or ["Team A", "Team B"]
    if team_id is None:
        return ""
    if team_id == TEAM_REFEREE:
        return "Referee"
    if team_id == TEAM_UNKNOWN or team_id >= len(names):
        return "Unknown"
    return names[team_id]


class DetectionRole:
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"


class EventType:
    PASS = "pass"
    SHOT = "shot"
    GOAL = "goal"
    POSSESSION_CHANGE = "possession_change"
    SPRINT = "sprint"
    TACKLE_CANDIDATE = "tackle_candidate"
    INTERCEPTION_CANDIDATE = "interception_candidate"
    OUT_OF_BOUNDS = "out_of_bounds"


# --------------------------------------------------------------------------
# Detections
# --------------------------------------------------------------------------
@dataclass(slots=True)
class Detection:
    """A person detection (optionally carrying a tracker id)."""

    frame: int
    timestamp: float
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    role: str = DetectionRole.PLAYER
    track_id: Optional[int] = None

    @property
    def ground_point(self) -> Point:
        """Bottom-centre of the box: where the feet touch the pitch."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass(slots=True)
class BallDetection:
    frame: int
    timestamp: float
    bbox: BBox
    confidence: float

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def ground_point(self) -> Point:
        """Bottom-centre: the ball's contact point when it is on the grass."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def size(self) -> float:
        return max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


@dataclass(slots=True)
class BallState:
    """Output of the BallTracker for one frame."""

    frame: int
    timestamp: float
    center: Point
    bbox: BBox
    confidence: float
    predicted: bool                      # True = motion prediction, no detection
    velocity_px: Point = (0.0, 0.0)      # pixels / second
    direction_deg: float = 0.0           # image-space heading
    field_xy: Optional[Point] = None
    raw_detection: Optional[BallDetection] = None

    @property
    def ground_point(self) -> Point:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


# --------------------------------------------------------------------------
# Per-frame observations
# --------------------------------------------------------------------------
@dataclass(slots=True)
class PlayerObservation:
    """One tracked player in one analysed frame."""

    player_id: int
    tracker_id: int
    frame: int
    timestamp: float
    bbox: BBox
    confidence: float
    role: str
    ground_image: Point
    jersey_feature: Optional[np.ndarray] = None
    team_id: int = TEAM_UNKNOWN
    field_xy: Optional[Point] = None      # raw homography output
    on_pitch: bool = False
    smoothed_xy: Optional[Point] = None   # filled by MovementAnalyzer
    speed_kmh: Optional[float] = None
    accel_ms2: Optional[float] = None
    is_outlier: bool = False


@dataclass(slots=True)
class FrameData:
    frame: int
    timestamp: float
    players: list[PlayerObservation] = field(default_factory=list)
    ball: Optional[BallState] = None


# --------------------------------------------------------------------------
# Derived records
# --------------------------------------------------------------------------
@dataclass(slots=True)
class SprintRecord:
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration_s: float
    max_speed_kmh: float
    distance_m: float
    observed_fraction: float


@dataclass(slots=True)
class MovementEvent:
    """Per-player kinematic event (acceleration, deceleration, stop, turn)."""

    kind: str
    start_time: float
    end_time: float
    value: float


@dataclass(slots=True)
class PossessionSpell:
    player_id: int
    team_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    confidence: float
    start_ball_xy: Optional[Point] = None
    end_ball_xy: Optional[Point] = None
    end_reason: str = ""


@dataclass
class Event:
    event_type: str
    timestamp: float
    frame: int
    player_id: Optional[int]
    team_id: Optional[int]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    inferred: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event_type,
            "timestamp": round(self.timestamp, 3),
            "frame": self.frame,
            "player_id": self.player_id,
            "team_id": self.team_id,
            "confidence": round(self.confidence, 3),
            "inferred": self.inferred,
            "metadata": self.metadata,
        }


@dataclass
class PlayerTrack:
    """Everything known about one tracked identity (extensible via ``extra_metrics``)."""

    player_id: int
    team_id: int = TEAM_UNKNOWN
    team_confidence: float = 0.0
    role: str = DetectionRole.PLAYER
    observations: list[PlayerObservation] = field(default_factory=list)
    speeds: np.ndarray = field(default_factory=lambda: np.empty(0))         # km/h per observation
    accelerations: np.ndarray = field(default_factory=lambda: np.empty(0))  # m/s^2 per observation
    distances: np.ndarray = field(default_factory=lambda: np.empty(0))      # metres since previous observation
    possession_time: float = 0.0
    passes: int = 0
    successful_passes: int = 0
    shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    assists: int = 0
    interceptions: int = 0
    sprint_count: int = 0
    maximum_speed: Optional[float] = None   # km/h
    average_speed: Optional[float] = None   # km/h
    total_distance_m: Optional[float] = None
    movement_available: bool = False
    sprints: list[SprintRecord] = field(default_factory=list)
    movement_events: list[MovementEvent] = field(default_factory=list)
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([o.timestamp for o in self.observations])

    @property
    def bounding_boxes(self) -> list[BBox]:
        return [o.bbox for o in self.observations]

    @property
    def positions(self) -> np.ndarray:
        """Raw field positions (NaN where unavailable)."""
        return np.array(
            [o.field_xy if o.field_xy is not None else (np.nan, np.nan) for o in self.observations],
            dtype=float,
        ).reshape(-1, 2)

    @property
    def first_time(self) -> float:
        return self.observations[0].timestamp

    @property
    def last_time(self) -> float:
        return self.observations[-1].timestamp

    @property
    def duration_s(self) -> float:
        return self.last_time - self.first_time if self.observations else 0.0

    @property
    def mean_confidence(self) -> float:
        return float(np.mean([o.confidence for o in self.observations])) if self.observations else 0.0
