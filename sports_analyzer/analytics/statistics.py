"""PlayerStatistics and TeamStatistics: aggregation + honest data gating.

Rule: never fabricate. A statistic that the data cannot support is ``None``
(empty CSV cell / JSON null) rather than 0:
  * no calibration                      -> all metric statistics are None
  * ball detected in < min coverage     -> possession/pass/shot/... are None
  * track shorter than min duration     -> identity is excluded from reports
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from ..config import Config
from ..models import Event, EventType, PlayerTrack, PossessionSpell, TEAM_REFEREE, team_label
from ..utils import safe_div

PLAYER_COLUMNS = [
    "player_id", "team", "playing_time", "distance_km", "average_speed_kmh", "max_speed_kmh", "sprints",
    "possession_seconds", "passes", "successful_passes", "shots", "shots_on_target", "goals", "assists",
    "interceptions",
]
TEAM_COLUMNS = [
    "team", "possession_percentage", "total_distance_km", "average_distance_km", "passes", "successful_passes",
    "shots", "shots_on_target", "goals", "sprints",
]
_ON_TARGET = {"on_target", "saved_candidate", "goal"}


@dataclass
class DataQuality:
    """What the input data can support (drives which statistics are withheld)."""

    calibrated: bool
    ball_coverage: float
    min_ball_coverage: float
    notes: list[str] = field(default_factory=list)

    @property
    def movement_reliable(self) -> bool:
        return self.calibrated

    @property
    def ball_events_reliable(self) -> bool:
        return self.calibrated and self.ball_coverage >= self.min_ball_coverage

    def to_dict(self) -> dict[str, Any]:
        return {"calibrated": self.calibrated, "ball_detection_coverage": round(self.ball_coverage, 3),
                "movement_statistics_available": self.movement_reliable,
                "ball_event_statistics_available": self.ball_events_reliable, "notes": self.notes}


def report_tracks(tracks: dict[int, PlayerTrack], min_seconds: float) -> list[PlayerTrack]:
    """Identities that go into reports: not referees, observed long enough."""
    return sorted((t for t in tracks.values() if t.team_id != TEAM_REFEREE and t.duration_s >= min_seconds),
                  key=lambda t: t.player_id)


class PlayerStatistics:
    def __init__(self, config: Config, quality: DataQuality) -> None:
        self.cfg, self.quality = config, quality

    def apply_events(self, tracks: dict[int, PlayerTrack], spells: list[PossessionSpell], events: list[Event]) -> None:
        """Copy event/possession results onto ``PlayerTrack`` counters."""
        for spell in spells:
            if spell.player_id in tracks:
                tracks[spell.player_id].possession_time += spell.duration
        for e in events:
            track = tracks.get(e.player_id) if e.player_id is not None else None
            if e.event_type == EventType.PASS and track:
                track.passes += 1
                track.successful_passes += int(bool(e.metadata.get("successful")))
            elif e.event_type == EventType.SHOT and track:
                track.shots += 1
                track.shots_on_target += int(e.metadata.get("outcome") in _ON_TARGET)
            elif e.event_type == EventType.GOAL:
                if track:
                    track.goals += 1
                assist = tracks.get(e.metadata.get("assist_player_id"))
                if assist:
                    assist.assists += 1
            elif e.event_type == EventType.INTERCEPTION_CANDIDATE and track:
                track.interceptions += 1

    def to_dataframe(self, tracks: dict[int, PlayerTrack]) -> pd.DataFrame:
        names = self.cfg.visualization.team_names
        movement_ok, ball_ok = self.quality.movement_reliable, self.quality.ball_events_reliable
        rows = []
        for t in report_tracks(tracks, self.cfg.tracking.min_report_track_s):
            has_move = movement_ok and t.movement_available
            rows.append({
                "player_id": t.player_id, "team": team_label(t.team_id, names),
                "playing_time": round(float(t.extra_metrics.get("analysed_time_s", t.duration_s)), 2) if has_move else round(t.duration_s, 2),
                "distance_km": round(t.total_distance_m / 1000, 3) if has_move else None,
                "average_speed_kmh": round(t.average_speed, 2) if has_move and t.average_speed is not None else None,
                "max_speed_kmh": round(t.maximum_speed, 2) if has_move and t.maximum_speed is not None else None,
                "sprints": t.sprint_count if has_move else None,
                "possession_seconds": round(t.possession_time, 2) if ball_ok else None,
                "passes": t.passes if ball_ok else None,
                "successful_passes": t.successful_passes if ball_ok else None,
                "shots": t.shots if ball_ok else None,
                "shots_on_target": t.shots_on_target if ball_ok else None,
                "goals": t.goals if ball_ok else None,
                "assists": t.assists if ball_ok else None,
                "interceptions": t.interceptions if ball_ok else None,
            })
        return pd.DataFrame(rows, columns=PLAYER_COLUMNS, dtype=object)


class TeamStatistics:
    def __init__(self, config: Config, quality: DataQuality) -> None:
        self.cfg, self.quality = config, quality

    def compute(self, tracks: dict[int, PlayerTrack], spells: list[PossessionSpell], events: list[Event]) -> tuple[pd.DataFrame, dict[int, dict]]:
        names = self.cfg.visualization.team_names
        movement_ok, ball_ok = self.quality.movement_reliable, self.quality.ball_events_reliable
        reported = report_tracks(tracks, self.cfg.tracking.min_report_track_s)
        team_time = {0: 0.0, 1: 0.0}
        for s in spells:
            if s.team_id in team_time:
                team_time[s.team_id] += s.duration
        total_time = sum(team_time.values())
        changes = self._possession_changes(spells)
        rows, details = [], {}
        for team_id in (0, 1):
            members = [t for t in reported if t.team_id == team_id]
            moving = [t for t in members if t.movement_available]
            distance_m = sum(t.total_distance_m or 0.0 for t in moving)
            mine = [e for e in events if e.team_id == team_id]
            pct = safe_div(team_time[team_id], total_time)
            row = {
                "team": names[team_id],
                "possession_percentage": round(100 * pct, 1) if (ball_ok and pct is not None) else None,
                "total_distance_km": round(distance_m / 1000, 3) if movement_ok and moving else None,
                "average_distance_km": round(distance_m / len(moving) / 1000, 3) if movement_ok and moving else None,
                "passes": sum(e.event_type == EventType.PASS for e in mine) if ball_ok else None,
                "successful_passes": sum(e.event_type == EventType.PASS and e.metadata.get("successful") for e in mine) if ball_ok else None,
                "shots": sum(e.event_type == EventType.SHOT for e in mine) if ball_ok else None,
                "shots_on_target": sum(e.event_type == EventType.SHOT and e.metadata.get("outcome") in _ON_TARGET for e in mine) if ball_ok else None,
                "goals": sum(e.event_type == EventType.GOAL for e in mine) if ball_ok else None,
                "sprints": sum(t.sprint_count for t in moving) if movement_ok and moving else None,
            }
            rows.append(row)
            team_spells = [s for s in spells if s.team_id == team_id]
            details[team_id] = {
                "tracked_identities": len(members),
                "identities_with_movement_data": len(moving),
                "possession_seconds": round(team_time[team_id], 2) if ball_ok else None,
                "possession_spells": len(team_spells) if ball_ok else None,
                "average_possession_duration_s": round(team_time[team_id] / len(team_spells), 2) if ball_ok and team_spells else None,
                "possession_changes_gained": changes.get(team_id) if ball_ok else None,
                "mean_team_classification_confidence": round(sum(t.team_confidence for t in members) / len(members), 3) if members else None,
            }
        details["possession_total_s"] = round(total_time, 2)
        details["unattributed_possession_s"] = round(sum(s.duration for s in spells if s.team_id not in team_time), 2)
        return pd.DataFrame(rows, columns=TEAM_COLUMNS, dtype=object), details

    @staticmethod
    def _possession_changes(spells: list[PossessionSpell]) -> dict[int, int]:
        gained = {0: 0, 1: 0}
        for a, b in zip(spells, spells[1:]):
            if a.team_id in gained and b.team_id in gained and a.team_id != b.team_id:
                gained[b.team_id] += 1
        return gained


def validate_possession_percentages(df: pd.DataFrame, tolerance: float = 0.2) -> Optional[str]:
    """Return an error string if team possession does not sum to ~100 (rounding aside)."""
    vals = df["possession_percentage"].dropna()
    if len(vals) == 2 and abs(vals.sum() - 100.0) > tolerance:
        return f"Possession percentages sum to {vals.sum():.1f}, expected 100."
    return None
