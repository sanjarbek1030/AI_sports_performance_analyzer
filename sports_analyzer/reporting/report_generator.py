"""ReportGenerator: CSV and JSON outputs.

Conventions
-----------
* ``None`` -> empty CSV cell / JSON ``null``  ("unavailable", never faked).
* ``player_tracks.csv`` is streamed row by row (full matches have millions of rows).
* ``field_x/field_y`` are the raw homography output; ``field_x_smooth`` /
  ``field_y_smooth`` (extra columns) are the smoothed positions all analytics use.
* Event timestamps are written ``MM:SS.ff`` in the CSV and seconds in the JSON.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..analysis import AnalysisResult
from ..config import Config
from ..exceptions import SportsAnalyzerError
from ..models import Event, FrameData, PlayerTrack, TEAM_REFEREE, team_label
from ..utils import format_timestamp

logger = logging.getLogger(__name__)
REPORT_VERSION = "1.0"
TRACK_COLUMNS = ["frame", "timestamp", "player_id", "team", "image_x", "image_y", "field_x", "field_y",
                 "speed_kmh", "confidence", "field_x_smooth", "field_y_smooth"]
EVENT_COLUMNS = ["timestamp", "frame", "event_type", "player_id", "team_id", "confidence", "metadata"]

LIMITATIONS = [
    "All events (pass, shot, goal, possession, tackle/interception candidates, out of bounds) are inferred from "
    "tracking data and are not verified ground truth.",
    "Field coordinates assume a static camera and a ground-plane homography; an airborne ball is mapped to the wrong "
    "place and a panning camera invalidates a static calibration.",
    "Player identities are tracker-based (no jersey-number OCR / re-identification); long occlusions can split one "
    "player into several identities.",
    "Goals cannot be distinguished from shots over the crossbar without ball height.",
    "Formation and role estimates are positional heuristics, not tactical ground truth.",
]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / dataclass / enum / NaN values into strict-JSON-safe types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict(orient="records")]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


class ReportGenerator:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.names = config.visualization.team_names

    # ------------------------------------------------------------------ CSV
    def write_player_tracks(self, frames: list[FrameData], path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(TRACK_COLUMNS)
                for fd in frames:
                    for o in fd.players:
                        fx, fy = o.field_xy if o.field_xy is not None else ("", "")
                        sx, sy = o.smoothed_xy if o.smoothed_xy is not None else ("", "")
                        writer.writerow([
                            o.frame, f"{o.timestamp:.3f}", o.player_id, team_label(o.team_id, self.names),
                            f"{o.ground_image[0]:.1f}", f"{o.ground_image[1]:.1f}",
                            _fmt(fx, 2), _fmt(fy, 2), _fmt(o.speed_kmh, 2), f"{o.confidence:.3f}", _fmt(sx, 2), _fmt(sy, 2)])
                        rows += 1
        except OSError as exc:
            raise SportsAnalyzerError(f"Cannot write {path}: {exc}") from exc
        return rows

    @staticmethod
    def write_dataframe(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, na_rep="")

    def write_event_log(self, events: list[Event], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(EVENT_COLUMNS)
            for e in events:
                writer.writerow([
                    format_timestamp(e.timestamp), e.frame, e.event_type.upper(),
                    "" if e.player_id is None else e.player_id,
                    "" if e.team_id is None else team_label(e.team_id, self.names),
                    f"{e.confidence:.3f}", json.dumps(to_jsonable({**e.metadata, "inferred": e.inferred}), separators=(",", ":"))])

    # ----------------------------------------------------------------- JSON
    def build_report(self, r: AnalysisResult) -> dict[str, Any]:
        players = {}
        reported = set(r.player_stats["player_id"]) if len(r.player_stats) else set()
        stats_by_id = {int(row["player_id"]): row for row in r.player_stats.to_dict(orient="records")}
        for pid in sorted(reported):
            t = r.tracks[int(pid)]
            players[str(pid)] = {
                "team": team_label(t.team_id, self.names), "team_confidence": round(t.team_confidence, 3),
                "mean_detection_confidence": round(t.mean_confidence, 3),
                "first_seen_s": round(t.first_time, 2), "last_seen_s": round(t.last_time, 2),
                "statistics": stats_by_id[int(pid)],
                "movement": self._movement(t),
                "estimated_role": r.roles.get(int(pid)),
            }
        excluded = sorted(int(t.player_id) for t in r.tracks.values()
                          if t.team_id != TEAM_REFEREE and t.player_id not in reported)
        possession = {"total_possession_s": r.team_details.get("possession_total_s"),
                      "unattributed_possession_s": r.team_details.get("unattributed_possession_s"),
                      "spells": [dataclasses.asdict(s) for s in r.spells] if r.quality.ball_events_reliable else None}
        return to_jsonable({
            "report_version": REPORT_VERSION,
            "generated_at": datetime.now(timezone.utc),
            "video": r.video.to_dict(),
            "processing": {
                "analysis_stride": self.cfg.video.analysis_stride, "effective_analysis_fps": r.effective_fps,
                "analysed_frames": len(r.frames), "tracker": self.cfg.tracking.tracker, "calibrated": r.calibrated,
                "identity_relinks": r.identity_relinks, "ball_lost_events": r.ball_lost_events,
                "config": self.cfg.to_dict()},
            "data_quality": {**r.quality.to_dict(), "warnings": r.warnings},
            "match": {
                "duration_s": r.video.duration_s, "possession": possession,
                "event_counts": _count_events(r.events),
                "ball_detection_mean_confidence": _mean_conf(r),
                "note": "Event counts are inferred; see limitations."},
            "teams": {self.names[i]: {**r.team_stats.iloc[i].to_dict(), **r.team_details[i]} for i in (0, 1)},
            "players": players,
            "excluded_identities_too_short": excluded,
            "events": [e.to_dict() for e in r.events],
            "tactical_analysis": r.tactical,
            "formation_estimates": r.formations,
            "limitations": LIMITATIONS,
        })

    def write_match_report(self, report: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)

    @staticmethod
    def _movement(t: PlayerTrack) -> dict[str, Any] | None:
        if not t.movement_available:
            return None
        return {
            "total_distance_m": t.total_distance_m, "average_speed_kmh": t.average_speed,
            "max_speed_kmh": t.maximum_speed, "sprints": [dataclasses.asdict(s) for s in t.sprints],
            "movement_events": [dataclasses.asdict(m) for m in t.movement_events], **t.extra_metrics}


def _fmt(value, digits: int):
    return "" if value is None or value == "" else f"{value:.{digits}f}"


def _count_events(events: list[Event]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        counts[e.event_type] = counts.get(e.event_type, 0) + 1
    return counts


def _mean_conf(r: AnalysisResult):
    vals = r.ball.confidence[r.ball.observed]
    return round(float(vals.mean()), 3) if len(vals) else None
