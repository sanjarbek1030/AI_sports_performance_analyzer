"""Team shape metrics (MEASURED) and their interpretation (INFERRED).

Per sampled frame and team: centroid, width (lateral extent), depth
(longitudinal extent), mean distance to centroid ("spread"), mean pairwise
distance and convex-hull area. Values are only produced when at least
``min_players`` of the team are visible AND the camera must show the whole
team for them to be meaningful - the report states the coverage so a broadcast
close-up is not mistaken for a tactical view.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

from ..config import Config
from ..models import FrameData
from .possession import PossessionFrameState

SHAPE_COLUMNS = ["frame", "timestamp", "team_id", "n_players", "centroid_x", "centroid_y", "width", "depth",
                 "spread", "mean_pairwise_distance", "hull_area", "phase"]


def compute_team_shape(frames: list[FrameData], states: list[PossessionFrameState], cfg: Config,
                       effective_fps: float) -> pd.DataFrame:
    step = max(1, int(round(cfg.tactical.sample_interval_s * effective_fps)))
    rows = []
    for i in range(0, len(frames), step):
        fd = frames[i]
        holder_team = states[i].holder_team if i < len(states) else None
        for team_id in (0, 1):
            pts = np.array([o.smoothed_xy for o in fd.players
                            if o.team_id == team_id and o.on_pitch and o.smoothed_xy is not None])
            if len(pts) < cfg.tactical.min_players:
                continue
            centroid = pts.mean(axis=0)
            try:
                hull = float(ConvexHull(pts).volume)
            except QhullError:
                hull = float("nan")
            phase = "neutral" if holder_team is None else ("in_possession" if holder_team == team_id else "out_of_possession")
            rows.append({
                "frame": fd.frame, "timestamp": fd.timestamp, "team_id": team_id, "n_players": len(pts),
                "centroid_x": centroid[0], "centroid_y": centroid[1],
                "width": float(np.ptp(pts[:, 1])), "depth": float(np.ptp(pts[:, 0])),
                "spread": float(np.linalg.norm(pts - centroid, axis=1).mean()),
                "mean_pairwise_distance": float(pdist(pts).mean()), "hull_area": hull, "phase": phase,
            })
    return pd.DataFrame(rows, columns=SHAPE_COLUMNS)


def summarize_tactics(shape: pd.DataFrame, cfg: Config, n_sampled_frames: int) -> dict:
    """Aggregate measured metrics, then add clearly-labelled inferred descriptors."""
    pitch_area = cfg.field.field_length * cfg.field.field_width
    out: dict = {"measured": {}, "inferred": {}, "notes": [
        "Measured values describe only the players visible in the video; they equal true team shape only "
        "when the camera covers the whole team.",
        "'inferred' labels are heuristics relative to pitch size, not tactical ground truth."]}
    for team_id in (0, 1):
        sub = shape[shape["team_id"] == team_id]
        name = cfg.visualization.team_names[team_id]
        if sub.empty:
            out["measured"][name] = None
            out["inferred"][name] = None
            continue
        coverage = len(sub) / max(n_sampled_frames, 1)
        measured = {"samples": int(len(sub)), "coverage_of_sampled_frames": round(coverage, 3)}
        inferred = {}
        for phase in ("all", "in_possession", "out_of_possession"):
            part = sub if phase == "all" else sub[sub["phase"] == phase]
            if part.empty:
                continue
            m = {k: round(float(part[k].mean()), 2) for k in
                 ("centroid_x", "centroid_y", "width", "depth", "spread", "mean_pairwise_distance", "hull_area")}
            measured[phase] = m
            inferred[phase] = {
                "width": _label(m["width"] / cfg.field.field_width, cfg.tactical.narrow_ratio, cfg.tactical.wide_ratio, "narrow", "wide"),
                "depth": _label(m["depth"] / cfg.field.field_length, cfg.tactical.shallow_ratio, cfg.tactical.deep_ratio, "compact", "stretched"),
                "compactness": _label(m["hull_area"] / pitch_area, cfg.tactical.compact_hull_ratio, cfg.tactical.stretched_hull_ratio, "compact", "stretched"),
                "is_inference": True,
            }
        out["measured"][name], out["inferred"][name] = measured, inferred
    return out


def _label(value: float, low: float, high: float, low_name: str, high_name: str) -> Optional[str]:
    if not np.isfinite(value):
        return None
    return low_name if value <= low else (high_name if value >= high else "medium")
