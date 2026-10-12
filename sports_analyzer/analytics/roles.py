"""RoleEstimator (optional, INFERENCE ONLY): GK/CB/LB/RB/DM/CM/AM/LW/RW/ST.

From a player's *average* position, normalised to the team's attacking
direction: depth 0 = own goal line, 1 = opponent's; lateral 0 = left touchline
from the player's point of view. Pure positional heuristics - a fullback who
plays high is reported as a winger. The confidence reflects how far the
average lies from a zone boundary and how long the player was observed.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..config import Config
from ..models import PlayerTrack, TEAM_UNKNOWN

_ZONES = {"gk": 0.12, "defence": 0.33, "dm": 0.45, "cm": 0.55, "am": 0.65}
_LEFT, _RIGHT = 0.33, 0.67
_BOUNDARY_SCALE = 0.1


def role_from_position(depth: float, lateral: float) -> tuple[str, float]:
    """Return (role, margin to nearest zone boundary in 0..1 depth units)."""
    left, right = lateral < _LEFT, lateral > _RIGHT
    if depth < _ZONES["gk"]:
        return "GK", _ZONES["gk"] - depth
    if depth < _ZONES["defence"]:
        return ("LB" if left else "RB" if right else "CB"), min(depth - _ZONES["gk"], _ZONES["defence"] - depth)
    if depth < _ZONES["dm"]:
        return ("LW" if left else "RW" if right else "DM"), min(depth - _ZONES["defence"], _ZONES["dm"] - depth)
    if depth < _ZONES["cm"]:
        return ("LW" if left else "RW" if right else "CM"), min(depth - _ZONES["dm"], _ZONES["cm"] - depth)
    if depth < _ZONES["am"]:
        return ("LW" if left else "RW" if right else "AM"), min(depth - _ZONES["cm"], _ZONES["am"] - depth)
    return ("LW" if left else "RW" if right else "ST"), depth - _ZONES["am"]


def estimate_roles(tracks: dict[int, PlayerTrack], cfg: Config,
                   attack_direction: Callable[[int, float], Optional[int]]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    L, W = cfg.field.field_length, cfg.field.field_width
    for track in tracks.values():
        if track.team_id == TEAM_UNKNOWN or track.team_id < 0 or track.duration_s < cfg.roles.min_track_seconds:
            continue
        pts, dirs = [], []
        for o in track.observations:
            d = attack_direction(track.team_id, o.timestamp)
            if o.smoothed_xy is not None and o.on_pitch and d is not None:
                pts.append(o.smoothed_xy)
                dirs.append(d)
        if len(pts) < 10:
            continue
        pts_arr, dirs_arr = np.array(pts), np.array(dirs)
        depth = float(np.mean(np.where(dirs_arr > 0, pts_arr[:, 0], L - pts_arr[:, 0]) / L))
        lateral = float(np.mean(np.where(dirs_arr > 0, pts_arr[:, 1], W - pts_arr[:, 1]) / W))
        role, margin = role_from_position(depth, lateral)
        duration_factor = min(1.0, track.duration_s / (4 * cfg.roles.min_track_seconds))
        confidence = float(np.clip(margin / _BOUNDARY_SCALE, 0, 1) * 0.7 * duration_factor + 0.3 * duration_factor * 0.5)
        out[track.player_id] = {"role": role, "confidence": round(min(confidence, 0.8), 3),
                                "mean_depth_fraction": round(depth, 3), "mean_lateral_fraction": round(lateral, 3),
                                "is_inference": True}
    return out
