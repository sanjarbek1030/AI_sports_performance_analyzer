"""FormationEstimator (optional, INFERENCE ONLY).

Idea: in each time window, average every player's position, express depth as
"metres from own goal line", sort the 10 outfield players by depth and cut
them into lines. Each template (4-4-2 ...) fixes the line sizes; the template
whose lines are the most internally compact (lowest within-line depth variance
+ a small per-line penalty) wins. Confidence = softmax margin between the best
and other templates.
Not assumed fixed: one estimate per window; the report lists every window.
Needs 10 outfield players visible (11 -> the deepest is treated as GK).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Optional

import numpy as np

from ..config import Config
from ..models import FrameData

TEMPLATES: dict[str, list[int]] = {
    "4-3-3": [4, 3, 3], "4-4-2": [4, 4, 2], "4-2-3-1": [4, 2, 3, 1], "3-5-2": [3, 5, 2], "5-3-2": [5, 3, 2],
}
_OUTFIELD_PLAYERS = 10


def classify_lines(depths: np.ndarray, line_penalty: float, temperature: float) -> Optional[dict]:
    """Return {'formation', 'confidence', 'costs'} for 10 sorted outfield depths (metres)."""
    if len(depths) != _OUTFIELD_PLAYERS:
        return None
    depths = np.sort(depths)
    costs = {}
    for name, sizes in TEMPLATES.items():
        cost, start = 0.0, 0
        for size in sizes:
            group = depths[start:start + size]
            cost += float(((group - group.mean()) ** 2).sum())
            start += size
        costs[name] = cost + line_penalty * len(sizes)
    best = min(costs, key=costs.get)
    weights = {k: math.exp(-(v - costs[best]) / temperature) for k, v in costs.items()}
    return {"formation": best, "confidence": round(1.0 / sum(weights.values()), 3),
            "costs": {k: round(v, 1) for k, v in costs.items()}}


def estimate_formations(frames: list[FrameData], cfg: Config, effective_fps: float,
                        attack_direction: Callable[[int, float], Optional[int]]) -> dict:
    window_frames = max(1, int(round(cfg.formation.window_s * effective_fps)))
    result: dict = {"is_inference": True, "windows": {}, "overall": {}}
    for team_id in (0, 1):
        name = cfg.visualization.team_names[team_id]
        estimates = []
        for start in range(0, len(frames), window_frames):
            chunk = frames[start:start + window_frames]
            depths: dict[int, list[float]] = {}
            sample_frames = 0
            for fd in chunk:
                direction = attack_direction(team_id, fd.timestamp)
                members = [o for o in fd.players if o.team_id == team_id and o.on_pitch and o.smoothed_xy is not None]
                if direction is None or len(members) < cfg.tactical.min_players:
                    continue
                sample_frames += 1
                for o in members:
                    x = o.smoothed_xy[0]
                    depths.setdefault(o.player_id, []).append(x if direction > 0 else cfg.field.field_length - x)
            if sample_frames == 0:
                continue
            avg = {pid: float(np.mean(v)) for pid, v in depths.items() if len(v) >= cfg.formation.min_obs_fraction * sample_frames}
            values = np.array(sorted(avg.values()))
            if len(values) == _OUTFIELD_PLAYERS + 1:
                values = values[1:]  # drop the deepest player: probably the goalkeeper
            est = classify_lines(values, cfg.formation.line_penalty, cfg.formation.temperature)
            if est:
                est.update({"start_time": round(chunk[0].timestamp, 1), "end_time": round(chunk[-1].timestamp, 1)})
                estimates.append(est)
        result["windows"][name] = estimates
        if estimates:
            counts = Counter(e["formation"] for e in estimates)
            top, n = counts.most_common(1)[0]
            result["overall"][name] = {"most_frequent_formation": top, "windows_agreeing": n,
                                       "windows_total": len(estimates),
                                       "mean_confidence": round(float(np.mean([e["confidence"] for e in estimates if e["formation"] == top])), 3)}
        else:
            result["overall"][name] = None
    return result
