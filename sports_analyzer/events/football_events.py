"""Football event detectors. Every event here is an INFERENCE from tracking
data, not ground truth; confidences are heuristic scores in [0, cap].

Common ingredients: possession spells (who had the ball and when), ball
kinematics in field metres, and pitch geometry (goal mouth, boundaries).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..models import Event, EventType, PossessionSpell, TEAM_UNKNOWN
from ..utils import clip01, find_runs
from .base import EventContext, EventDetector

# Confidence blending weights (each set sums to 1).
_W_PASS = {"spell": 0.40, "ball": 0.30, "direction": 0.30}
_W_SHOT = {"speed": 0.35, "direction": 0.25, "ball": 0.20, "spell": 0.20}
_NEUTRAL_DIRECTION_SCORE = 0.5
_MIN_BALL_SAMPLES = 3


def _known(team: Optional[int]) -> bool:
    return team is not None and team != TEAM_UNKNOWN and team >= 0


def _spell_ball_fraction(ctx: EventContext, t0: float, t1: float) -> float:
    idx = ctx.ball.window(t0, t1)
    return float(ctx.ball.observed[idx].mean()) if len(idx) else 0.0


def _release_velocity(ctx: EventContext, t_release: float, window_s: float) -> Optional[np.ndarray]:
    idx = ctx.ball.window(t_release, t_release + window_s)
    vel = ctx.ball.velocity[idx]
    vel = vel[np.isfinite(vel).all(axis=1)]
    return vel.mean(axis=0) if len(vel) >= 2 else None


# ----------------------------------------------------------------------------
class PossessionChangeDetector(EventDetector):
    name = "possession_change"

    def detect(self, ctx: EventContext) -> list[Event]:
        events = []
        for a, b in zip(ctx.spells, ctx.spells[1:]):
            if _known(a.team_id) and _known(b.team_id) and a.team_id != b.team_id:
                events.append(Event(
                    EventType.POSSESSION_CHANGE, b.start_time, b.start_frame, b.player_id, b.team_id,
                    min(a.confidence, b.confidence),
                    {"from_team": a.team_id, "from_player": a.player_id, "gap_s": round(max(b.start_time - a.end_time, 0), 3)},
                ))
        return events


# ----------------------------------------------------------------------------
class ShotDetector(EventDetector):
    """possession -> rapid release -> high ball speed -> trajectory toward goal."""

    name = "shot"

    def detect(self, ctx: EventContext) -> list[Event]:
        cfg, ball = ctx.config.events, ctx.ball
        events: list[Event] = []
        for k, spell in enumerate(ctx.spells):
            if not _known(spell.team_id) or spell.end_ball_xy is None:
                continue
            t_rel = spell.end_time
            direction = ctx.attack_direction(spell.team_id, t_rel)
            if direction is None:
                continue
            idx = ball.window(t_rel, t_rel + cfg.shot_speed_window_s)
            speeds = ball.speed[idx]
            if np.isfinite(speeds).sum() < _MIN_BALL_SAMPLES:
                continue
            v_peak = float(np.nanmax(speeds))
            if v_peak < cfg.shot_min_speed_ms:
                continue
            fast = idx[np.isfinite(ball.speed[idx]) & (ball.speed[idx] >= 0.7 * v_peak)]
            vel = np.nanmean(ball.velocity[fast], axis=0)
            gx, y_lo, y_hi = ctx.field.goal_mouth(direction)
            x0, y0 = spell.end_ball_xy
            if abs(gx - x0) > cfg.shot_max_distance_m or vel[0] * direction <= 0:
                continue
            ang_err, cone = self._angle_to_goal(vel, x0, y0, gx, y_lo, y_hi)
            tolerance = cone + math.radians(cfg.shot_angle_margin_deg)
            if ang_err > tolerance:
                continue
            nxt = ctx.spells[k + 1] if k + 1 < len(ctx.spells) else None
            if nxt and nxt.start_time - t_rel <= cfg.shot_outcome_window_s and nxt.team_id == spell.team_id \
                    and nxt.player_id != spell.player_id:
                continue  # received by a team-mate: a (long) pass, not a shot
            y_cross = float(y0 + vel[1] * (gx - x0) / vel[0]) if abs(vel[0]) > 1e-6 else math.nan
            outcome, meta = self._outcome(ctx, spell, nxt, t_rel, v_peak, x0, gx, direction, y_cross, y_lo, y_hi)
            speed_score = clip01((v_peak - cfg.shot_min_speed_ms) / cfg.shot_min_speed_ms)
            dir_score = clip01(1.0 - ang_err / max(tolerance, 1e-6))
            ball_score = float(ball.observed[idx].mean())
            conf = (_W_SHOT["speed"] * speed_score + _W_SHOT["direction"] * dir_score
                    + _W_SHOT["ball"] * ball_score + _W_SHOT["spell"] * spell.confidence)
            conf = min(conf * (0.5 + 0.5 * max(ctx.team_confidence(spell.player_id), 0.3)), cfg.confidence_cap)
            events.append(Event(EventType.SHOT, t_rel, spell.end_frame, spell.player_id, spell.team_id, conf, {
                "outcome": outcome, "outcome_is_inferred": True, "estimated_speed_kmh": round(v_peak * 3.6, 1),
                "distance_to_goal_line_m": round(abs(gx - x0), 1), "projected_goal_line_y": None if math.isnan(y_cross) else round(y_cross, 2),
                "spell_start_frame": spell.start_frame, "attack_direction": direction, **meta,
            }))
        return events

    @staticmethod
    def _angle_to_goal(vel, x0, y0, gx, y_lo, y_hi) -> tuple[float, float]:
        to_lo = math.atan2(y_lo - y0, gx - x0)
        to_hi = math.atan2(y_hi - y0, gx - x0)
        to_center = math.atan2((y_lo + y_hi) / 2 - y0, gx - x0)
        heading = math.atan2(vel[1], vel[0])
        err = abs((heading - to_center + math.pi) % (2 * math.pi) - math.pi)
        cone = abs(to_hi - to_lo) / 2
        return err, cone

    def _outcome(self, ctx, spell, nxt, t_rel, v_peak, x0, gx, direction, y_cross, y_lo, y_hi):
        cfg, ball = ctx.config.events, ctx.ball
        on_line = (not math.isnan(y_cross)) and (y_lo - cfg.post_margin_m <= y_cross <= y_hi + cfg.post_margin_m)
        idx = ball.window(t_rel + cfg.shot_speed_window_s, t_rel + cfg.shot_outcome_window_s)
        xs = ball.xy[idx, 0] if len(idx) else np.array([])
        progress = 0.0
        if len(xs) and np.isfinite(xs).any():
            progress = float((np.nanmax(xs * direction) - x0 * direction) / max(abs(gx - x0), 1e-6))
        later_speed = ball.speed[idx]
        slowed = len(later_speed) > 0 and np.isfinite(later_speed).any() and np.nanmin(later_speed) <= cfg.block_speed_drop * v_peak
        near_defender = False
        if slowed:
            j = idx[int(np.nanargmin(np.where(np.isfinite(later_speed), later_speed, np.inf)))]
            bx, by = ball.xy[j]
            frame = int(ball.frames[j])
            for obs in ctx.frames[j].players:
                if obs.team_id != spell.team_id and _known(obs.team_id) and obs.smoothed_xy is not None \
                        and math.dist(obs.smoothed_xy, (bx, by)) <= cfg.block_player_radius_m:
                    near_defender = True
                    break
        if on_line and progress < cfg.block_progress_fraction and near_defender:
            return "blocked_candidate", {}
        if on_line:
            if nxt and nxt.team_id != spell.team_id and nxt.start_time - t_rel <= cfg.shot_outcome_window_s \
                    and nxt.start_ball_xy is not None and abs(gx - nxt.start_ball_xy[0]) <= cfg.save_goal_distance_m:
                return "saved_candidate", {"collected_by_player": nxt.player_id}
            return "on_target", {}
        if near_defender and progress < cfg.block_progress_fraction:
            return "blocked_candidate", {}
        return "off_target", {}


# ----------------------------------------------------------------------------
class PassDetector(EventDetector):
    """Consecutive possession spells + ball travel between them."""

    name = "pass"

    def detect(self, ctx: EventContext) -> list[Event]:
        cfg = ctx.config.events
        shot_spells = {e.metadata.get("spell_start_frame") for e in ctx.events_of(EventType.SHOT)}
        events: list[Event] = []
        for a, b in zip(ctx.spells, ctx.spells[1:]):
            if a.player_id == b.player_id or not (_known(a.team_id) and _known(b.team_id)):
                continue
            if a.start_frame in shot_spells or a.start_ball_xy is None or a.end_ball_xy is None or b.start_ball_xy is None:
                continue
            flight = max(b.start_time - a.end_time, 0.0)
            if flight > cfg.pass_max_flight_s:
                continue
            travel = math.dist(a.end_ball_xy, b.start_ball_xy)
            if travel < cfg.pass_min_distance_m:
                continue
            avg_speed = travel / max(flight, ctx.frame_dt)
            if avg_speed < cfg.pass_min_avg_speed_ms or avg_speed > ctx.config.ball.max_plausible_speed_ms:
                continue
            if self._left_pitch(ctx, a.end_time, b.start_time):
                continue
            direction_score = self._direction_score(ctx, a, b, travel, cfg.pass_min_direction_cosine)
            if direction_score is None:
                continue
            ball_score = _spell_ball_fraction(ctx, a.end_time, b.start_time) if flight > 0 else 1.0
            conf = (_W_PASS["spell"] * min(a.confidence, b.confidence) + _W_PASS["ball"] * ball_score
                    + _W_PASS["direction"] * direction_score)
            conf *= 0.5 + 0.5 * max(min(ctx.team_confidence(a.player_id), ctx.team_confidence(b.player_id)), 0.3)
            successful = a.team_id == b.team_id
            if not successful:
                conf *= 0.8  # an "unsuccessful pass" is also what a clearance/loose ball looks like
            events.append(Event(EventType.PASS, a.end_time, a.end_frame, a.player_id, a.team_id,
                                min(conf, cfg.confidence_cap), {
                "to_player": b.player_id, "to_team": b.team_id, "successful": successful,
                "distance_m": round(travel, 1), "flight_s": round(flight, 2), "avg_ball_speed_kmh": round(avg_speed * 3.6, 1),
                "receive_frame": b.start_frame,
            }))
        return events

    @staticmethod
    def _left_pitch(ctx: EventContext, t0: float, t1: float) -> bool:
        idx = ctx.ball.window(t0, t1)
        margin = ctx.config.events.out_margin_m
        for j in idx:
            if ctx.ball.observed[j] and np.isfinite(ctx.ball.xy[j]).all() and not ctx.field.contains(*ctx.ball.xy[j], margin):
                return True
        return False

    @staticmethod
    def _direction_score(ctx, a: PossessionSpell, b: PossessionSpell, travel: float, min_cos: float) -> Optional[float]:
        vel = _release_velocity(ctx, a.end_time, 0.3)
        if vel is None or np.linalg.norm(vel) < 1e-6:
            return _NEUTRAL_DIRECTION_SCORE
        disp = np.array(b.start_ball_xy) - np.array(a.end_ball_xy)
        cos = float(np.dot(vel, disp) / (np.linalg.norm(vel) * np.linalg.norm(disp) + 1e-9))
        return None if cos < min_cos else clip01((cos - min_cos) / (1 - min_cos))


# ----------------------------------------------------------------------------
class GoalDetector(EventDetector):
    """Ball crosses the goal line inside the posts, then stays out of play."""

    name = "goal"

    def detect(self, ctx: EventContext) -> list[Event]:
        cfg, ball = ctx.config.events, ctx.ball
        events: list[Event] = []
        valid = np.flatnonzero(np.isfinite(ball.xy).all(axis=1))
        for side in (-1, +1):
            gx, y_lo, y_hi = ctx.field.goal_mouth(side)
            last_goal_time = -math.inf
            for a, b in zip(valid[:-1], valid[1:]):
                xa, ya = ball.xy[a]; xb, yb = ball.xy[b]
                inside_before, beyond_after = (xa - gx) * side < 0, (xb - gx) * side >= 0
                t_cross = None
                kind = ""
                if inside_before and beyond_after:
                    frac = (gx - xa) / (xb - xa) if xb != xa else 0.0
                    y_c = ya + frac * (yb - ya)
                    if y_lo - cfg.post_margin_m <= y_c <= y_hi + cfg.post_margin_m and self._approach_ok(ball, a, side, cfg):
                        t_cross, kind = ball.times[a] + frac * (ball.times[b] - ball.times[a]), "line_crossing"
                elif inside_before and ball.times[b] - ball.times[a] >= cfg.goal_confirm_s:
                    near_line = abs(gx - xa) <= cfg.goal_lost_near_line_m and y_lo <= ya <= y_hi
                    if near_line and self._approach_ok(ball, a, side, cfg):
                        t_cross, kind = ball.times[a], "lost_near_goal_line"
                if t_cross is None or t_cross - last_goal_time < cfg.goal_cooldown_s:
                    continue
                if not self._stays_out(ball, b if kind == "line_crossing" else a, gx, side, t_cross, cfg):
                    continue
                last_goal_time = t_cross
                events.append(self._make_event(ctx, side, t_cross, kind, a))
        events.sort(key=lambda e: e.timestamp)
        return events

    @staticmethod
    def _approach_ok(ball, i, side, cfg) -> bool:
        vx = ball.velocity[i, 0]
        return bool(np.isfinite(vx) and vx * side >= cfg.goal_min_approach_speed_ms * 0.5 and ball.speed[i] >= cfg.goal_min_approach_speed_ms)

    @staticmethod
    def _stays_out(ball, start_idx, gx, side, t_cross, cfg) -> bool:
        """Temporal confirmation: no re-entry into open play during the confirm window."""
        idx = ball.window(t_cross + 1e-6, t_cross + cfg.goal_confirm_s)
        for j in idx:
            if np.isfinite(ball.xy[j]).all() and (ball.xy[j, 0] - gx) * side < -cfg.goal_reentry_m:
                return False
        return True

    def _make_event(self, ctx: EventContext, side: int, t_cross: float, kind: str, idx: int) -> Event:
        cfg = ctx.config.events
        frame = int(ctx.ball.frames[idx])
        attackers = [team for team in (0, 1) if ctx.attack_direction(team, t_cross) == side]
        team = attackers[0] if attackers else None
        scorer = None
        for spell in reversed(ctx.spells):
            if spell.end_time <= t_cross and t_cross - spell.end_time <= cfg.goal_attribution_window_s and spell.team_id == team:
                scorer = spell
                break
        base = 0.55 if kind == "line_crossing" else 0.35
        conf = base + 0.15 * float(ctx.ball.observed[max(idx - 1, 0): idx + 2].mean()) + 0.1 * (scorer.confidence if scorer else 0.0)
        conf = min(conf, cfg.goal_confidence_cap)
        event = Event(EventType.GOAL, float(t_cross), frame, scorer.player_id if scorer else None, team, conf, {
            "detection": kind, "height_verified": False,
            "note": "Ground-plane homography cannot tell a goal from a ball over the crossbar.",
            "team_attribution": "attack_direction", "goal_side_x": 0 if side < 0 else ctx.field.length,
        })
        # link to the shot and the assist
        for shot in ctx.events_of(EventType.SHOT):
            if 0 <= t_cross - shot.timestamp <= cfg.goal_attribution_window_s and shot.team_id == team:
                shot.metadata["outcome"] = "goal"
                event.player_id = event.player_id if event.player_id is not None else shot.player_id
                event.metadata["linked_shot_frame"] = shot.frame
        if event.player_id is not None:
            for p in reversed(ctx.events_of(EventType.PASS)):
                if p.metadata.get("successful") and p.metadata.get("to_player") == event.player_id \
                        and 0 <= t_cross - p.timestamp <= cfg.assist_window_s and p.player_id != event.player_id:
                    event.metadata["assist_player_id"] = p.player_id
                    break
        return event


# ----------------------------------------------------------------------------
class TackleInterceptionDetector(EventDetector):
    """Team-changing possession transitions classified by how the ball moved."""

    name = "tackle_interception"

    def detect(self, ctx: EventContext) -> list[Event]:
        cfg = ctx.config.events
        events: list[Event] = []
        for a, b in zip(ctx.spells, ctx.spells[1:]):
            if not (_known(a.team_id) and _known(b.team_id)) or a.team_id == b.team_id:
                continue
            if a.end_ball_xy is None or b.start_ball_xy is None:
                continue
            gap = b.start_time - a.end_time
            travel = math.dist(a.end_ball_xy, b.start_ball_xy)
            cap = cfg.weak_event_confidence_cap
            conf = min(min(a.confidence, b.confidence) * 0.8, cap)
            pa, pb = ctx.player_xy(a.player_id, a.end_frame), ctx.player_xy(b.player_id, b.start_frame)
            close = pa is not None and pb is not None and math.dist(pa, pb) <= cfg.tackle_max_player_distance_m
            if gap <= cfg.tackle_max_gap_s and travel < cfg.pass_min_distance_m and close:
                events.append(Event(EventType.TACKLE_CANDIDATE, b.start_time, b.start_frame, b.player_id, b.team_id, conf,
                                    {"against_player": a.player_id, "gap_s": round(max(gap, 0), 2)}))
            elif cfg.pass_min_distance_m <= travel and gap <= cfg.pass_max_flight_s:
                events.append(Event(EventType.INTERCEPTION_CANDIDATE, b.start_time, b.start_frame, b.player_id, b.team_id, conf,
                                    {"from_player": a.player_id, "ball_travel_m": round(travel, 1)}))
        return events


# ----------------------------------------------------------------------------
class OutOfBoundsDetector(EventDetector):
    name = "out_of_bounds"

    def detect(self, ctx: EventContext) -> list[Event]:
        cfg, ball, fld = ctx.config.events, ctx.ball, ctx.field
        outside = np.zeros(len(ball), dtype=bool)
        for i in range(len(ball)):
            if ball.observed[i] and np.isfinite(ball.xy[i]).all():
                outside[i] = not fld.contains(ball.xy[i, 0], ball.xy[i, 1], cfg.out_margin_m)
        min_len = max(2, int(round(cfg.out_min_duration_s * ctx.effective_fps)))
        events: list[Event] = []
        last = -math.inf
        for s, e in find_runs(outside, min_len):
            t = float(ball.times[s])
            if t - last < cfg.out_cooldown_s:
                continue
            x, y = ball.xy[s]
            boundary = "goal_line" if (x < 0 or x > fld.length) else "touchline"
            if boundary == "goal_line":
                side = -1 if x < 0 else 1
                _, y_lo, y_hi = fld.goal_mouth(side)
                if y_lo - cfg.post_margin_m <= y <= y_hi + cfg.post_margin_m:
                    continue  # between the posts: handled by the goal detector
            last = t
            previous = [sp for sp in ctx.spells if sp.end_time <= t]
            last_touch = previous[-1] if previous and t - previous[-1].end_time < ctx.config.events.pass_max_flight_s * 2 else None
            events.append(Event(EventType.OUT_OF_BOUNDS, t, int(ball.frames[s]),
                                last_touch.player_id if last_touch else None, last_touch.team_id if last_touch else None,
                                min(0.5 + 0.3 * float(ball.observed[s:e + 1].mean()), cfg.confidence_cap),
                                {"boundary": boundary, "ball_xy": [round(float(x), 1), round(float(y), 1)],
                                 "last_touch_is_approximate": True}))
        return events


# ----------------------------------------------------------------------------
class SprintEventDetector(EventDetector):
    name = "sprint"

    def detect(self, ctx: EventContext) -> list[Event]:
        cap = ctx.config.events.confidence_cap
        events = []
        for track in ctx.tracks.values():
            for sp in track.sprints:
                events.append(Event(EventType.SPRINT, sp.start_time, sp.start_frame, track.player_id, track.team_id,
                                    min(0.5 + 0.4 * sp.observed_fraction, cap), {
                    "duration_s": round(sp.duration_s, 2), "max_speed_kmh": round(sp.max_speed_kmh, 1),
                    "distance_m": round(sp.distance_m, 1), "end_time": round(sp.end_time, 2)}))
        return events
