"""Analysis stage: everything that happens AFTER perception.

Why two stages? Savitzky-Golay smoothing, outlier rejection and event
confirmation need future frames. Running them offline over the stored
per-frame observations is both more accurate and easier to test than a
strictly causal per-frame update. The pipeline therefore does:

    pass 1 (perception, per frame): detect -> track -> identity -> jersey colour
                                    -> field mapping -> ball tracking
    analysis (this module)        : teams -> movement -> ball kinematics ->
                                    possession -> events -> statistics -> tactics
    pass 2 (render, per frame)    : overlay + mini-map using the finished analysis

``run_analysis`` takes plain data (``FrameData``), so tests can feed synthetic
matches without video or models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .analytics.ball_kinematics import BallKinematics, build_ball_kinematics
from .analytics.formation import estimate_formations
from .analytics.movement import MovementAnalyzer
from .analytics.possession import PossessionFrameState, PossessionResult, PossessionState, run_possession
from .analytics.roles import estimate_roles
from .analytics.statistics import (DataQuality, PlayerStatistics, TeamStatistics, validate_possession_percentages)
from .analytics.tactical import compute_team_shape, summarize_tactics
from .analytics.tracks import group_observations_into_tracks
from .config import Config
from .events.base import EventContext, EventEngine
from .exceptions import InsufficientDataError
from .field.football_field import FieldModel
from .models import Event, FrameData, PlayerTrack, PossessionSpell
from .sports.base import SportsAnalyzer
from .teams.classifier import TeamClassifier, assign_track_teams
from .video.processor import VideoInfo

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    video: VideoInfo
    config: Config
    effective_fps: float
    calibrated: bool
    field: FieldModel
    frames: list[FrameData]
    tracks: dict[int, PlayerTrack]
    ball: BallKinematics
    possession: PossessionResult
    events: list[Event]
    player_stats: pd.DataFrame
    team_stats: pd.DataFrame
    team_details: dict
    quality: DataQuality
    shape: pd.DataFrame
    tactical: dict
    formations: dict
    roles: dict[int, dict]
    attack_direction: object = None
    identity_relinks: int = 0
    ball_lost_events: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def spells(self) -> list[PossessionSpell]:
        return self.possession.spells


def run_analysis(frames: list[FrameData], video: VideoInfo, config: Config, sport: SportsAnalyzer,
                 sport_field: FieldModel, calibrated: bool, classifier: Optional[TeamClassifier],
                 identity_relinks: int = 0, ball_lost_events: int = 0) -> AnalysisResult:
    stride = config.video.analysis_stride
    effective_fps = video.fps / stride
    warnings: list[str] = []
    tracks = group_observations_into_tracks(frames)
    if not tracks:
        raise InsufficientDataError("No players were detected/tracked. Check the model, confidence threshold and video.")
    logger.info("Analysing %d frames, %d tracked identities.", len(frames), len(tracks))

    _assign_teams(tracks, classifier, config, warnings)

    if calibrated:
        MovementAnalyzer(config.movement, effective_fps, video.fps).analyze_all(tracks)
    else:
        warnings.append("No field calibration: distance, speed, sprint, possession and event statistics are unavailable.")

    ball = build_ball_kinematics(frames, config.ball, effective_fps)
    quality = DataQuality(calibrated, ball.coverage, config.ball.min_coverage_for_events, warnings)
    if calibrated and not quality.ball_events_reliable:
        warnings.append(f"Ball detected in only {ball.coverage:.0%} of frames (< {config.ball.min_coverage_for_events:.0%}): "
                        "possession, pass, shot and goal statistics are withheld.")

    if calibrated:
        possession = run_possession(frames, ball, config.possession, effective_fps)
        ctx = EventContext(config, sport_field, effective_fps, frames, tracks, possession.spells, ball, sport.attack_direction)
        events = EventEngine(sport.create_event_detectors()).run(ctx)
    else:
        possession = _empty_possession(frames)
        events = []

    player_stats_builder = PlayerStatistics(config, quality)
    player_stats_builder.apply_events(tracks, possession.spells, events)
    player_stats = player_stats_builder.to_dataframe(tracks)
    team_stats, team_details = TeamStatistics(config, quality).compute(tracks, possession.spells, events)
    problem = validate_possession_percentages(team_stats)
    if problem:
        logger.error(problem)
        warnings.append(problem)

    shape, tactical, formations, roles = pd.DataFrame(), {}, {}, {}
    if calibrated and config.tactical.enabled:
        shape = compute_team_shape(frames, possession.frame_states, config, effective_fps)
        sampled = max(1, len(frames) // max(1, int(round(config.tactical.sample_interval_s * effective_fps))))
        tactical = summarize_tactics(shape, config, sampled)
    if calibrated and config.formation.enabled:
        formations = estimate_formations(frames, config, effective_fps, sport.attack_direction)
    if calibrated and config.roles.enabled:
        roles = estimate_roles(tracks, config, sport.attack_direction)

    return AnalysisResult(
        video=video, config=config, effective_fps=effective_fps, calibrated=calibrated, field=sport_field,
        frames=frames, tracks=tracks, ball=ball, possession=possession, events=events, player_stats=player_stats,
        team_stats=team_stats, team_details=team_details, quality=quality, shape=shape, tactical=tactical,
        formations=formations, roles=roles, attack_direction=sport.attack_direction,
        identity_relinks=identity_relinks, ball_lost_events=ball_lost_events, warnings=warnings,
    )


def _assign_teams(tracks, classifier, config: Config, warnings: list[str]) -> None:
    for track in tracks.values():  # referees never belong to a team, with or without a classifier
        if track.role == "referee":
            track.team_id, track.team_confidence = -2, 1.0
            for obs in track.observations:
                obs.team_id = -2
    features = [o.jersey_feature for t in tracks.values() for o in t.observations if o.jersey_feature is not None]
    if classifier is None:
        warnings.append("No team classifier configured: all players are 'Unknown' team.")
        return
    try:
        import numpy as np

        classifier.fit(np.stack(features) if features else np.empty((0, 3)))
    except InsufficientDataError as exc:
        logger.warning("Team classification skipped: %s", exc)
        warnings.append(f"Team classification skipped: {exc}")
        return
    assign_track_teams(classifier, tracks, config.team)
    known = sum(1 for t in tracks.values() if t.team_id in (0, 1))
    logger.info("Team classification: %d/%d identities assigned to a team.", known, len(tracks))


def _empty_possession(frames: list[FrameData]) -> PossessionResult:
    states = [PossessionFrameState(f.frame, f.timestamp, PossessionState.FREE, None, None, None, None) for f in frames]
    return PossessionResult([], states)
