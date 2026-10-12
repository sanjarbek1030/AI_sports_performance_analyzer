import numpy as np

from synthetic import analyse, at, build_frames, linear, piecewise_ball, stand
from sports_analyzer.analytics.statistics import (DataQuality, PlayerStatistics, TeamStatistics, report_tracks,
                                                  validate_possession_percentages)
from sports_analyzer.config import Config
from sports_analyzer.models import Event, EventType, PlayerTrack, PlayerObservation, PossessionSpell


def spell(pid, team, t0, t1):
    return PossessionSpell(pid, team, int(t0 * 25), int(t1 * 25), t0, t1, t1 - t0, 0.9, (0, 0), (0, 0))


def track(pid, team, seconds=10.0, distance=1000.0, sprints=2):
    tr = PlayerTrack(pid, team_id=team, team_confidence=0.9)
    tr.observations = [PlayerObservation(pid, pid, i, i / 25, (0, 0, 1, 1), 0.9, "player", (0, 0)) for i in range(int(seconds * 25))]
    tr.movement_available, tr.total_distance_m, tr.average_speed, tr.maximum_speed, tr.sprint_count = True, distance, 12.0, 28.0, sprints
    return tr


def test_team_possession_percentages_and_validation():
    tracks = {1: track(1, 0), 2: track(2, 1)}
    spells = [spell(1, 0, 0, 6), spell(2, 1, 6, 10), spell(1, 0, 10, 12)]
    quality = DataQuality(True, 0.9, 0.15)
    df, details = TeamStatistics(Config(), quality).compute(tracks, spells, [])
    pcts = df["possession_percentage"].astype(float).tolist()
    assert pcts == [66.7, 33.3] and abs(sum(pcts) - 100) < 0.2
    assert validate_possession_percentages(df) is None
    assert details[0]["possession_changes_gained"] == 1 and details[1]["possession_changes_gained"] == 1
    bad = df.copy(); bad.loc[0, "possession_percentage"] = 80.0
    assert validate_possession_percentages(bad) is not None


def test_player_and_team_aggregation_from_events():
    tracks = {1: track(1, 0), 2: track(2, 0), 3: track(3, 1)}
    events = [
        Event(EventType.PASS, 1, 25, 1, 0, 0.8, {"to_player": 2, "successful": True}),
        Event(EventType.PASS, 2, 50, 1, 0, 0.8, {"to_player": 3, "successful": False}),
        Event(EventType.SHOT, 3, 75, 2, 0, 0.7, {"outcome": "saved_candidate"}),
        Event(EventType.SHOT, 4, 100, 3, 1, 0.7, {"outcome": "off_target"}),
        Event(EventType.GOAL, 5, 125, 2, 0, 0.7, {"assist_player_id": 1}),
    ]
    quality = DataQuality(True, 0.9, 0.15)
    ps = PlayerStatistics(Config(), quality)
    ps.apply_events(tracks, [spell(1, 0, 0, 4)], events)
    df = ps.to_dataframe(tracks).set_index("player_id")
    assert (df.loc[1, "passes"], df.loc[1, "successful_passes"], df.loc[1, "assists"]) == (2, 1, 1)
    assert (df.loc[2, "shots"], df.loc[2, "shots_on_target"], df.loc[2, "goals"]) == (1, 1, 1)
    assert df.loc[3, "shots_on_target"] == 0 and abs(float(df.loc[1, "possession_seconds"]) - 4.0) < 1e-6
    team, _ = TeamStatistics(Config(), quality).compute(tracks, [], events)
    a = team.iloc[0]
    assert (a["passes"], a["successful_passes"], a["shots"], a["shots_on_target"], a["goals"]) == (2, 1, 1, 1, 1)
    assert a["total_distance_km"] == 2.0 and a["average_distance_km"] == 1.0 and a["sprints"] == 4


def test_unavailable_statistics_are_none_not_zero():
    tracks = {1: track(1, 0)}
    uncalibrated = PlayerStatistics(Config(), DataQuality(False, 0.0, 0.15)).to_dataframe(tracks)
    for col in ["distance_km", "max_speed_kmh", "sprints", "passes", "possession_seconds", "goals"]:
        assert uncalibrated.loc[0, col] is None
    no_ball = PlayerStatistics(Config(), DataQuality(True, 0.02, 0.15)).to_dataframe(tracks)
    assert no_ball.loc[0, "distance_km"] is not None and no_ball.loc[0, "passes"] is None


def test_short_identities_and_referees_are_excluded():
    tracks = {1: track(1, 0, seconds=10), 2: track(2, 0, seconds=1.0), 3: track(3, -2, seconds=10)}
    assert [t.player_id for t in report_tracks(tracks, 3.0)] == [1]


def test_end_to_end_aggregation_with_synthetic_match():
    players = {1: (0, stand(60, 34)), 2: (0, stand(75, 34))}
    ball = piecewise_ball([(0, 2, at((60.5, 34))), (2, 3, linear((60.5, 34), (75.5, 34), 2, 3)), (3, 6, at((75.5, 34)))])
    ctx, possession, events = analyse(build_frames(players, ball, 6))
    assert sum(possession.team_time().values()) > 5.0 and len([e for e in events if e.event_type == "pass"]) == 1
