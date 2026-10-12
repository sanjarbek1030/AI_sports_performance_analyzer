from synthetic import analyse, at, build_frames, linear, piecewise_ball, stand

# Team 0 (ids 1, 2) attacks +x; team 1 (ids 3, 4) attacks -x.
PLAYERS = {1: (0, stand(60, 34)), 2: (0, stand(75, 34)), 3: (1, stand(90, 45)), 4: (1, stand(30, 10))}


def types(events):
    return [e.event_type for e in events]


def test_successful_pass_goal_and_assist_chain():
    ball = piecewise_ball([(0, 2, at((60.5, 34))), (2, 3, linear((60.5, 34), (75.5, 34), 2, 3)),
                           (3, 5, at((75.5, 34))), (5, 6.4, linear((75.5, 34), (105.5, 35), 5, 6.4)),
                           (6.4, 9, at((106.5, 35)))])
    _, _, events = analyse(build_frames(PLAYERS, ball, 10))
    by_type = {e.event_type: e for e in events}
    assert by_type["pass"].player_id == 1 and by_type["pass"].metadata["to_player"] == 2 and by_type["pass"].metadata["successful"]
    assert by_type["shot"].player_id == 2 and by_type["shot"].metadata["outcome"] == "goal"
    goal = by_type["goal"]
    assert goal.team_id == 0 and goal.player_id == 2 and goal.metadata["assist_player_id"] == 1
    assert goal.confidence <= 0.7 and goal.metadata["height_verified"] is False   # honest cap


def test_pass_to_opponent_is_failed_pass_plus_interception_candidate():
    ball = piecewise_ball([(0, 2, at((60.5, 34))), (2, 3, linear((60.5, 34), (89.5, 45), 2, 3)), (3, 6, at((89.5, 45)))])
    _, _, events = analyse(build_frames(PLAYERS, ball, 6))
    pass_event = next(e for e in events if e.event_type == "pass")
    assert pass_event.metadata["successful"] is False and pass_event.metadata["to_team"] == 1
    assert "interception_candidate" in types(events) and "possession_change" in types(events)


def test_shot_wide_of_goal_is_off_target():
    ball = piecewise_ball([(0, 2, at((75.5, 34))), (2, 3.2, linear((75.5, 34), (105.5, 52), 2, 3.2)), (3.2, 6, at((107, 55)))])
    frames = build_frames({2: (0, stand(75, 34)), 3: (1, stand(90, 5))}, ball, 6)
    _, _, events = analyse(frames)
    shots = [e for e in events if e.event_type == "shot"]
    assert shots == [] or shots[0].metadata["outcome"] == "off_target"
    assert "goal" not in types(events)


def test_ball_rolling_past_a_player_is_not_a_pass_or_shot():
    ball = piecewise_ball([(0, 4, linear((40, 34), (44, 34), 0, 4))])      # slow roll, never reaches anyone
    _, _, events = analyse(build_frames(PLAYERS, ball, 4))
    assert not [e for e in events if e.event_type in ("pass", "shot", "goal")]


def test_short_touch_between_adjacent_players_is_not_a_pass():
    players = {1: (0, stand(60, 34)), 2: (0, stand(62, 34))}                  # 2 m apart: transfer < 3 m
    ball = piecewise_ball([(0, 2, at((60.5, 34))), (2, 4, at((62.5, 34)))])
    _, _, events = analyse(build_frames(players, ball, 4))
    assert "pass" not in types(events)


def test_no_goal_when_ball_returns_to_play():
    ball = piecewise_ball([(0, 2, at((75.5, 34))), (2, 3.4, linear((75.5, 34), (105.5, 34), 2, 3.4)),
                           (3.4, 3.6, at((105.5, 34))), (3.6, 4.6, linear((105.3, 34), (95, 34), 3.6, 4.6))])  # bounces back out
    _, _, events = analyse(build_frames({2: (0, stand(75, 34)), 3: (1, stand(95, 40))}, ball, 6))
    assert "goal" not in types(events)


def test_ball_leaving_pitch_gives_out_of_bounds():
    ball = piecewise_ball([(0, 1, at((50, 34))), (1, 3, linear((50, 34), (50, 75), 1, 3)), (3, 6, at((50, 75)))])
    _, _, events = analyse(build_frames({1: (0, stand(50, 34.5))}, ball, 6))
    oob = [e for e in events if e.event_type == "out_of_bounds"]
    assert len(oob) == 1 and oob[0].metadata["boundary"] == "touchline"


def test_events_have_required_fields_and_bounded_confidence():
    ball = piecewise_ball([(0, 2, at((60.5, 34))), (2, 3, linear((60.5, 34), (75.5, 34), 2, 3)), (3, 5, at((75.5, 34)))])
    _, _, events = analyse(build_frames(PLAYERS, ball, 5))
    assert events
    for e in events:
        d = e.to_dict()
        assert set(["event", "timestamp", "frame", "player_id", "team_id", "confidence", "metadata"]) <= set(d)
        assert 0.0 <= e.confidence <= 0.85 and e.inferred
