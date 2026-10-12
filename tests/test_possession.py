from sports_analyzer.analytics.possession import PlayerPoint, PossessionAnalyzer, PossessionState
from sports_analyzer.config import PossessionConfig

FPS = 25.0
PLAYERS = [PlayerPoint(1, 0, (10.0, 10.0)), PlayerPoint(2, 1, (30.0, 10.0))]


def run(ball_positions, players=PLAYERS):
    analyzer = PossessionAnalyzer(PossessionConfig(), FPS)
    states = [analyzer.step(i, i / FPS, b, players) for i, b in enumerate(ball_positions)]
    return analyzer, states


def test_single_frame_proximity_is_not_possession():
    ball = [(50.0, 50.0)] * 5 + [(10.5, 10.0)] + [(50.0, 50.0)] * 10
    analyzer, states = run(ball)
    assert not analyzer.finalize() and all(s.state != PossessionState.POSSESSION for s in states)


def test_state_machine_sequence():
    ball = [(50.0, 50.0)] * 3 + [(10.5, 10.0)] * 20 + [(25.0, 40.0)] * 30
    analyzer, states = run(ball)
    order = []
    for s in states:
        if not order or order[-1] != s.state:
            order.append(s.state)
    assert order == [PossessionState.FREE, PossessionState.CANDIDATE, PossessionState.POSSESSION,
                     PossessionState.RELEASE, PossessionState.FREE]
    spells = analyzer.finalize()
    assert len(spells) == 1 and spells[0].player_id == 1 and spells[0].team_id == 0
    assert 0.7 < spells[0].duration < 1.0 and spells[0].end_reason == "ball_left"


def test_short_ball_dropout_does_not_end_possession():
    ball = [(10.5, 10.0)] * 15 + [None] * 5 + [(10.5, 10.0)] * 15
    analyzer, _ = run(ball)
    spells = analyzer.finalize()
    assert len(spells) == 1 and spells[0].end_reason == "end_of_video"


def test_long_dropout_releases_with_reason():
    ball = [(10.5, 10.0)] * 15 + [None] * 40
    analyzer, _ = run(ball)
    assert analyzer.finalize()[0].end_reason == "ball_or_player_lost"


def test_challenger_takes_over_when_clearly_closer():
    players = [PlayerPoint(1, 0, (10.0, 10.0)), PlayerPoint(2, 1, (11.5, 10.0))]
    ball = [(10.4, 10.0)] * 15 + [(11.4, 10.0)] * 20
    analyzer, _ = run(ball, players)
    spells = analyzer.finalize()
    assert [s.player_id for s in spells] == [1, 2] and spells[0].end_reason == "challenged"


def test_referees_are_ignored():
    players = [PlayerPoint(99, -2, (10.0, 10.0))]
    analyzer, _ = run([(10.3, 10.0)] * 20, players)
    assert not analyzer.finalize()
