import numpy as np

from helpers import raises
from sports_analyzer.config import TeamConfig
from sports_analyzer.exceptions import InsufficientDataError
from sports_analyzer.models import PlayerObservation, PlayerTrack
from sports_analyzer.teams.classifier import ColorKMeansTeamClassifier, assign_track_teams


def player_image(color, rng, brightness=0):
    frame = np.zeros((200, 100, 3), np.uint8)
    frame[:] = (40, 140, 40)                                   # grass background
    frame[30:120, 20:80] = np.clip(np.array(color) + brightness, 0, 255)
    noise = rng.integers(-12, 12, frame.shape)
    return np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)


def test_two_kits_separated_under_lighting_changes():
    rng, clf = np.random.default_rng(1), ColorKMeansTeamClassifier(TeamConfig())
    feats, labels = [], []
    for i in range(80):
        team = i % 2
        img = player_image((30, 30, 220) if team else (235, 235, 235), rng, brightness=int(rng.integers(-25, 15)))
        feats.append(clf.extract(img, (0, 0, 100, 200))); labels.append(team)
    clf.fit(np.stack(feats))
    preds = clf.predict(np.stack(feats))
    lighter_is_zero = clf.centroids[0, 0] > clf.centroids[1, 0]
    assert lighter_is_zero
    assert np.mean([p.team_id == l for p, l in zip(preds, labels)]) > 0.95


def test_unusual_kit_is_unknown_not_forced_into_a_team():
    rng, clf = np.random.default_rng(2), ColorKMeansTeamClassifier(TeamConfig())
    feats = [clf.extract(player_image((30, 30, 220) if i % 2 else (235, 235, 235), rng), (0, 0, 100, 200)) for i in range(60)]
    clf.fit(np.stack(feats))
    referee = clf.extract(player_image((0, 230, 230), rng), (0, 0, 100, 200))
    assert clf.predict(referee)[0].team_id == -1


def test_grass_only_crop_returns_none_and_few_samples_raise():
    clf = ColorKMeansTeamClassifier(TeamConfig())
    assert clf.extract(np.full((200, 100, 3), (40, 140, 40), np.uint8), (0, 0, 100, 200)) is None
    with raises(InsufficientDataError):
        clf.fit(np.zeros((5, 3)))


def test_track_vote_is_robust_to_a_few_wrong_crops():
    rng, clf = np.random.default_rng(3), ColorKMeansTeamClassifier(TeamConfig())
    feats = [clf.extract(player_image((30, 30, 220) if i % 2 else (235, 235, 235), rng), (0, 0, 100, 200)) for i in range(60)]
    clf.fit(np.stack(feats))
    red, white = clf.extract(player_image((30, 30, 220), rng), (0, 0, 100, 200)), clf.extract(player_image((235, 235, 235), rng), (0, 0, 100, 200))
    track = PlayerTrack(1)
    for i, f in enumerate([red] * 12 + [white] * 2):
        track.observations.append(PlayerObservation(1, 1, i, i / 25, (0, 0, 1, 1), 0.9, "player", (0, 0), jersey_feature=f))
    assign_track_teams(clf, {1: track}, TeamConfig())
    assert track.team_id in (0, 1) and track.team_confidence > 0.3
    assert all(o.team_id == track.team_id for o in track.observations)
