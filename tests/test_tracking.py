import numpy as np

from sports_analyzer.config import BallConfig, TrackingConfig
from sports_analyzer.models import BallDetection, Detection
from sports_analyzer.tracking.ball_tracker import BallTracker
from sports_analyzer.tracking.identity import PlayerIdentityManager
from sports_analyzer.tracking.iou_tracker import IoUTracker


def det(frame, x, y=200, conf=0.9, w=40, h=100):
    return Detection(frame, frame / 25, (x, y, x + w, y + h), conf, 0, "person")


def test_ids_are_stable_with_shuffled_order_and_short_occlusion():
    rng, tracker, seen = np.random.default_rng(0), IoUTracker(0.35, 0.1, 0.5, 0.8, 50, 2), {}
    for f in range(80):
        dets = [det(f, 100 + k * 150 + 3 * f) for k in range(4) if not (k == 2 and 30 <= f < 40)]
        rng.shuffle(dets)
        for d in tracker.update(dets):
            seen.setdefault(round(d.bbox[0] - 3 * f), set()).add(d.track_id)
    assert len(seen) == 4 and all(len(ids) == 1 for ids in seen.values())


def test_low_confidence_boxes_keep_existing_track_alive():
    tracker = IoUTracker(0.35, 0.1, 0.5, 0.8, 30, 2)
    ids = set()
    for f in range(30):
        for d in tracker.update([det(f, 100 + 3 * f, conf=0.9 if f < 15 else 0.2)]):
            ids.add(d.track_id)
    assert ids == {1}


def test_identity_manager_relinks_after_tracker_forgets():
    manager = PlayerIdentityManager(TrackingConfig())
    color = np.array([10.0, 20.0, 5.0])
    a = det(0, 100); a.track_id = 7
    assert manager.assign([a], [color], 0.0) == [1]
    b = det(75, 150); b.track_id = 42                # new tracker id 3 s later, 50 px away, same colour
    assert manager.assign([b], [color + 1], 3.0) == [1] and manager.relink_count == 1
    c = det(80, 900); c.track_id = 43                # far away -> new player
    assert manager.assign([c], [color], 3.2) == [2]


def test_identity_manager_rejects_different_jersey_colour():
    manager = PlayerIdentityManager(TrackingConfig())
    a = det(0, 100); a.track_id = 1
    manager.assign([a], [np.array([10.0, 20.0, 5.0])], 0.0)
    b = det(10, 110); b.track_id = 2
    assert manager.assign([b], [np.array([-40.0, 30.0, 25.0])], 0.4) == [2]


def test_ball_tracker_coasts_through_misses_and_rejects_jumps():
    tracker, states = BallTracker(BallConfig(), (1920, 1080)), []
    for f in range(40):
        cands = [] if 15 <= f < 18 else [BallDetection(f, f / 25, (500 + 10 * f, 500, 510 + 10 * f, 510), 0.6)]
        if f == 25:
            cands.append(BallDetection(f, f / 25, (1700, 100, 1710, 110), 0.95))   # high-confidence false positive
        states.append(tracker.update(cands, f, f / 25))
    assert states[0] is None                                    # needs confirmation first
    assert [s.predicted for s in states[15:18]] == [True, True, True] and states[18].predicted is False
    assert abs(states[25].center[0] - (505 + 250)) < 15         # followed the consistent ball, not the jump
    assert states[16].confidence < states[14].confidence


def test_ball_tracker_gives_up_after_long_loss():
    tracker = BallTracker(BallConfig(), (1920, 1080))
    for f in range(10):
        tracker.update([BallDetection(f, f / 25, (500 + 10 * f, 500, 510 + 10 * f, 510), 0.6)], f, f / 25)
    results = [tracker.update([], f, f / 25) for f in range(10, 40)]
    assert results[0] is not None and results[-1] is None and tracker.lost_events == 1


def test_tracker_falls_back_to_builtin_when_ultralytics_unavailable():
    from sports_analyzer.config import Config
    from sports_analyzer.tracking.player_tracker import PlayerTracker

    cfg = Config()
    cfg.tracking.tracker = "bytetrack"          # in CI / offline machines Ultralytics may be missing or incompatible
    tracker = PlayerTracker(cfg, 25.0)
    seen = set()
    for f in range(10):
        for d in tracker.update([det(f, 100 + 3 * f)], np.zeros((480, 640, 3), np.uint8)):
            seen.add(d.track_id)
    assert len(seen) == 1                       # works with whichever backend started
