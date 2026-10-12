import numpy as np

from sports_analyzer.analytics.movement import MovementAnalyzer, reject_position_outliers, smooth_series
from sports_analyzer.config import Config
from sports_analyzer.models import PlayerObservation, PlayerTrack

FPS = 25.0


def make_track(speed_fn, seconds=10.0, noise=0.0, outlier_frame=None, seed=0):
    rng, track, x = np.random.default_rng(seed), PlayerTrack(1), 0.0
    for i in range(int(seconds * FPS)):
        t = i / FPS
        x += speed_fn(t) / FPS
        xy = (x + rng.normal(0, noise), 30 + rng.normal(0, noise))
        if i == outlier_frame:
            xy = (x + 30.0, 30.0)
        track.observations.append(PlayerObservation(1, 1, i, t, (0, 0, 1, 1), 0.9, "player", (0, 0), field_xy=xy, on_pitch=True))
    return track


def analyzer():
    return MovementAnalyzer(Config().movement, FPS, FPS)


def test_distance_and_speed_constant_velocity():
    track = make_track(lambda t: 5.0)
    analyzer().analyze_track(track)
    assert abs(track.total_distance_m - 50.0) < 0.5
    assert abs(track.average_speed - 18.0) < 0.3
    assert abs(track.maximum_speed - 18.0) < 0.5


def test_noise_and_teleport_do_not_create_absurd_speed():
    track = make_track(lambda t: 5.0, noise=0.3, outlier_frame=100)
    analyzer().analyze_track(track)
    assert track.maximum_speed < 40.0          # plausibility ceiling
    assert abs(track.total_distance_m - 50.0) < 3.0   # raw jitter would give hundreds of metres
    assert track.observations[100].is_outlier


def test_outlier_rejection_resyncs_after_genuine_jump():
    t = np.arange(40) / FPS
    xy = np.stack([np.where(np.arange(40) < 20, 0.0, 50.0), np.zeros(40)], axis=1)
    keep = reject_position_outliers(t, xy, 12.0)
    assert keep[:20].all() and keep[-1]        # accepted again after the resync limit


def test_acceleration_and_deceleration():
    ramp = make_track(lambda t: min(2.0 * t, 8.0) if t < 6 else max(8.0 - 5.0 * (t - 6), 0.0))
    analyzer().analyze_track(ramp)
    accel = ramp.accelerations
    assert abs(np.nanmedian(accel[25:80]) - 2.0) < 0.2
    assert np.nanmin(accel) < -3.0
    kinds = {m.kind for m in ramp.movement_events}
    assert "rapid_deceleration" in kinds


def test_sprint_requires_minimum_duration():
    def profile(t):
        if 3.0 <= t < 5.2:
            return 8.0        # 28.8 km/h for 2.2 s -> sprint
        if 7.0 <= t < 7.2:
            return 9.5        # 0.2 s spike -> NOT a sprint
        return 2.0
    track = make_track(profile)
    analyzer().analyze_track(track)
    assert track.sprint_count == 1
    sprint = track.sprints[0]
    assert 3.0 <= sprint.start_time <= 3.6 and sprint.duration_s >= 1.5 and sprint.max_speed_kmh > 25.2
    assert sprint.distance_m > 10


def test_gaps_are_not_bridged():
    track = make_track(lambda t: 5.0, seconds=8.0)
    track.observations = [o for o in track.observations if not (3.0 <= o.timestamp < 5.0)]   # 2 s occlusion
    analyzer().analyze_track(track)
    assert 25.0 < track.total_distance_m < 32.0     # ~6 s of motion, the 2 s gap contributes nothing


def test_direction_change_detected():
    track = PlayerTrack(1)
    for i in range(int(8 * FPS)):
        t = i / FPS
        x, y = (6.0 * t, 20.0) if t < 4 else (24.0, 20.0 + 6.0 * (t - 4))      # 90 degree turn
        track.observations.append(PlayerObservation(1, 1, i, t, (0, 0, 1, 1), 0.9, "player", (0, 0), field_xy=(x, y), on_pitch=True))
    analyzer().analyze_track(track)
    assert any(m.kind == "direction_change" for m in track.movement_events)


def test_smoothing_methods_preserve_length():
    values = np.random.default_rng(0).normal(size=(50, 2))
    for method in ("savgol", "moving_average", "none"):
        assert smooth_series(values, method, 9, 2).shape == values.shape
