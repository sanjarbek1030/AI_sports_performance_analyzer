"""End-to-end: real pipeline on a synthetic video, fake YOLO detectors."""
import csv
import json
import tempfile
from pathlib import Path

import cv2

import e2e_helpers as h
from sports_analyzer import pipeline as pipeline_module
from sports_analyzer.config import Config
from sports_analyzer.pipeline import AnalysisPipeline


def run_pipeline(tmp: Path, **overrides):
    h.write_video(tmp / "in.mp4")
    h.write_calibration(tmp / "cal.json")
    cfg = Config()
    cfg.paths.input_video = str(tmp / "in.mp4")
    cfg.paths.output_dir = str(tmp / "out")
    cfg.field.calibration_file = str(tmp / "cal.json")
    cfg.tracking.tracker = "iou"
    cfg.team.feature_every_n_frames = 2
    cfg.heatmap.min_player_seconds = 3.0
    for key, value in overrides.items():
        setattr(cfg.video, key, value)
    original = (pipeline_module.PlayerDetector, pipeline_module.BallDetector)
    pipeline_module.PlayerDetector, pipeline_module.BallDetector = h.FakePlayerDetector, h.FakeBallDetector
    try:
        return AnalysisPipeline(cfg).run(), cfg
    finally:
        pipeline_module.PlayerDetector, pipeline_module.BallDetector = original


def test_full_pipeline_outputs_and_content():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        result, cfg = run_pipeline(tmp)
        out = tmp / "out"
        for name in ["sports_analysis.mp4", "player_statistics.csv", "team_statistics.csv", "event_log.csv",
                     "player_tracks.csv", "match_report.json"]:
            assert (out / name).is_file(), name
        # video: resolution and FPS preserved
        cap = cv2.VideoCapture(str(out / "sports_analysis.mp4"))
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (h.WIDTH, h.HEIGHT)
        assert abs(cap.get(cv2.CAP_PROP_FPS) - h.FPS) < 1e-6
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == int(h.DURATION_S * h.FPS)
        cap.release()
        # stable identities: 10 players -> 10 ids (no ID inflation)
        assert len(result.tracks) == 10
        # teams: 5 + 5
        teams = sorted(t.team_id for t in result.tracks.values())
        assert teams == [0] * 5 + [1] * 5
        # distance of the runner (id with 8 m/s for 3 s = 24 m)
        runner = next(t for t in result.tracks.values() if t.observations[0].field_xy[1] == 15.0)
        assert abs(runner.total_distance_m - 24.0) < 2.5, runner.total_distance_m
        assert runner.sprint_count == 1
        # events: pass 1 -> 2, shot, goal
        kinds = [e.event_type for e in result.events]
        assert "pass" in kinds and "shot" in kinds and "goal" in kinds, kinds
        # possession percentages validated: sum to 100
        pcts = result.team_stats["possession_percentage"].astype(float)
        assert abs(pcts.sum() - 100.0) < 0.2
        # CSV columns and JSON
        with open(out / "player_tracks.csv") as f:
            assert next(csv.reader(f))[:10] == ["frame", "timestamp", "player_id", "team", "image_x", "image_y",
                                                "field_x", "field_y", "speed_kmh", "confidence"]
        report = json.loads((out / "match_report.json").read_text())
        assert set(["video", "match", "teams", "players", "events"]) <= set(report)
        assert report["data_quality"]["ball_event_statistics_available"] is True
        assert (out / "heatmaps" / "match_combined.png").is_file()


def test_stride_keeps_all_output_frames():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        result, _ = run_pipeline(tmp, analysis_stride=2)
        cap = cv2.VideoCapture(str(tmp / "out" / "sports_analysis.mp4"))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == int(h.DURATION_S * h.FPS)
        cap.release()
        assert len(result.frames) == int(h.DURATION_S * h.FPS) // 2
