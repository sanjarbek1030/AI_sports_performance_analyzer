"""AnalysisPipeline: orchestrates the components; contains no analytics logic.

    initialize configuration / video / models / trackers / analytics / writer
    pass 1, per frame:  detect players + ball -> track players + ball -> identity
                        -> jersey colour feature -> map to field coordinates
    analysis:           teams -> movement -> ball trajectory -> possession ->
                        events -> statistics -> tactical / formation / roles
    pass 2, per frame:  render overlay + mini-map -> write frame
    finish:             heatmaps, CSV, JSON, summary

Pass 1 and pass 2 are separate because smoothing and event confirmation need
future frames (see ``analysis.py``). The video is decoded twice, which is cheap
next to neural-network inference.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .analysis import AnalysisResult, run_analysis
from .analytics.heatmaps import HeatmapGenerator
from .config import Config, resolve_device
from .detection.ball_detector import BallDetector
from .detection.player_detector import PlayerDetector
from .exceptions import CalibrationError, VideoReadError
from .field.calibration import CalibrationProvider, ManualCalibration, NullCalibration
from .field.field_mapper import FieldMapper
from .models import BallState, Detection, DetectionRole, FrameData, PlayerObservation
from .reporting.report_generator import ReportGenerator
from .sports import get_sport_analyzer
from .teams.classifier import ColorKMeansTeamClassifier
from .tracking.ball_tracker import BallTracker
from .tracking.identity import PlayerIdentityManager
from .tracking.player_tracker import PlayerTracker
from .video.processor import VideoInfo, VideoProcessor
from .visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)
_PROGRESS_EVERY = 200


class AnalysisPipeline:
    def __init__(self, config: Config) -> None:
        config.validate()
        self.cfg = config
        self.device = resolve_device(config.detection.device)
        self.sport = get_sport_analyzer(config)

    # ------------------------------------------------------------------ run
    def run(self) -> AnalysisResult:
        cfg, started = self.cfg, time.time()
        logger.info("Device: %s | sport: %s", self.device, cfg.sport)
        video = VideoProcessor(cfg.paths.input_video, cfg.video.fallback_fps, cfg.video.max_consecutive_read_failures)
        info = video.open()
        field = self.sport.create_field()
        mapper = FieldMapper(field, self._build_calibration(field, info))
        if not mapper.is_calibrated:
            logger.warning("No calibration: running detection/tracking/teams only (field statistics unavailable). "
                           "Create one with scripts/select_calibration_points.py")

        frames, relinks, lost, classifier = self._perception_pass(video, info, mapper)
        result = run_analysis(frames, info, cfg, self.sport, field, mapper.is_calibrated, classifier, relinks, lost)

        self._write_reports(result)
        if cfg.visualization.render_video:
            self._render_pass(video, info, result, mapper)
        self._print_summary(result, time.time() - started)
        return result

    # ----------------------------------------------------------- calibration
    def _build_calibration(self, field, info: VideoInfo) -> CalibrationProvider:
        fcfg = self.cfg.field
        if not fcfg.calibration_file:
            return NullCalibration()
        try:
            return ManualCalibration.from_json(fcfg.calibration_file, field, (info.width, info.height), fcfg.max_reprojection_error_m)
        except CalibrationError:
            logger.error("Invalid calibration file '%s'.", fcfg.calibration_file)
            raise

    # ------------------------------------------------------------ pass 1
    def _perception_pass(self, video: VideoProcessor, info: VideoInfo, mapper: FieldMapper):
        cfg = self.cfg
        stride = cfg.video.analysis_stride
        player_detector = PlayerDetector(cfg.detection, cfg.video, self.device)
        ball_detector = BallDetector(cfg.detection, self.device)
        tracker = PlayerTracker(cfg, info.fps)
        identity = PlayerIdentityManager(cfg.tracking)
        ball_tracker = BallTracker(cfg.ball, (info.width, info.height))
        classifier = ColorKMeansTeamClassifier(cfg.team)

        frames: list[FrameData] = []
        logger.info("Processing started (%d frames, stride %d, batch %d).", info.frame_count, stride, cfg.video.batch_size)
        if stride > 1:
            logger.warning("analysis_stride=%d: overlays hold the last analysed frame; the output keeps all frames.", stride)
        for batch in video.iter_batches(cfg.video.batch_size, stride, cfg.video.max_frames):
            indices = [i for i, _ in batch]
            images = [f for _, f in batch]
            stamps = [i / info.fps for i in indices]
            player_batches = player_detector.detect_batch(images, indices, stamps)
            ball_batches = ball_detector.detect_batch(images, indices, stamps)
            for idx, image, ts, dets, ball_cands in zip(indices, images, stamps, player_batches, ball_batches):
                fd = FrameData(frame=idx, timestamp=ts)
                fd.players = self._observe_players(dets, image, idx, ts, tracker, identity, classifier, mapper)
                fd.ball = self._observe_ball(ball_tracker, ball_cands, idx, ts, mapper)
                frames.append(fd)
                if len(frames) % _PROGRESS_EVERY == 0:
                    total = f"/{info.frame_count // stride}" if info.frame_count else ""
                    logger.info("Analysed %d%s frames", len(frames), total)
        if not frames:
            raise VideoReadError("No frames could be read from the video.")
        logger.info("Pass 1 complete: %d frames, %d identity re-links, %d ball-lost events.",
                    len(frames), identity.relink_count, ball_tracker.lost_events)
        return frames, identity.relink_count, ball_tracker.lost_events, classifier

    def _observe_players(self, dets: list[Detection], image, idx, ts, tracker, identity, classifier, mapper) -> list[PlayerObservation]:
        tracked = tracker.update(dets, image)
        if not tracked:
            return []
        sampled = (idx // self.cfg.video.analysis_stride) % self.cfg.team.feature_every_n_frames == 0
        features = [
            classifier.extract(image, d.bbox) if (d.role != DetectionRole.REFEREE and (sampled or not identity.knows(d.track_id))) else None
            for d in tracked
        ]
        player_ids = identity.assign(tracked, features, ts)
        ground = np.array([d.ground_point for d in tracked])
        field_pts = mapper.to_field(ground, idx, image)
        margin = self.cfg.field.pitch_margin_m
        observations = []
        for k, (det, pid, feat) in enumerate(zip(tracked, player_ids, features)):
            fxy = None if field_pts is None or np.isnan(field_pts[k]).any() else (float(field_pts[k, 0]), float(field_pts[k, 1]))
            observations.append(PlayerObservation(
                player_id=pid, tracker_id=int(det.track_id), frame=idx, timestamp=ts, bbox=det.bbox,
                confidence=det.confidence, role=det.role, ground_image=det.ground_point, jersey_feature=feat,
                field_xy=fxy, on_pitch=mapper.on_pitch(fxy, margin)))
        return observations

    def _observe_ball(self, ball_tracker: BallTracker, candidates, idx, ts, mapper) -> Optional[BallState]:
        state = ball_tracker.update(candidates, idx, ts)
        if state is None:
            return None
        mapped = mapper.to_field([state.ground_point], idx)
        if mapped is not None and np.isfinite(mapped).all():
            xy = (float(mapped[0, 0]), float(mapped[0, 1]))
            if mapper.on_pitch(xy, self.cfg.field.ball_margin_m):
                state.field_xy = xy
        return state

    # ------------------------------------------------------------ outputs
    def _write_reports(self, r: AnalysisResult) -> None:
        paths, gen = self.cfg.paths, ReportGenerator(self.cfg)
        rows = gen.write_player_tracks(r.frames, paths.player_tracks_path)
        gen.write_dataframe(r.player_stats, paths.player_statistics_path)
        gen.write_dataframe(r.team_stats, paths.team_statistics_path)
        gen.write_event_log(r.events, paths.event_log_path)
        gen.write_match_report(gen.build_report(r), paths.match_report_path)
        logger.info("Reports written to %s (%d track rows).", paths.output_dir, rows)
        if self.cfg.heatmap.enabled and r.calibrated:
            HeatmapGenerator(self.cfg, r.field, r.effective_fps).generate_all(r.tracks, paths.heatmap_path)

    def _render_pass(self, video: VideoProcessor, info: VideoInfo, r: AnalysisResult, mapper: FieldMapper) -> None:
        vcfg = self.cfg.video
        size = _output_size(info, vcfg.output_width, vcfg.output_height)
        video.open_writer(self.cfg.paths.output_video_path, size)
        visualizer = Visualizer(self.cfg, r.frames, r.tracks, r.events, r.possession.frame_states, r.spells, r.field,
                                mapper, (info.width, info.height), vcfg.analysis_stride)
        written = 0
        try:
            for idx, frame in video.iter_frames(stride=1, max_frames=vcfg.max_frames, fill_missing=True):
                video.write(visualizer.render(frame, idx))
                written += 1
        finally:
            video.close_writer()
        logger.info("Rendered %d frames (input FPS %.3f preserved).", written, info.fps)

    @staticmethod
    def _print_summary(r: AnalysisResult, elapsed: float) -> None:
        logger.info("Processing completed in %.1fs.", elapsed)
        print("\n" + "=" * 60 + "\nAI SPORTS PERFORMANCE ANALYZER - SUMMARY\n" + "=" * 60)
        print(f"Video: {r.video.width}x{r.video.height} @ {r.video.fps:.2f} FPS, {r.video.duration_s:.1f}s")
        print(f"Tracked identities: {len(r.tracks)} | events: {len(r.events)} | calibrated: {r.calibrated}")
        print(f"Ball detection coverage: {r.quality.ball_coverage:.0%}")
        if len(r.team_stats):
            print(r.team_stats.to_string(index=False))
        for warning in r.warnings:
            print(f"WARNING: {warning}")
        print("=" * 60)


def _output_size(info: VideoInfo, width: Optional[int], height: Optional[int]) -> tuple[int, int]:
    """Preserve the input resolution unless the user configured otherwise."""
    if width and height:
        return width, height
    if width:
        return width, max(1, round(width * info.height / info.width))
    if height:
        return max(1, round(height * info.width / info.height)), height
    return info.width, info.height
