import tempfile
from pathlib import Path

import cv2
import numpy as np

from helpers import raises
from sports_analyzer.config import Config
from sports_analyzer.exceptions import ConfigError, VideoOpenError, VideoWriteError
from sports_analyzer.video.processor import VideoProcessor


def make_video(path: Path, n=30, size=(320, 240), fps=25.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        writer.write(np.full((size[1], size[0], 3), i * 7 % 255, np.uint8))
    writer.release()


def test_video_info_and_frame_iteration():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v.mp4"
        make_video(path)
        vp = VideoProcessor(path)
        info = vp.open()
        assert (info.width, info.height, info.frame_count) == (320, 240, 30) and info.fps == 25.0
        assert abs(info.duration_s - 1.2) < 1e-9
        assert len(list(vp.iter_frames())) == 30 and [i for i, _ in vp.iter_frames(stride=10)] == [0, 10, 20]
        assert [len(b) for b in vp.iter_batches(4)] == [4] * 7 + [2]


def test_writer_preserves_fps_and_resolution():
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "a.mp4", Path(d) / "out" / "b.mp4"
        make_video(src, fps=30.0)
        vp = VideoProcessor(src)
        vp.open()
        vp.open_writer(dst)
        for _, frame in vp.iter_frames():
            vp.write(frame)
        vp.close_writer()
        out = VideoProcessor(dst).open()
        assert (out.width, out.height, out.frame_count, out.fps) == (320, 240, 30, 30.0)


def test_missing_and_corrupt_video_raise_clear_errors():
    with tempfile.TemporaryDirectory() as d:
        with raises(VideoOpenError):
            VideoProcessor(Path(d) / "nope.mp4").open()
        bad = Path(d) / "bad.mp4"
        bad.write_bytes(b"this is not a video")
        with raises(VideoOpenError):
            VideoProcessor(bad).open()


def test_unwritable_output_raises():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "a.mp4"
        make_video(src)
        vp = VideoProcessor(src)
        vp.open()
        blocker = Path(d) / "file"
        blocker.write_text("x")
        with raises(VideoWriteError):
            vp.open_writer(blocker / "sub" / "out.mp4")      # parent is a file


def test_config_validation_and_overrides():
    cfg = Config.from_dict({"movement": {"sprint_threshold": 27.0}, "video": {"analysis_stride": 2}})
    assert cfg.movement.sprint_threshold == 27.0 and cfg.video.analysis_stride == 2
    for bad in ({"video": {"nonexistent": 1}}, {"video": {"analysis_stride": 0}},
                {"movement": {"sprint_threshold": 5.0}}, {"tracking": {"tracker": "magic"}}):
        with raises(ConfigError):
            Config.from_dict(bad)
    assert Config().paths.match_report_path == Path("output/match_report.json")
