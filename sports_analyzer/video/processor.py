"""VideoProcessor: opening, reading and writing video robustly.

Problem solved : OpenCV fails quietly (empty frames, FPS = 0, writers that
                 silently do not open). This class validates everything.
Receives       : an input path (and later an output path).
Produces       : ``VideoInfo`` metadata, frame iterators, a validated writer.
Interacts with: the pipeline calls it twice - once for the analysis pass, once
                 for the render pass - and the Visualizer's frames go to ``write``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

from ..exceptions import VideoOpenError, VideoReadError, VideoWriteError

logger = logging.getLogger(__name__)

_CODECS_BY_SUFFIX = {
    ".mp4": ("mp4v", "avc1"),
    ".mov": ("mp4v", "avc1"),
    ".avi": ("XVID", "MJPG"),
}
_DEFAULT_CODECS = ("mp4v", "XVID", "MJPG")


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int      # 0 when the container does not report it
    duration_s: float

    def to_dict(self) -> dict:
        return asdict(self)


class VideoProcessor:
    def __init__(
        self,
        input_path: str | Path,
        fallback_fps: Optional[float] = None,
        max_consecutive_failures: int = 10,
    ) -> None:
        self.input_path = Path(input_path)
        self.fallback_fps = fallback_fps
        self.max_consecutive_failures = max_consecutive_failures
        self.info: Optional[VideoInfo] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_size: Optional[tuple[int, int]] = None
        self.output_path: Optional[Path] = None

    # ------------------------------------------------------------------ open
    def open(self) -> VideoInfo:
        """Validate the video and read its properties."""
        if not self.input_path.is_file():
            raise VideoOpenError(f"Input video not found: {self.input_path}")
        cap = cv2.VideoCapture(str(self.input_path))
        try:
            if not cap.isOpened():
                raise VideoOpenError(
                    f"OpenCV cannot open {self.input_path}. The file may be corrupted or use an "
                    "unsupported codec; try re-encoding: ffmpeg -i in.mp4 -c:v libx264 out.mp4"
                )
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0:
                if self.fallback_fps is None:
                    raise VideoOpenError("Video reports no valid FPS; set video.fallback_fps.")
                logger.warning("Video reports invalid FPS; using fallback_fps=%.3f", self.fallback_fps)
                fps = float(self.fallback_fps)
            if width <= 0 or height <= 0:
                raise VideoOpenError("Video reports an invalid resolution.")
            ok, _ = cap.read()
            if not ok:
                raise VideoOpenError("The first frame cannot be decoded (unsupported codec or corrupted file).")
        finally:
            cap.release()
        if frame_count <= 0:
            logger.warning("Container does not report a frame count; progress/duration unknown.")
            frame_count = 0
        duration = frame_count / fps if frame_count else 0.0
        self.info = VideoInfo(str(self.input_path), width, height, fps, frame_count, duration)
        logger.info(
            "Video loaded: %s | %dx%d | %.3f FPS | %d frames | %.2f s",
            self.input_path, width, height, fps, frame_count, duration,
        )
        return self.info

    # ------------------------------------------------------------------ read
    def iter_frames(
        self, stride: int = 1, max_frames: Optional[int] = None, fill_missing: bool = False
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` for every ``stride``-th frame.

        Unreadable frames are skipped with a warning (or, with ``fill_missing``,
        replaced by the previous frame so the output keeps its frame count).
        Stops after too many consecutive failures.
        """
        if self.info is None:
            raise VideoReadError("open() must be called before iter_frames().")
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            raise VideoReadError(f"Cannot reopen {self.input_path}")
        known_count = self.info.frame_count
        idx, failures = 0, 0
        last_good: Optional[np.ndarray] = None
        try:
            while max_frames is None or idx < max_frames:
                frame: Optional[np.ndarray] = None
                if cap.grab():
                    if idx % stride == 0:
                        ok, decoded = cap.retrieve()
                        frame = decoded if ok else None
                        if frame is None:
                            logger.warning("Frame %d could not be decoded.", idx)
                    else:
                        failures = 0
                        idx += 1
                        continue
                elif known_count == 0 or idx >= known_count:
                    break  # normal end of stream
                else:
                    logger.warning("Read failed at frame %d of %d.", idx, known_count)
                    failures += 1
                    if failures >= self.max_consecutive_failures:
                        logger.error("Too many consecutive read failures; stopping at frame %d.", idx)
                        break

                if frame is not None:
                    failures = 0
                    last_good = frame
                    yield idx, frame
                elif fill_missing and last_good is not None and idx % stride == 0:
                    logger.warning("Frame %d unreadable: repeating previous frame in output.", idx)
                    yield idx, last_good.copy()
                idx += 1
        finally:
            cap.release()

    def iter_batches(
        self, batch_size: int, stride: int = 1, max_frames: Optional[int] = None
    ) -> Iterator[list[tuple[int, np.ndarray]]]:
        batch: list[tuple[int, np.ndarray]] = []
        for item in self.iter_frames(stride=stride, max_frames=max_frames):
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # ----------------------------------------------------------------- write
    def open_writer(
        self, output_path: str | Path, size: Optional[tuple[int, int]] = None,
        codecs: Optional[Sequence[str]] = None,
    ) -> None:
        """Create the output writer at the input FPS (and size unless overridden)."""
        if self.info is None:
            raise VideoWriteError("open() must be called before open_writer().")
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VideoWriteError(f"Cannot create output directory {path.parent}: {exc}") from exc
        size = size or (self.info.width, self.info.height)
        if size != (self.info.width, self.info.height):
            logger.warning("Output resolution %s differs from input %dx%d (explicitly configured).",
                           size, self.info.width, self.info.height)
        candidates = codecs or _CODECS_BY_SUFFIX.get(path.suffix.lower(), _DEFAULT_CODECS)
        for codec in candidates:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), self.info.fps, size)
            if writer.isOpened():
                self._writer, self._writer_size, self.output_path = writer, size, path
                logger.info("Output writer ready: %s | codec=%s | %dx%d @ %.3f FPS",
                            path, codec, size[0], size[1], self.info.fps)
                return
            writer.release()
        raise VideoWriteError(
            f"Could not open an output writer for {path} (tried codecs {list(candidates)}). "
            "Check that the directory is writable and your OpenCV build has video encoding."
        )

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None or self._writer_size is None:
            raise VideoWriteError("open_writer() must be called before write().")
        if (frame.shape[1], frame.shape[0]) != self._writer_size:
            frame = cv2.resize(frame, self._writer_size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)

    def close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Output video written: %s", self.output_path)

    def __enter__(self) -> "VideoProcessor":
        return self

    def __exit__(self, *exc) -> None:
        self.close_writer()
