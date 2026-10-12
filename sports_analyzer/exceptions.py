"""Project-specific exceptions so failures are explicit and catchable."""


class SportsAnalyzerError(Exception):
    """Base class for all errors raised by this project."""


class ConfigError(SportsAnalyzerError):
    """Invalid or inconsistent configuration."""


class VideoOpenError(SportsAnalyzerError):
    """Input video missing, corrupted, or using an unsupported codec."""


class VideoReadError(SportsAnalyzerError):
    """Frames could not be decoded."""


class VideoWriteError(SportsAnalyzerError):
    """The output video could not be created or written."""


class ModelLoadError(SportsAnalyzerError):
    """A detection model is missing or incompatible."""


class ModelInferenceError(SportsAnalyzerError):
    """A detection model failed while running."""


class CalibrationError(SportsAnalyzerError):
    """Field calibration is missing, degenerate, or inaccurate."""


class InsufficientDataError(SportsAnalyzerError):
    """Not enough detections/observations to compute a result."""
