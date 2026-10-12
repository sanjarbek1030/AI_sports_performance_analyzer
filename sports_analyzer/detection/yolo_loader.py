"""Loading Ultralytics YOLO models (lazily, cached, with clear errors)."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from ..exceptions import ModelLoadError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def load_yolo(model_path: str, device: str) -> Any:
    """Load a YOLO model once per (path, device); both detectors may share it."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    path = Path(model_path)
    bare_name = path.parent == Path(".")  # e.g. "yolov8m.pt": Ultralytics may auto-download it
    if not path.exists() and not bare_name:
        raise ModelLoadError(f"Model file not found: {model_path}")
    try:
        model = YOLO(model_path)
        model.to(device)
    except Exception as exc:  # Ultralytics raises many types (download, torch, cuda)
        raise ModelLoadError(
            f"Failed to load model '{model_path}' on '{device}': {exc}. "
            "If you are offline, download the weights manually and point detection.player_model to them."
        ) from exc
    logger.info("Model loaded: %s on %s (%d classes)", model_path, device, len(model.names))
    return model


def match_class_ids(model_names: dict[int, str], wanted: Iterable[str]) -> list[int]:
    """Return class ids whose (lower-cased) names are in ``wanted``."""
    wanted_set = {w.lower() for w in wanted}
    return sorted(int(i) for i, name in model_names.items() if str(name).lower() in wanted_set)
