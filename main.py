#!/usr/bin/env python3
"""Command-line entry point.

    python main.py                                   # defaults: input/sports_input.mp4 -> output/
    python main.py --config config/example_config.json --debug
"""
from __future__ import annotations

import argparse
import logging
import sys

from sports_analyzer.config import Config
from sports_analyzer.exceptions import SportsAnalyzerError
from sports_analyzer.logging_utils import setup_logging
from sports_analyzer.pipeline import AnalysisPipeline

logger = logging.getLogger("main")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Sports Performance Analyzer")
    p.add_argument("--config", help="JSON config overriding the defaults")
    p.add_argument("--input", help="input video (default: input/sports_input.mp4)")
    p.add_argument("--output-dir", help="directory for all outputs (default: output)")
    p.add_argument("--calibration", help="calibration JSON (see scripts/select_calibration_points.py)")
    p.add_argument("--device", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--debug", action="store_true", help="draw debug information on the video")
    p.add_argument("--max-frames", type=int, help="process only the first N frames")
    p.add_argument("--no-video", action="store_true", help="skip rendering the annotated video")
    p.add_argument("--log-level", default=None)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config) if args.config else Config()
    if args.input:
        config.paths.input_video = args.input
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.calibration:
        config.field.calibration_file = args.calibration
    if args.device:
        config.detection.device = args.device
    if args.debug:
        config.visualization.debug_mode = True
    if args.max_frames:
        config.video.max_frames = args.max_frames
    if args.no_video:
        config.visualization.render_video = False
    if args.log_level:
        config.runtime.log_level = args.log_level
    config.validate()
    return config


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = build_config(args)
        setup_logging(config.runtime.log_level, config.runtime.log_file)
        AnalysisPipeline(config).run()
        return 0
    except SportsAnalyzerError as exc:
        setup_logging()
        logger.error("%s: %s", type(exc).__name__, exc)
        return 2
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
