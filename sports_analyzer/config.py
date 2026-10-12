"""Central configuration.

Every tunable value lives here (no magic numbers elsewhere). Settings are
grouped in small dataclasses; field names match the spec's constants in
lower case (``INPUT_VIDEO`` -> ``paths.input_video``, ``SPRINT_THRESHOLD`` ->
``movement.sprint_threshold`` ...). Override any value from a JSON file:

    cfg = Config.load("config/example_config.json")

Units: speeds in km/h, distances in metres, durations in seconds unless a
name says otherwise.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, Optional

from .exceptions import ConfigError


def resolve_device(requested: str = "auto") -> str:
    """``DEVICE = "cuda" if available else "cpu"``; explicit values pass through."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class PathsConfig:
    input_video: str = "input/sports_input.mp4"
    output_dir: str = "output"
    # Blank -> derived from output_dir with the spec's file names.
    output_video: str = ""
    player_statistics_csv: str = ""
    team_statistics_csv: str = ""
    event_log_csv: str = ""
    player_tracks_csv: str = ""
    match_report_json: str = ""
    heatmap_dir: str = ""

    def _resolve(self, explicit: str, name: str) -> Path:
        return Path(explicit) if explicit else Path(self.output_dir) / name

    @property
    def output_video_path(self) -> Path:
        return self._resolve(self.output_video, "sports_analysis.mp4")

    @property
    def player_statistics_path(self) -> Path:
        return self._resolve(self.player_statistics_csv, "player_statistics.csv")

    @property
    def team_statistics_path(self) -> Path:
        return self._resolve(self.team_statistics_csv, "team_statistics.csv")

    @property
    def event_log_path(self) -> Path:
        return self._resolve(self.event_log_csv, "event_log.csv")

    @property
    def player_tracks_path(self) -> Path:
        return self._resolve(self.player_tracks_csv, "player_tracks.csv")

    @property
    def match_report_path(self) -> Path:
        return self._resolve(self.match_report_json, "match_report.json")

    @property
    def heatmap_path(self) -> Path:
        return self._resolve(self.heatmap_dir, "heatmaps")


@dataclass
class VideoConfig:
    process_width: Optional[int] = None    # detector input width (None = native)
    process_height: Optional[int] = None   # derived from aspect if only width given
    output_width: Optional[int] = None     # None = preserve original resolution
    output_height: Optional[int] = None
    analysis_stride: int = 1               # analyse every Nth frame (output keeps all frames)
    batch_size: int = 4                    # frames per detector call
    max_frames: Optional[int] = None       # stop early (debugging)
    fallback_fps: Optional[float] = None   # only used if the file reports no FPS
    max_consecutive_read_failures: int = 10


@dataclass
class DetectionConfig:
    player_model: str = "yolov8m.pt"
    ball_model: Optional[str] = None       # None -> reuse player_model
    confidence_threshold: float = 0.35     # "high" threshold used by the tracker
    low_confidence_threshold: float = 0.10 # detector floor (ByteTrack needs low-conf boxes)
    iou_threshold: float = 0.5
    device: str = "auto"
    half_precision: bool = True            # only applied on CUDA
    player_imgsz: int = 1280
    ball_imgsz: int = 1280
    ball_confidence_threshold: float = 0.15
    max_ball_candidates: int = 5
    ball_max_size_frac: float = 0.04       # reject "balls" wider than 4% of frame width
    ball_min_size_px: float = 3.0
    min_player_box_px: float = 12.0
    player_class_names: list[str] = dc_field(default_factory=lambda: ["person", "player"])
    goalkeeper_class_names: list[str] = dc_field(default_factory=lambda: ["goalkeeper"])
    referee_class_names: list[str] = dc_field(default_factory=lambda: ["referee"])
    ball_class_names: list[str] = dc_field(default_factory=lambda: ["sports ball", "ball"])


@dataclass
class TrackingConfig:
    tracker: str = "bytetrack"             # bytetrack | botsort | iou (built-in fallback)
    track_buffer_s: float = 2.0            # how long a lost track is kept alive
    match_threshold: float = 0.8           # max (1 - IoU) for association
    new_track_threshold: float = 0.5
    min_track_hits: int = 2                # built-in tracker confirmation
    relink_enabled: bool = True            # identity re-linking after long occlusion
    relink_max_gap_s: float = 3.0
    relink_base_heights: float = 1.0       # allowed jump, in player-box heights
    relink_heights_per_s: float = 3.0
    relink_max_color_distance: float = 18.0
    min_report_track_s: float = 3.0        # shorter identities are excluded from reports


@dataclass
class BallConfig:
    gate_min_frac_width: float = 0.02
    max_speed_width_per_s: float = 1.0     # fastest plausible ball, in frame widths/s
    process_accel_width_per_s2: float = 4.0
    measurement_noise_px: float = 3.0
    max_coast_s: float = 0.6               # predict through misses this long
    init_confirm_detections: int = 2
    init_min_confidence: float = 0.25
    coast_confidence_decay: float = 0.85
    smoothing_window_s: float = 0.24
    max_plausible_speed_ms: float = 45.0
    min_coverage_for_events: float = 0.15  # below this, ball-based stats are withheld


@dataclass
class TeamConfig:
    upper_body_top: float = 0.15
    upper_body_bottom: float = 0.55
    horizontal_margin: float = 0.20
    grass_hue_low: int = 35
    grass_hue_high: int = 85
    grass_min_saturation: int = 40
    min_jersey_pixels: int = 30
    lightness_weight: float = 0.6          # Lab L is lighting-sensitive: down-weight it
    feature_every_n_frames: int = 5
    fit_max_samples: int = 20000
    min_fit_samples: int = 40
    trim_fraction: float = 0.15
    trim_iterations: int = 2
    outlier_scale: float = 2.0
    min_vote_share: float = 0.6
    min_votes: int = 3
    random_seed: int = 0


@dataclass
class FieldConfig:
    field_length: float = 105.0            # metres (x axis)
    field_width: float = 68.0              # metres (y axis)
    goal_width: float = 7.32
    calibration_file: Optional[str] = None
    max_reprojection_error_m: float = 1.0
    pitch_margin_m: float = 2.0            # players farther outside are ignored
    ball_margin_m: float = 15.0
    team_a_attacks_positive_x: bool = True
    attack_direction_switch_s: Optional[float] = None  # half-time swap (video seconds)


@dataclass
class MovementConfig:
    smoothing_method: str = "savgol"       # savgol | moving_average | none
    smoothing_window_s: float = 0.8
    savgol_polyorder: int = 2
    max_gap_s: float = 0.6                 # interpolate gaps up to this long
    max_plausible_speed_kmh: float = 40.0
    max_plausible_accel_ms2: float = 9.0
    accel_smoothing_window_s: float = 0.4
    max_speed_median_window_s: float = 0.4
    walking_max_speed: float = 7.2
    jogging_max_speed: float = 14.4
    running_max_speed: float = 19.8        # above this = high-intensity
    sprint_threshold: float = 25.2
    min_sprint_duration: float = 1.0
    sprint_merge_gap_s: float = 0.3
    explosive_accel_ms2: float = 3.0
    rapid_decel_ms2: float = -3.0
    min_accel_duration_s: float = 0.4
    stop_speed_kmh: float = 2.0
    stop_prior_speed_kmh: float = 10.0
    direction_change_deg: float = 45.0
    direction_change_window_s: float = 0.5
    direction_change_min_speed_kmh: float = 7.2


@dataclass
class PossessionConfig:
    possession_radius: float = 2.0         # metres between ball and player's feet
    release_radius_factor: float = 1.5     # hysteresis when deciding the ball left
    confirm_s: float = 0.3                 # N consecutive frames -> possession
    release_s: float = 0.4
    max_missing_s: float = 0.6
    max_ball_coast_s: float = 0.2          # predicted ball positions older than this are ignored
    challenger_margin_m: float = 0.5


@dataclass
class EventConfig:
    pass_min_distance_m: float = 3.0
    pass_max_flight_s: float = 4.0
    pass_min_avg_speed_ms: float = 1.5
    pass_min_direction_cosine: float = 0.3
    shot_min_speed_ms: float = 14.0
    shot_speed_window_s: float = 0.6
    shot_max_distance_m: float = 35.0
    shot_angle_margin_deg: float = 10.0
    shot_outcome_window_s: float = 3.0
    post_margin_m: float = 0.5
    block_speed_drop: float = 0.4
    block_player_radius_m: float = 2.5
    block_progress_fraction: float = 0.6
    save_goal_distance_m: float = 10.0
    goal_confirm_s: float = 1.0
    goal_min_approach_speed_ms: float = 3.0
    goal_lost_near_line_m: float = 3.0
    goal_reentry_m: float = 1.0
    goal_cooldown_s: float = 20.0
    goal_attribution_window_s: float = 6.0
    assist_window_s: float = 5.0
    tackle_max_gap_s: float = 0.6
    tackle_max_player_distance_m: float = 3.0
    out_margin_m: float = 1.0
    out_min_duration_s: float = 0.4
    out_cooldown_s: float = 3.0
    confidence_cap: float = 0.85           # no inferred event is ever "certain"
    goal_confidence_cap: float = 0.70
    weak_event_confidence_cap: float = 0.60  # tackle / interception candidates


@dataclass
class TacticalConfig:
    enabled: bool = True
    sample_interval_s: float = 0.5
    min_players: int = 6
    compact_hull_ratio: float = 0.20
    stretched_hull_ratio: float = 0.32
    wide_ratio: float = 0.60
    narrow_ratio: float = 0.40
    shallow_ratio: float = 0.30
    deep_ratio: float = 0.45


@dataclass
class FormationConfig:
    enabled: bool = True
    window_s: float = 120.0
    min_obs_fraction: float = 0.5
    line_penalty: float = 15.0
    temperature: float = 20.0


@dataclass
class RoleConfig:
    enabled: bool = True
    min_track_seconds: float = 20.0


@dataclass
class HeatmapConfig:
    enabled: bool = True
    bin_m: float = 1.0
    sigma_bins: float = 2.0
    min_player_seconds: float = 10.0
    dpi: int = 120
    colormap: str = "inferno"


@dataclass
class VisualizationConfig:
    render_video: bool = True
    debug_mode: bool = False
    trail_length: int = 50
    ball_trail_length: int = 30
    show_trails: bool = True
    show_minimap: bool = True
    minimap_width_frac: float = 0.26
    minimap_show_ids: bool = True
    event_display_s: float = 3.0
    match_clock_offset_s: float = 0.0
    team_colors_bgr: list[list[int]] = dc_field(default_factory=lambda: [[255, 140, 0], [60, 60, 230]])
    team_names: list[str] = dc_field(default_factory=lambda: ["Team A", "Team B"])


@dataclass
class RuntimeConfig:
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class Config:
    sport: str = "football"
    paths: PathsConfig = dc_field(default_factory=PathsConfig)
    video: VideoConfig = dc_field(default_factory=VideoConfig)
    detection: DetectionConfig = dc_field(default_factory=DetectionConfig)
    tracking: TrackingConfig = dc_field(default_factory=TrackingConfig)
    ball: BallConfig = dc_field(default_factory=BallConfig)
    team: TeamConfig = dc_field(default_factory=TeamConfig)
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    movement: MovementConfig = dc_field(default_factory=MovementConfig)
    possession: PossessionConfig = dc_field(default_factory=PossessionConfig)
    events: EventConfig = dc_field(default_factory=EventConfig)
    tactical: TacticalConfig = dc_field(default_factory=TacticalConfig)
    formation: FormationConfig = dc_field(default_factory=FormationConfig)
    roles: RoleConfig = dc_field(default_factory=RoleConfig)
    heatmap: HeatmapConfig = dc_field(default_factory=HeatmapConfig)
    visualization: VisualizationConfig = dc_field(default_factory=VisualizationConfig)
    runtime: RuntimeConfig = dc_field(default_factory=RuntimeConfig)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_dict(cls, overrides: dict[str, Any]) -> "Config":
        config = cls()
        _apply_overrides(config, overrides)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                overrides = json.load(handle)
        except FileNotFoundError as exc:
            raise ConfigError(f"Config file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Config file is not valid JSON: {path}: {exc}") from exc
        return cls.from_dict(overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def debug_mode(self) -> bool:
        return self.visualization.debug_mode

    # ---- validation ---------------------------------------------------
    def validate(self) -> None:
        problems: list[str] = []
        if self.video.analysis_stride < 1:
            problems.append("video.analysis_stride must be >= 1")
        if self.video.batch_size < 1:
            problems.append("video.batch_size must be >= 1")
        det = self.detection
        if not 0 <= det.low_confidence_threshold <= det.confidence_threshold <= 1:
            problems.append("need 0 <= low_confidence_threshold <= confidence_threshold <= 1")
        if self.tracking.tracker not in {"bytetrack", "botsort", "iou"}:
            problems.append("tracking.tracker must be bytetrack, botsort or iou")
        if self.field.field_length <= 0 or self.field.field_width <= 0:
            problems.append("field dimensions must be positive")
        mv = self.movement
        if not (0 < mv.walking_max_speed < mv.jogging_max_speed < mv.running_max_speed < mv.sprint_threshold):
            problems.append("speed thresholds must satisfy walking < jogging < running < sprint")
        if mv.sprint_threshold >= mv.max_plausible_speed_kmh:
            problems.append("sprint_threshold must be below max_plausible_speed_kmh")
        if mv.smoothing_method not in {"savgol", "moving_average", "none"}:
            problems.append("movement.smoothing_method must be savgol, moving_average or none")
        if len(self.visualization.team_colors_bgr) < 2 or len(self.visualization.team_names) < 2:
            problems.append("visualization needs two team colours and two team names")
        if problems:
            raise ConfigError("Invalid configuration: " + "; ".join(problems))


def _apply_overrides(target: Any, overrides: dict[str, Any], prefix: str = "") -> None:
    """Recursively apply a dict onto nested dataclasses, rejecting unknown keys."""
    if not isinstance(overrides, dict):
        raise ConfigError(f"Expected an object at '{prefix or 'root'}'")
    for key, value in overrides.items():
        if key.startswith("_"):  # allow "_comment" keys in JSON files
            continue
        if not hasattr(target, key):
            raise ConfigError(f"Unknown config key: {prefix}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            _apply_overrides(current, value, f"{prefix}{key}.")
        else:
            setattr(target, key, value)
