from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class RerunConfig:
    app_name: str = "WinGaze"
    spawn: bool = True
    base_path: str = "metrics"
    rolling_window_seconds: float = 5.0
    history_window_seconds: float = 5.0
    history_clear_interval_seconds: float = 1.0
    history_max_points: int = 10
    metric_sample_rate_hz: float = 30.0
    velocity_range: Optional[Tuple[float, float]] = None
    mean_x_range: Optional[Tuple[float, float]] = None
    mean_y_range: Optional[Tuple[float, float]] = None
    range_padding: float = 0.1
    range_percentiles: Optional[Tuple[float, float]] = (5.0, 95.0)


@dataclass(frozen=True)
class AppConfig:
    window_title: str = "WinGaze"
    window_width: int = 220
    window_height: int = 500
    default_fps: int = 30
    eyetrack_offset_ms: int = -32123
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    mask_participant: bool = True
    rerun: RerunConfig = field(default_factory=RerunConfig)
