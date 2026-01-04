from dataclasses import dataclass


@dataclass
class SegmentTimes:
    start_ms: float
    end_ms: float


@dataclass
class TrackingMetrics:
    velocity: float
    mean_x: float
    mean_y: float
