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


@dataclass
class TextAnnotation:
    start_ms: float
    end_ms: float
    speaker: str
    text: str
