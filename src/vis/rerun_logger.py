from collections import deque
import math
from typing import Deque, Dict, List, Optional, Tuple

try:
    import rerun as rr
except ImportError as exc:
    raise ImportError(
        "rerun module is not installed. Please install it using 'pip install rerun-sdk'."
    ) from exc

try:
    from rerun.components import MarkerShape
except ImportError:
    MarkerShape = None

try:
    import rerun.blueprint as rrb
except ImportError:
    rrb = None

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "cv2 module is not installed. Please install it using 'pip install opencv-python'."
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "numpy module is not installed. Please install it using 'pip install numpy'."
    ) from exc

from core.data_types import TextAnnotation, TrackingMetrics
from utils.aoi_probability import compute_aoi_probability_series
from vis.layouts import create_default_rrb

METRIC_SERIES_COLORS = {
    "velocity": [162, 109, 255],
    "mean_x": [84, 200, 120],
    "mean_y": [235, 190, 80],
}

AOI_SERIES_COLORS = {
    "Robot": [210, 100, 255],
    "Task": [70, 200, 200],
}


class RerunLogger:
    def __init__(
        self,
        app_name: str,
        spawn: bool,
        base_path: str,
        rolling_window_seconds: float = 5.0,
        history_window_seconds: float = 300.0,
        history_clear_interval_seconds: float = 1.0,
        history_max_points: int = 0,
        metric_sample_rate_hz: float = 30.0,
        velocity_range: Optional[Tuple[float, float]] = None,
        mean_x_range: Optional[Tuple[float, float]] = None,
        mean_y_range: Optional[Tuple[float, float]] = None,
        range_padding: float = 0.1,
        range_percentiles: Optional[Tuple[float, float]] = (5.0, 95.0),
    ) -> None:
        self._metrics_root = base_path.strip("/") or "metrics"
        self.base_path = self._metrics_root
        self._segment_index = 0
        self.video_main_path = "video/main"
        self.video_eye_path = "video/eyetrack"
        self._annotation_path = "annotations/speech"
        self._aoi_root = "aoi"
        self._aoi_path = f"{self._aoi_root}/segment_{self._segment_index}/probability"
        self._app_name = app_name
        self._spawn_once = spawn
        self._rolling_window_ms = max(0, int(rolling_window_seconds * 1000))
        self._history_window_ms = max(0, int(history_window_seconds * 1000))
        self._history_clear_interval_ms = max(0, int(history_clear_interval_seconds * 1000))
        self._last_history_clear_ms: Optional[int] = None
        self._window_samples = None
        self._metric_buffer: Deque[TrackingMetrics] = deque()
        self._history_max_points = max(0, int(history_max_points))
        self._metric_sample_rate_hz = float(metric_sample_rate_hz)
        self._metric_log_interval_ms = (
            int(1000.0 / self._metric_sample_rate_hz)
            if self._metric_sample_rate_hz > 0
            else 0
        )
        self._last_metric_log_ms: Optional[int] = None
        self._time_offset_ms = 0
        maxlen = self._history_max_points or None
        self._history_buffer: Deque[Tuple[int, TrackingMetrics]] = deque(maxlen=maxlen)
        self._range_padding = max(0.0, range_padding)
        self._range_percentiles = range_percentiles
        self._fixed_ranges: Dict[str, Optional[Tuple[float, float]]] = {
            "velocity": velocity_range,
            "mean_x": mean_x_range,
            "mean_y": mean_y_range,
        }
        self._range_cache: Dict[str, Tuple[float, float]] = {}
        self._annotation_segments: List[TextAnnotation] = []
        self._annotation_index = 0
        self._active_annotations: Dict[str, TextAnnotation] = {}
        self._annotation_speaker_order: List[str] = []
        self._last_annotation_text: Optional[str] = None
        self._speech_onsets: List[Tuple[int, str]] = []
        self._speech_onset_index = 0
        self._speech_marker_paths: Dict[str, str] = {}
        self._aoi_series_labels: List[str] = []
        self._aoi_series_values: Optional[np.ndarray] = None
        self._aoi_bin_ms: float = 0.0
        self._aoi_series_end_ms: float = 0.0
        self._aoi_last_bin: Optional[int] = None
        self._aoi_label_index: Dict[str, int] = {}
        self._supports_text_document = hasattr(rr, "TextDocument")
        self._supports_text_log = hasattr(rr, "TextLog")
        self._supports_text_document_view = hasattr(rrb, "TextDocumentView") if rrb else False
        self._supports_text_log_view = hasattr(rrb, "TextLogView") if rrb else False
        if self._supports_text_log and self._supports_text_log_view:
            self._annotation_view_kind = "log"
        elif self._supports_text_document and self._supports_text_document_view:
            self._annotation_view_kind = "document"
        else:
            self._annotation_view_kind = None
        self._setup()

    def _setup(self) -> None:
        try:
            rr.init(self._app_name, spawn=self._spawn_once)
        except TypeError:
            rr.init(self._app_name)
            if self._spawn_once:
                rr.spawn()
        self._spawn_once = False
        self._send_blueprint()

    def log_frame(
        self,
        _frame_id: int,
        time_ms: int,
        metrics: TrackingMetrics,
        frame_bgr,
        eye_frame_bgr=None,
        fps: float = 0.0,
    ) -> None:
        log_time_ms = max(0, int(time_ms - self._time_offset_ms))
        should_log_metrics = self._should_log_metrics(log_time_ms)
        if self._history_window_ms:
            if should_log_metrics:
                self._store_history(log_time_ms, metrics)

        if should_log_metrics and self._rolling_window_ms:
            self._ensure_window_samples(fps)
            self._metric_buffer.append(metrics)
            self._update_ranges_if_ready()

        did_relog = self._maybe_clear_history(log_time_ms) if should_log_metrics else False
        has_annotations = bool(self._annotation_segments or self._active_annotations)
        has_aoi = self._aoi_series_values is not None
        if should_log_metrics or has_annotations or has_aoi:
            rr.set_time_seconds("time", log_time_ms / 1000.0)
            if should_log_metrics and not did_relog:
                if self._rolling_window_ms:
                    self._log_clamped_metrics(metrics)
                else:
                    self._log_metrics(metrics)
            if has_annotations:
                self._log_speech_annotations(log_time_ms)
            if has_aoi:
                self._log_aoi_at_time(log_time_ms)
            if self._speech_onsets:
                self._log_speech_onset_markers(log_time_ms)

        self._log_image(self.video_main_path, frame_bgr)
        if eye_frame_bgr is not None:
            self._log_image(self.video_eye_path, eye_frame_bgr)

    def reset(self) -> None:
        self._reset_state()
        rr.set_time_seconds("time", 0.0)
        self._clear_path(self.base_path, force=True)
        self._clear_path(self.base_path, timeless=True, force=True)
        self._clear_path(self.video_main_path, timeless=True, force=True)
        self._clear_path(self.video_eye_path, timeless=True, force=True)
        self._clear_path(self._annotation_path, force=True)
        self._clear_path(self._annotation_path, timeless=True, force=True)
        self._clear_path(self._aoi_path, force=True)
        self._clear_path(self._aoi_path, timeless=True, force=True)

    def _reset_state(self) -> None:
        self._metric_buffer.clear()
        self._history_buffer.clear()
        self._last_history_clear_ms = None
        self._last_metric_log_ms = None
        self._time_offset_ms = 0
        self._range_cache.clear()
        self._annotation_segments = []
        self._annotation_index = 0
        self._active_annotations.clear()
        self._annotation_speaker_order = []
        self._last_annotation_text = None
        self._speech_onsets = []
        self._speech_onset_index = 0
        self._speech_marker_paths = {}
        self._aoi_series_labels = []
        self._aoi_series_values = None
        self._aoi_bin_ms = 0.0
        self._aoi_series_end_ms = 0.0
        self._aoi_last_bin = None
        self._aoi_label_index = {}

    def start_segment(self, start_time_ms: int) -> None:
        self._segment_index += 1
        self.base_path = f"{self._metrics_root}/segment_{self._segment_index}"
        self._aoi_path = f"{self._aoi_root}/segment_{self._segment_index}/probability"
        self._reset_state()
        self._time_offset_ms = max(0, int(start_time_ms))
        self._send_blueprint()
        self._log_metric_labels()
        self._clear_path(self._annotation_path, force=True)
        self._clear_path(self._annotation_path, timeless=True, force=True)
        self._clear_path(self._aoi_path, force=True)
        self._clear_path(self._aoi_path, timeless=True, force=True)

    def _send_blueprint(self) -> None:
        if not hasattr(rr, "send_blueprint"):
            return
        if self._history_window_ms:
            window_seconds = self._history_window_ms / 1000.0
        else:
            window_seconds = self._rolling_window_ms / 1000.0 if self._rolling_window_ms else None
        rr.send_blueprint(
            create_default_rrb(
                window_seconds=window_seconds,
                metrics_root=self.base_path,
                annotation_path=self._annotation_path,
                aoi_path=self._aoi_path,
                annotation_view=self._annotation_view_kind,
            )
        )

    def set_speech_annotations(
        self,
        annotations: List[TextAnnotation],
        *,
        speaker_order: Optional[List[str]] = None,
        segment_start_ms: int = 0,
        segment_end_ms: Optional[int] = None,
    ) -> None:
        self._annotation_segments = []
        self._annotation_index = 0
        self._active_annotations.clear()
        self._last_annotation_text = None
        if speaker_order:
            self._annotation_speaker_order = list(speaker_order)
        else:
            self._annotation_speaker_order = []
            for annotation in annotations:
                if annotation.speaker not in self._annotation_speaker_order:
                    self._annotation_speaker_order.append(annotation.speaker)
        segment_start_ms = int(segment_start_ms)
        segment_end_ms_int = int(segment_end_ms) if segment_end_ms is not None else None
        max_rel_end = (
            max(0, segment_end_ms_int - segment_start_ms)
            if segment_end_ms_int is not None
            else None
        )
        for annotation in annotations:
            if segment_end_ms_int is not None and annotation.start_ms > segment_end_ms_int:
                continue
            if annotation.end_ms < segment_start_ms:
                continue
            rel_start = int(annotation.start_ms - segment_start_ms)
            rel_end = int(annotation.end_ms - segment_start_ms)
            if rel_end < 0:
                continue
            rel_start = max(0, rel_start)
            if max_rel_end is not None:
                rel_end = min(rel_end, max_rel_end)
            if rel_end < rel_start:
                continue
            self._annotation_segments.append(
                TextAnnotation(
                    start_ms=rel_start,
                    end_ms=rel_end,
                    speaker=annotation.speaker,
                    text=annotation.text,
                )
            )
        self._annotation_segments.sort(key=lambda ann: ann.start_ms)
        self._speech_onsets = []
        for annotation in self._annotation_segments:
            speaker_norm = annotation.speaker.strip().lower()
            if speaker_norm not in ("robot", "participant"):
                continue
            label = "Robot" if speaker_norm == "robot" else "Participant"
            self._speech_onsets.append((int(annotation.start_ms), label))
        self._speech_onsets.sort(key=lambda item: item[0])
        self._speech_onset_index = 0

    def set_aoi_probabilities(
        self,
        fixations: List[Tuple[float, float, str]],
        *,
        bin_ms: float,
        smooth_kind: str,
        smooth_ms: float,
        include_none: bool = False,
        segment_start_ms: int = 0,
        segment_end_ms: Optional[int] = None,
        eyetrack_offset_ms: int = 0,
    ) -> None:
        self._clear_path(self._aoi_path, timeless=True, force=True)
        if not fixations:
            return
        segment_start_ms = int(segment_start_ms)
        segment_end_ms_int = int(segment_end_ms) if segment_end_ms is not None else None
        max_rel_end = (
            max(0, segment_end_ms_int - segment_start_ms)
            if segment_end_ms_int is not None
            else None
        )
        adjusted = []
        for start_ms, end_ms, label in fixations:
            main_start = float(start_ms) - float(eyetrack_offset_ms)
            main_end = float(end_ms) - float(eyetrack_offset_ms)
            if segment_end_ms_int is not None and main_start > segment_end_ms_int:
                continue
            if main_end < segment_start_ms:
                continue
            rel_start = main_start - segment_start_ms
            rel_end = main_end - segment_start_ms
            if max_rel_end is not None:
                rel_start = max(0.0, min(rel_start, max_rel_end))
                rel_end = max(0.0, min(rel_end, max_rel_end))
            if rel_end <= rel_start:
                continue
            adjusted.append((rel_start, rel_end, label))

        if not adjusted:
            return

        if max_rel_end is not None and max_rel_end > 0:
            series_end_ms = float(max_rel_end)
        else:
            series_end_ms = max(end for _, end, _ in adjusted)
        if series_end_ms <= 0:
            return

        time_ms, labels, values = compute_aoi_probability_series(
            adjusted,
            start_ms=0.0,
            end_ms=series_end_ms,
            bin_ms=float(bin_ms),
            include_none=include_none,
            smooth_kind=smooth_kind,
            smooth_ms=float(smooth_ms),
        )
        self._aoi_series_labels = labels
        self._aoi_series_values = values
        self._aoi_bin_ms = float(bin_ms)
        self._aoi_series_end_ms = float(len(time_ms)) * float(bin_ms)
        self._aoi_last_bin = None
        self._aoi_label_index = {label: idx for idx, label in enumerate(labels)}
        self._log_aoi_labels()
        self._speech_onset_index = 0

    def _log_image(self, path: str, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rr.log(path, rr.Image(rgb), static=True)

    def _log_metrics(self, metrics: TrackingMetrics) -> None:
        rr.log(f"{self.base_path}/velocity", rr.Scalar(metrics.velocity))
        if self._is_finite(metrics.mean_x):
            rr.log(f"{self.base_path}/mean/x", rr.Scalar(metrics.mean_x))
        if self._is_finite(metrics.mean_y):
            rr.log(f"{self.base_path}/mean/y", rr.Scalar(metrics.mean_y))

    def _log_speech_annotations(self, time_ms: int) -> None:
        if not self._annotation_segments and not self._active_annotations:
            return
        while (
            self._annotation_index < len(self._annotation_segments)
            and self._annotation_segments[self._annotation_index].start_ms <= time_ms
        ):
            annotation = self._annotation_segments[self._annotation_index]
            if annotation.end_ms >= time_ms:
                self._active_annotations[annotation.speaker] = annotation
            self._annotation_index += 1
        expired = [
            speaker
            for speaker, annotation in self._active_annotations.items()
            if annotation.end_ms < time_ms
        ]
        for speaker in expired:
            del self._active_annotations[speaker]
        text = self._build_annotation_text()
        if text != self._last_annotation_text:
            self._emit_annotation(text)
            self._last_annotation_text = text

    def _build_annotation_text(self) -> str:
        if not self._active_annotations:
            return ""
        lines = []
        used = set()
        for speaker in self._annotation_speaker_order:
            annotation = self._active_annotations.get(speaker)
            if annotation is None:
                continue
            lines.append(f"{speaker}: {annotation.text}")
            used.add(speaker)
        extras = [
            annotation
            for speaker, annotation in self._active_annotations.items()
            if speaker not in used
        ]
        for annotation in sorted(extras, key=lambda ann: (ann.start_ms, ann.speaker)):
            lines.append(f"{annotation.speaker}: {annotation.text}")
        return "\n".join(lines)

    def _emit_annotation(self, text: str) -> None:
        if self._annotation_view_kind == "document" and self._supports_text_document:
            rr.log(self._annotation_path, rr.TextDocument(text))
            return
        if self._annotation_view_kind == "log" and self._supports_text_log:
            rr.log(self._annotation_path, rr.TextLog(text))

    def _log_metric_labels(self) -> None:
        try:
            rr.log(
                f"{self.base_path}/velocity",
                rr.SeriesLine(
                    name="Velocity",
                    color=METRIC_SERIES_COLORS.get("velocity"),
                ),
                timeless=True,
            )
            rr.log(
                f"{self.base_path}/mean/x",
                rr.SeriesLine(
                    name="MeanX",
                    color=METRIC_SERIES_COLORS.get("mean_x"),
                ),
                timeless=True,
            )
            rr.log(
                f"{self.base_path}/mean/y",
                rr.SeriesLine(
                    name="MeanY",
                    color=METRIC_SERIES_COLORS.get("mean_y"),
                ),
                timeless=True,
            )
        except Exception:
            return

    def _log_aoi_labels(self) -> None:
        if not self._aoi_series_labels:
            return
        for label in self._aoi_series_labels:
            label_str = str(label)
            try:
                rr.log(
                    f"{self._aoi_path}/{label_str}",
                    rr.SeriesLine(
                        name=label_str,
                        color=AOI_SERIES_COLORS.get(label_str),
                    ),
                    timeless=True,
                )
            except Exception:
                continue

    def _log_speech_onset_markers(self, time_ms: int) -> None:
        if not self._speech_onsets:
            return
        if self._aoi_series_values is None or self._aoi_bin_ms <= 0:
            return
        marker_labels = [
            label for label in ("Robot", "Task") if label in self._aoi_label_index
        ]
        if not marker_labels:
            return
        for label in marker_labels:
            if label in self._speech_marker_paths:
                continue
            path = f"{self._aoi_path}/speech_markers/{label}"
            self._speech_marker_paths[label] = path
            try:
                marker_color = AOI_SERIES_COLORS.get(label)
                rr.log(
                    path,
                    rr.SeriesLine(
                        width=0.0,
                        name="",
                        color=marker_color,
                    ),
                    timeless=True,
                )
                marker = MarkerShape.Circle if MarkerShape is not None else None
                rr.log(
                    path,
                    rr.SeriesPoint(
                        name="",
                        marker=marker,
                        marker_size=4.0,
                        color=marker_color,
                    ),
                    timeless=True,
                )
            except Exception:
                continue

        while self._speech_onset_index < len(self._speech_onsets):
            onset_ms, speaker = self._speech_onsets[self._speech_onset_index]
            if onset_ms > time_ms:
                break
            rr.set_time_seconds("time", float(onset_ms) / 1000.0)
            for label in marker_labels:
                path = self._speech_marker_paths.get(label)
                if path is None:
                    continue
                y_value = self._aoi_value_at_time(label, onset_ms)
                rr.log(path, rr.Scalar(float(y_value)))
            self._speech_onset_index += 1
        rr.set_time_seconds("time", float(time_ms) / 1000.0)

    def _aoi_value_at_time(self, label: str, time_ms: int) -> float:
        if self._aoi_series_values is None or self._aoi_bin_ms <= 0:
            return 0.0
        idx = int(time_ms // self._aoi_bin_ms)
        idx = max(0, min(idx, self._aoi_series_values.shape[0] - 1))
        col = self._aoi_label_index.get(label, 0)
        return float(self._aoi_series_values[idx, col])

    def _log_aoi_at_time(self, time_ms: int) -> None:
        if not self._aoi_series_labels or self._aoi_series_values is None:
            return
        if self._aoi_bin_ms <= 0:
            return
        if time_ms < 0 or time_ms >= self._aoi_series_end_ms:
            return
        idx = int(time_ms // self._aoi_bin_ms)
        if idx < 0 or idx >= self._aoi_series_values.shape[0]:
            return
        if idx == self._aoi_last_bin:
            return
        self._aoi_last_bin = idx
        row = self._aoi_series_values[idx]
        for col_idx, label in enumerate(self._aoi_series_labels):
            rr.log(f"{self._aoi_path}/{label}", rr.Scalar(float(row[col_idx])))

    def _store_history(self, time_ms: int, metrics: TrackingMetrics) -> None:
        self._history_buffer.append((time_ms, metrics))
        if self._history_window_ms:
            cutoff = time_ms - self._history_window_ms
            while self._history_buffer and self._history_buffer[0][0] < cutoff:
                self._history_buffer.popleft()

    def _maybe_clear_history(self, time_ms: int) -> bool:
        if not self._history_window_ms and not self._history_max_points:
            return False
        if self._history_window_ms and time_ms < self._history_window_ms:
            if not self._history_max_points or len(self._history_buffer) < self._history_max_points:
                return False
        if self._history_clear_interval_ms and self._last_history_clear_ms is not None:
            if time_ms - self._last_history_clear_ms < self._history_clear_interval_ms:
                return False
        self._last_history_clear_ms = time_ms
        rr.set_time_seconds("time", 0.0)
        self._clear_path(self.base_path, force=True)
        self._relog_history()
        return True

    def _relog_history(self) -> None:
        for timestamp_ms, metrics in self._history_buffer:
            rr.set_time_seconds("time", timestamp_ms / 1000.0)
            if self._rolling_window_ms:
                self._log_clamped_metrics(metrics)
            else:
                self._log_metrics(metrics)

    def _ensure_window_samples(self, fps: float) -> None:
        if self._metric_sample_rate_hz > 0:
            target_rate = self._metric_sample_rate_hz
        else:
            target_rate = fps if fps > 0 else 30.0
        window_samples = max(2, int((self._rolling_window_ms / 1000.0) * target_rate))
        if self._window_samples == window_samples and self._metric_buffer.maxlen == window_samples:
            return
        self._window_samples = window_samples
        self._metric_buffer = deque(self._metric_buffer, maxlen=window_samples)

    def _log_clamped_metrics(self, metrics: TrackingMetrics) -> None:
        velocity = self._clamp_value(metrics.velocity, self._get_range("velocity"))
        rr.log(f"{self.base_path}/velocity", rr.Scalar(velocity))
        if self._is_finite(metrics.mean_x):
            mean_x = self._clamp_value(metrics.mean_x, self._get_range("mean_x"))
            rr.log(f"{self.base_path}/mean/x", rr.Scalar(mean_x))
        if self._is_finite(metrics.mean_y):
            mean_y = self._clamp_value(metrics.mean_y, self._get_range("mean_y"))
            rr.log(f"{self.base_path}/mean/y", rr.Scalar(mean_y))

    def _update_ranges_if_ready(self) -> None:
        if not self._metric_buffer:
            return
        if self._window_samples and len(self._metric_buffer) < self._window_samples:
            return
        self._range_cache = {
            "velocity": self._padded_range([m.velocity for m in self._metric_buffer]),
            "mean_x": self._padded_range([m.mean_x for m in self._metric_buffer]),
            "mean_y": self._padded_range([m.mean_y for m in self._metric_buffer]),
        }

    def _get_range(self, name: str) -> Tuple[float, float]:
        fixed = self._fixed_ranges.get(name)
        if fixed is not None:
            return fixed
        cached = self._range_cache.get(name)
        if cached is not None:
            return cached
        values = [getattr(m, name) for m in self._metric_buffer]
        return self._padded_range(values)

    def _padded_range(self, values: List[float]) -> Tuple[float, float]:
        values = [value for value in values if self._is_finite(value)]
        if not values:
            return (0.0, 1.0)
        if self._range_percentiles:
            low, high = self._range_percentiles
            low = min(max(low, 0.0), 100.0)
            high = min(max(high, 0.0), 100.0)
            if low > high:
                low, high = high, low
            min_val, max_val = np.percentile(values, [low, high]).tolist()
        else:
            min_val = min(values)
            max_val = max(values)
        if min_val == max_val:
            return (min_val - 1.0, max_val + 1.0)
        padding = (max_val - min_val) * self._range_padding
        return (min_val - padding, max_val + padding)

    @staticmethod
    def _clamp_value(value: float, value_range: Tuple[float, float]) -> float:
        low, high = value_range
        return min(max(value, low), high)

    @staticmethod
    def _is_finite(value: float) -> bool:
        return math.isfinite(value)

    def _clear_path(self, path: str, *, timeless: bool = False, force: bool = False) -> None:
        if timeless and hasattr(rr, "Clear"):
            try:
                rr.log(path, rr.Clear(recursive=True), timeless=True)
                return
            except TypeError:
                try:
                    rr.log(path, rr.Clear(), timeless=True)
                    return
                except TypeError:
                    rr.log(path, rr.Clear())
                return

        if force and hasattr(rr, "clear"):
            try:
                rr.clear(path, recursive=True)
            except TypeError:
                rr.clear(path)
            return

        if hasattr(rr, "Clear"):
            try:
                rr.log(path, rr.Clear(recursive=True), timeless=timeless)
                return
            except TypeError:
                try:
                    rr.log(path, rr.Clear(), timeless=timeless)
                    return
                except TypeError:
                    rr.log(path, rr.Clear())
                return

        if hasattr(rr, "clear"):
            try:
                rr.clear(path, recursive=True, timeless=timeless)
            except TypeError:
                try:
                    rr.clear(path, recursive=True)
                except TypeError:
                    rr.clear(path)

    def _should_log_metrics(self, time_ms: int) -> bool:
        if self._metric_log_interval_ms <= 0:
            return True
        if self._last_metric_log_ms is None:
            self._last_metric_log_ms = time_ms
            return True
        if time_ms - self._last_metric_log_ms < self._metric_log_interval_ms:
            return False
        self._last_metric_log_ms = time_ms
        return True
