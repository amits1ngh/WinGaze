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

from core.data_types import TrackingMetrics
from vis.layouts import create_default_rrb


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
        self.video_main_path = "video/main"
        self.video_eye_path = "video/eyetrack"
        self._app_name = app_name
        self._spawn_once = spawn
        self._segment_index = 0
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
        if should_log_metrics and not did_relog:
            rr.set_time_seconds("time", log_time_ms / 1000.0)
            if self._rolling_window_ms:
                self._log_clamped_metrics(metrics)
            else:
                self._log_metrics(metrics)

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

    def _reset_state(self) -> None:
        self._metric_buffer.clear()
        self._history_buffer.clear()
        self._last_history_clear_ms = None
        self._last_metric_log_ms = None
        self._time_offset_ms = 0
        self._range_cache.clear()

    def start_segment(self, start_time_ms: int) -> None:
        self._segment_index += 1
        self.base_path = f"{self._metrics_root}/segment_{self._segment_index}"
        self._reset_state()
        self._time_offset_ms = max(0, int(start_time_ms))
        self._send_blueprint()

    def _send_blueprint(self) -> None:
        if not hasattr(rr, "send_blueprint"):
            return
        if self._history_window_ms:
            window_seconds = self._history_window_ms / 1000.0
        else:
            window_seconds = self._rolling_window_ms / 1000.0 if self._rolling_window_ms else None
        rr.send_blueprint(
            create_default_rrb(window_seconds=window_seconds, metrics_root=self.base_path)
        )

    def _log_image(self, path: str, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rr.log(path, rr.Image(rgb), static=True)

    def _log_metrics(self, metrics: TrackingMetrics) -> None:
        rr.log(f"{self.base_path}/velocity", rr.Scalar(metrics.velocity))
        if self._is_finite(metrics.mean_x):
            rr.log(f"{self.base_path}/mean/x", rr.Scalar(metrics.mean_x))
        if self._is_finite(metrics.mean_y):
            rr.log(f"{self.base_path}/mean/y", rr.Scalar(metrics.mean_y))

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
