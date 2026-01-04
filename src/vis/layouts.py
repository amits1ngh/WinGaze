import inspect
from typing import Optional

import rerun.blueprint as rrb


# For more information about Rerun Blueprint, visit: https://rerun.io/docs/concepts/blueprint

def _build_time_range(window_seconds: float):
    time_range_cls = getattr(rrb, "TimeRange", None)
    if not time_range_cls:
        return window_seconds
    try:
        inspect.signature(time_range_cls)
    except (TypeError, ValueError):
        return window_seconds
    for args in ((window_seconds,), (0.0, window_seconds)):
        try:
            return time_range_cls(*args)
        except TypeError:
            continue
    return window_seconds


def _timeseries_view(origin: str, name: str, window_seconds: Optional[float]):
    if window_seconds and window_seconds > 0:
        kwargs = {"origin": origin, "name": name}
        try:
            params = inspect.signature(rrb.TimeSeriesView).parameters
        except (TypeError, ValueError):
            params = {}
        range_param = None
        for candidate in ("visible_time_range", "time_range", "time_range_seconds", "time_range_s"):
            if candidate in params:
                range_param = candidate
                break
        if range_param:
            kwargs[range_param] = _build_time_range(window_seconds)
        try:
            return rrb.TimeSeriesView(**kwargs)
        except TypeError:
            return rrb.TimeSeriesView(origin=origin, name=name)
    return rrb.TimeSeriesView(origin=origin, name=name)


def create_default_rrb(
    window_seconds: Optional[float] = None,
    metrics_root: str = "metrics",
) -> rrb.Blueprint:
    metrics_root = metrics_root.strip("/") or "metrics"
    velocity_path = f"{metrics_root}/velocity"
    mean_root = f"{metrics_root}/mean"
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="video/main", name="Main Video"),
                rrb.Spatial2DView(origin="video/eyetrack", name="Eye Tracking"),
                name="Video",
                row_shares=[1, 1],
            ),
            rrb.Vertical(
                _timeseries_view(velocity_path, "Velocity", window_seconds),
                _timeseries_view(mean_root, "Mean X/Y", window_seconds),
                name="Hand Metrics",
                row_shares=[1, 1],
            ),
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
