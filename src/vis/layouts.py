import inspect
from typing import Optional

import rerun.blueprint as rrb
try:
    import rerun.blueprint.components as rbc
except ImportError:
    rbc = None


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


def _pick_enum_value(enum_cls, candidates):
    if not enum_cls:
        return None
    for name in candidates:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)
    return None


def _legend_location_values():
    candidates = ("TOP_LEFT", "TopLeft", "TOPLEFT", "Top_Left", "top_left")
    enum_names = (
        "LegendLocation",
        "LegendPlacement",
        "LegendPosition",
        "PlotLegendLocation",
        "PlotLegendPlacement",
        "PlotLegendPosition",
        "Corner",
    )
    values = []
    for enum_name in enum_names:
        enum_cls = getattr(rrb, enum_name, None)
        value = _pick_enum_value(enum_cls, candidates)
        if value is not None:
            values.append(value)
    values.append("top_left")
    return values


def _plot_legend_value():
    corner_cls = None
    if rbc is not None and hasattr(rbc, "Corner2D"):
        corner_cls = rbc.Corner2D
    else:
        corner_cls = getattr(rrb, "Corner2D", None)
    corner = _pick_enum_value(
        corner_cls, ("LeftTop", "TOP_LEFT", "TopLeft", "TOPLEFT", "top_left")
    )
    if corner is None:
        corner = "top_left"
    plot_legend_cls = getattr(rrb, "PlotLegend", None)
    if plot_legend_cls is None:
        return corner
    for kwargs in ({"corner": corner, "visible": True}, {"corner": corner}):
        try:
            return plot_legend_cls(**kwargs)
        except TypeError:
            continue
    return corner


def _build_legend():
    legend_cls = getattr(rrb, "PlotLegend", None) or getattr(rrb, "Legend", None)
    if not legend_cls:
        return None
    try:
        legend_params = inspect.signature(legend_cls).parameters
    except (TypeError, ValueError):
        legend_params = {}
    loc_param = None
    for candidate in ("position", "placement", "location", "anchor", "corner"):
        if candidate in legend_params:
            loc_param = candidate
            break
    legend_kwargs_list = []
    if loc_param:
        for loc in _legend_location_values():
            kwargs = {loc_param: loc}
            if "visible" in legend_params:
                kwargs["visible"] = True
            legend_kwargs_list.append(kwargs)
    if "visible" in legend_params:
        legend_kwargs_list.append({"visible": True})
    legend_kwargs_list.append({})
    for kwargs in legend_kwargs_list:
        try:
            return legend_cls(**kwargs)
        except TypeError:
            continue
    return None


def _legend_hide_kwargs(params):
    if "plot_legend" in params:
        plot_legend_cls = getattr(rrb, "PlotLegend", None)
        if plot_legend_cls is not None:
            try:
                return {"plot_legend": plot_legend_cls(visible=False)}
            except TypeError:
                pass
        return {"plot_legend": None}
    if "legend" in params:
        legend_cls = getattr(rrb, "Legend", None)
        if legend_cls is not None:
            try:
                return {"legend": legend_cls(visible=False)}
            except TypeError:
                pass
    return {}


def _legend_kwargs(params):
    if "plot_legend" in params:
        return {"plot_legend": _plot_legend_value()}

    legend_candidates = ("legend_location", "legend_position", "legend_placement", "legend_anchor")
    for candidate in legend_candidates:
        if candidate in params:
            for loc in _legend_location_values():
                return {candidate: loc}

    if "legend" in params:
        legend = _build_legend()
        if legend is not None:
            return {"legend": legend}

    return {}


def _timeseries_view(
    origin: str,
    name: str,
    window_seconds: Optional[float],
    *,
    legend_visible: Optional[bool] = True,
):
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
    if range_param and window_seconds and window_seconds > 0:
        kwargs[range_param] = _build_time_range(window_seconds)
    if legend_visible is False:
        legend_kwargs = _legend_hide_kwargs(params)
    else:
        legend_kwargs = _legend_kwargs(params)
    kwargs.update(legend_kwargs)
    try:
        return rrb.TimeSeriesView(**kwargs)
    except TypeError:
        if legend_kwargs:
            for key in legend_kwargs:
                kwargs.pop(key, None)
            try:
                return rrb.TimeSeriesView(**kwargs)
            except TypeError:
                return rrb.TimeSeriesView(origin=origin, name=name)
        return rrb.TimeSeriesView(origin=origin, name=name)


def _text_view(origin: str, name: str, preferred: Optional[str] = None):
    if preferred == "log":
        view_cls = getattr(rrb, "TextLogView", None)
        if view_cls is None:
            return None
        try:
            return view_cls(origin=origin, name=name)
        except TypeError:
            return view_cls(origin=origin)
    if preferred == "document":
        view_cls = getattr(rrb, "TextDocumentView", None)
        if view_cls is None:
            return None
        try:
            return view_cls(origin=origin, name=name)
        except TypeError:
            return view_cls(origin=origin)
    for view_name in ("TextDocumentView", "TextLogView"):
        view_cls = getattr(rrb, view_name, None)
        if view_cls is None:
            continue
        try:
            return view_cls(origin=origin, name=name)
        except TypeError:
            return view_cls(origin=origin)
    return None


def create_default_rrb(
    window_seconds: Optional[float] = None,
    metrics_root: str = "metrics",
    annotation_path: str = "annotations/speech",
    aoi_path: str = "aoi/probability",
    annotation_view: Optional[str] = None,
) -> rrb.Blueprint:
    metrics_root = metrics_root.strip("/") or "metrics"
    annotation_path = annotation_path.strip("/") or "annotations/speech"
    aoi_path = aoi_path.strip("/") or "aoi/probability"
    velocity_path = f"{metrics_root}/velocity"
    mean_root = f"{metrics_root}/mean"
    metric_views = [
        _timeseries_view(velocity_path, "Velocity", window_seconds),
        _timeseries_view(
            aoi_path,
            "Looks Probability",
            window_seconds,
            legend_visible=True,
        ),
        _timeseries_view(mean_root, "Mean X/Y", window_seconds),
    ]
    speech_view = _text_view(annotation_path, "Speech", preferred=annotation_view)
    if speech_view is not None:
        metric_views.append(speech_view)
    row_shares = [1] * len(metric_views)
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="video/main", name="Main Video"),
                rrb.Spatial2DView(origin="video/eyetrack", name="Eye Tracking"),
                name="Video",
                row_shares=[1, 1],
            ),
            rrb.Vertical(
                *metric_views,
                name="Hand Metrics",
                row_shares=row_shares,
            ),
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
