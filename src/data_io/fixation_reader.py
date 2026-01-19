try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas module is not installed. Please install it using 'pip install pandas'."
    ) from exc


def read_fixations_csv(
    path: str,
    label: str,
    *,
    time_col: str = None,
    start_col: str = "start_timestamp",
    duration_col: str = "duration",
    time_scale: float = 1000.0,
    duration_scale: float = 1.0,
    fixation_id_col: str = "fixation_id",
    dedupe: bool = True,
):
    df = pd.read_csv(path)
    has_duration = False
    if time_col and time_col in df.columns:
        time_ms = pd.to_numeric(df[time_col], errors="coerce") * time_scale
        data = pd.DataFrame({"time_ms": time_ms})
        if duration_col in df.columns:
            duration_ms = pd.to_numeric(df[duration_col], errors="coerce") * duration_scale
            data["duration_ms"] = duration_ms
            has_duration = True
    else:
        if start_col not in df.columns:
            raise ValueError(f"Missing column '{start_col}' in {path}")
        if duration_col not in df.columns:
            raise ValueError(f"Missing column '{duration_col}' in {path}")
        start = pd.to_numeric(df[start_col], errors="coerce") * time_scale
        duration = pd.to_numeric(df[duration_col], errors="coerce") * duration_scale
        end = start + duration
        data = pd.DataFrame({"start_ms": start, "end_ms": end})

    if fixation_id_col in df.columns:
        data["fixation_id"] = df[fixation_id_col]

    if "time_ms" in data.columns:
        data = data.dropna(subset=["time_ms"])
        if "fixation_id" in data.columns and dedupe:
            grouped = (
                data.groupby("fixation_id", as_index=False)
                .agg(
                    start_ms=("time_ms", "min"),
                    end_ms=("time_ms", "max"),
                    duration_ms=("duration_ms", "median") if has_duration else ("time_ms", "size"),
                )
            )
            if has_duration and "duration_ms" in grouped.columns:
                valid = grouped["duration_ms"].notna() & (grouped["duration_ms"] > 0)
                grouped.loc[valid, "end_ms"] = grouped.loc[valid, "start_ms"] + grouped.loc[
                    valid, "duration_ms"
                ]
            data = grouped[["start_ms", "end_ms"]]
        else:
            if has_duration and "duration_ms" in data.columns:
                data["start_ms"] = data["time_ms"]
                data["end_ms"] = data["time_ms"] + data["duration_ms"]
                data = data.dropna(subset=["start_ms", "end_ms"])
            else:
                data = data.rename(columns={"time_ms": "start_ms"})
                data["end_ms"] = data["start_ms"]

    data = data.dropna(subset=["start_ms", "end_ms"])
    data = data[data["end_ms"] >= data["start_ms"]]

    return [(float(row.start_ms), float(row.end_ms), label) for row in data.itertuples(index=False)]
