import argparse
import math
from typing import List, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "numpy module is not installed. Please install it using 'pip install numpy'."
    ) from exc

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas module is not installed. Please install it using 'pip install pandas'."
    ) from exc


def _unique_in_order(values) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _smooth_series(values: np.ndarray, kind: str, smooth_ms: float, bin_ms: float) -> np.ndarray:
    if kind == "none" or smooth_ms <= 0:
        return values
    if bin_ms <= 0:
        return values
    count = len(values)
    if count <= 1:
        return values
    if kind == "moving":
        window_bins = max(1, int(round(smooth_ms / bin_ms)))
        window_bins = min(window_bins, count)
        if window_bins <= 1:
            return values
        kernel = np.ones(window_bins, dtype=float)
        kernel /= kernel.sum()
    else:
        sigma_bins = max(1e-6, smooth_ms / bin_ms)
        window_bins = max(3, int(round(sigma_bins * 6)))
        if window_bins % 2 == 0:
            window_bins += 1
        if window_bins > count:
            window_bins = count if count % 2 == 1 else count - 1
        if window_bins < 3:
            return values
        radius = window_bins // 2
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def _compute_bins(
    fixations: List[Tuple[float, float, str]],
    *,
    start_ms: float,
    end_ms: float,
    bin_ms: float,
    include_none: bool,
) -> Tuple[np.ndarray, List[str]]:
    if bin_ms <= 0:
        raise ValueError("bin_ms must be > 0")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be > start_ms")

    labels = _unique_in_order(label for _, _, label in fixations)
    label_index = {label: idx for idx, label in enumerate(labels)}
    num_bins = int(math.ceil((end_ms - start_ms) / bin_ms))
    probs = np.zeros((num_bins, len(labels)), dtype=float)

    for start, end, label in fixations:
        if end <= start_ms or start >= end_ms:
            continue
        start = max(start, start_ms)
        end = min(end, end_ms)
        if end <= start:
            continue
        first = int((start - start_ms) // bin_ms)
        last = int((end - start_ms) // bin_ms)
        if last >= num_bins:
            last = num_bins - 1
        for bin_idx in range(first, last + 1):
            bin_start = start_ms + bin_idx * bin_ms
            bin_end = bin_start + bin_ms
            overlap = min(end, bin_end) - max(start, bin_start)
            if overlap > 0:
                probs[bin_idx, label_index[label]] += overlap / bin_ms

    if include_none:
        none = 1.0 - probs.sum(axis=1)
        none = np.clip(none, 0.0, 1.0)
        probs = np.column_stack([probs, none])
        labels.append("None")

    return probs, labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AOI probability over time from fixation intervals."
    )
    parser.add_argument("inputs", nargs="+", help="Path(s) to fixation CSV file(s).")
    parser.add_argument("--start-col", default="start_ms", help="Start time column.")
    parser.add_argument("--end-col", default="end_ms", help="End time column.")
    parser.add_argument(
        "--duration-col",
        default="duration",
        help="Duration column (used if end-col is missing).",
    )
    parser.add_argument("--aoi-col", default="aoi", help="AOI label column.")
    parser.add_argument(
        "--aoi-label",
        action="append",
        default=None,
        help="Override AOI label for each input file (repeat per file).",
    )
    parser.add_argument(
        "--fixation-id-col",
        default="fixation_id",
        help="Fixation id column for de-duplication.",
    )
    parser.add_argument(
        "--keep-samples",
        action="store_true",
        help="Keep per-sample rows (skip fixation de-duplication).",
    )
    parser.add_argument("--bin-ms", type=float, default=100.0, help="Bin size in ms.")
    parser.add_argument(
        "--smooth-kind",
        choices=("none", "moving", "gaussian"),
        default="gaussian",
        help="Smoothing method.",
    )
    parser.add_argument(
        "--smooth-ms",
        type=float,
        default=500.0,
        help="Smoothing window in ms (sigma for gaussian).",
    )
    parser.add_argument("--time-scale", type=float, default=1.0, help="Scale to ms.")
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=None,
        help="Scale duration to ms (defaults to time-scale).",
    )
    parser.add_argument("--start-ms", type=float, default=None, help="Override start time.")
    parser.add_argument("--end-ms", type=float, default=None, help="Override end time.")
    parser.add_argument("--sep", default=",", help="CSV separator.")
    parser.add_argument("--decimal", default=".", help="Decimal separator.")
    parser.add_argument("--include-none", action="store_true", help="Add None AOI.")
    parser.add_argument("--stacked", action="store_true", help="Stacked area plot.")
    parser.add_argument("--output-csv", default=None, help="Write binned data to CSV.")
    parser.add_argument("--output-png", default=None, help="Save plot to PNG.")
    parser.add_argument("--show", action="store_true", help="Show plot window.")
    return parser.parse_args()


def _load_fixations(path: str, args: argparse.Namespace, label_override: str = None):
    df = pd.read_csv(path, sep=args.sep, decimal=args.decimal)
    if args.start_col not in df.columns:
        raise ValueError(f"Missing column '{args.start_col}' in {path}")

    end_col = args.end_col if args.end_col in df.columns else None
    duration_col = args.duration_col if args.duration_col in df.columns else None
    if end_col is None and duration_col is None:
        raise ValueError(
            f"Missing '{args.end_col}' and '{args.duration_col}' in {path}"
        )

    if label_override is None:
        if args.aoi_col not in df.columns:
            raise ValueError(f"Missing column '{args.aoi_col}' in {path}")
        label = df[args.aoi_col].astype(str)
    else:
        label = pd.Series([label_override] * len(df), index=df.index)

    start = pd.to_numeric(df[args.start_col], errors="coerce")
    if end_col:
        end = pd.to_numeric(df[end_col], errors="coerce")
        start_ms = start * args.time_scale
        end_ms = end * args.time_scale
    else:
        duration = pd.to_numeric(df[duration_col], errors="coerce")
        duration_scale = args.duration_scale if args.duration_scale is not None else args.time_scale
        start_ms = start * args.time_scale
        end_ms = start_ms + duration * duration_scale

    data = pd.DataFrame(
        {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "label": label,
        }
    )
    if args.fixation_id_col in df.columns:
        data["fixation_id"] = df[args.fixation_id_col]

    data = data.dropna(subset=["start_ms", "end_ms", "label"])
    data = data[data["end_ms"] > data["start_ms"]]
    if data.empty:
        return []

    if args.fixation_id_col in data.columns and not args.keep_samples:
        with_id = data[data["fixation_id"].notna()]
        without_id = data[data["fixation_id"].isna()][["start_ms", "end_ms", "label"]]
        if not with_id.empty:
            grouped = (
                with_id.groupby(["fixation_id", "label"], as_index=False)
                .agg(start_ms=("start_ms", "min"), end_ms=("end_ms", "max"))
            )
            grouped = grouped[["start_ms", "end_ms", "label"]]
            data = pd.concat([grouped, without_id], ignore_index=True)
        else:
            data = without_id

    return [
        (float(row.start_ms), float(row.end_ms), row.label)
        for row in data.itertuples(index=False)
    ]


def main() -> None:
    args = _parse_args()
    labels = args.aoi_label or []
    if labels and len(labels) != len(args.inputs):
        raise ValueError("Provide one --aoi-label per input file.")

    fixations = []
    for idx, path in enumerate(args.inputs):
        label_override = labels[idx] if labels else None
        fixations.extend(_load_fixations(path, args, label_override=label_override))

    if not fixations:
        raise ValueError("No valid fixation rows found.")

    start_ms = args.start_ms if args.start_ms is not None else min(f[0] for f in fixations)
    end_ms = args.end_ms if args.end_ms is not None else max(f[1] for f in fixations)

    probs, labels = _compute_bins(
        fixations,
        start_ms=start_ms,
        end_ms=end_ms,
        bin_ms=args.bin_ms,
        include_none=args.include_none,
    )

    smoothed = np.zeros_like(probs)
    for idx in range(probs.shape[1]):
        smoothed[:, idx] = _smooth_series(
            probs[:, idx], args.smooth_kind, args.smooth_ms, args.bin_ms
        )

    time_ms = start_ms + (np.arange(probs.shape[0]) + 0.5) * args.bin_ms
    out = {"time_ms": time_ms, "time_s": time_ms / 1000.0}
    for idx, label in enumerate(labels):
        out[label] = smoothed[:, idx]
    out_df = pd.DataFrame(out)

    if args.output_csv:
        out_df.to_csv(args.output_csv, index=False)

    if args.output_png or args.show:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is not installed. Please install it using 'pip install matplotlib'."
            ) from exc
        fig, ax = plt.subplots(figsize=(10, 4))
        x = out_df["time_s"].to_numpy()
        series = [out_df[label].to_numpy() for label in labels]
        if args.stacked:
            ax.stackplot(x, series, labels=labels, alpha=0.8)
        else:
            for label, values in zip(labels, series):
                ax.plot(x, values, label=label, linewidth=1.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("AOI probability")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        fig.tight_layout()
        if args.output_png:
            fig.savefig(args.output_png, dpi=150)
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
