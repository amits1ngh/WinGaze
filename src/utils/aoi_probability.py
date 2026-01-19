import math
from typing import List, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "numpy module is not installed. Please install it using 'pip install numpy'."
    ) from exc


def _unique_in_order(values: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _smooth_series(values: np.ndarray, kind: str, smooth_ms: float, bin_ms: float) -> np.ndarray:
    if kind == "none" or smooth_ms <= 0 or bin_ms <= 0:
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


def compute_aoi_probability_series(
    fixations: List[Tuple[float, float, str]],
    *,
    start_ms: float,
    end_ms: float,
    bin_ms: float,
    include_none: bool = False,
    smooth_kind: str = "gaussian",
    smooth_ms: float = 500.0,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if not fixations:
        return np.array([], dtype=float), [], np.zeros((0, 0), dtype=float)
    if bin_ms <= 0:
        raise ValueError("bin_ms must be > 0")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be > start_ms")

    labels = _unique_in_order([label for _, _, label in fixations])
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
        idx = label_index[label]
        for bin_idx in range(first, last + 1):
            bin_start = start_ms + bin_idx * bin_ms
            bin_end = bin_start + bin_ms
            overlap = min(end, bin_end) - max(start, bin_start)
            if overlap > 0:
                probs[bin_idx, idx] += overlap / bin_ms

    if include_none:
        none = 1.0 - probs.sum(axis=1)
        none = np.clip(none, 0.0, 1.0)
        probs = np.column_stack([probs, none])
        labels.append("None")

    smoothed = np.zeros_like(probs)
    for idx in range(probs.shape[1]):
        smoothed[:, idx] = _smooth_series(
            probs[:, idx], smooth_kind, smooth_ms, bin_ms
        )

    time_ms = start_ms + (np.arange(num_bins) + 0.5) * bin_ms
    return time_ms, labels, smoothed
