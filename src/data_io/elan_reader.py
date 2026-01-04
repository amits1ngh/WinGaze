try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas module is not installed. Please install it using 'pip install pandas'."
    ) from exc

from core.data_types import SegmentTimes

ELAN_START_COL = "Anfangszeit - msec"
ELAN_END_COL = "Endzeit - msec"


def read_elan_file(path: str):
    df = pd.read_csv(path, sep="\t")
    modality_columns = [
        col for col in df.columns if col not in [ELAN_START_COL, ELAN_END_COL]
    ]
    start_prefixes = ("block", "trial")
    start_index = None
    for idx, col in enumerate(modality_columns):
        name = col.strip().lower()
        if any(name.startswith(prefix) for prefix in start_prefixes):
            start_index = idx
            break
    if start_index is not None:
        modality_columns = modality_columns[start_index:]
    return df, modality_columns


def get_segment_times(df, modality: str, start_label: str, end_label: str) -> SegmentTimes:
    if modality not in df.columns:
        raise ValueError(f"Column '{modality}' not found in ELAN data")

    filtered = df[df[modality].notna()]
    start_row = filtered[filtered[modality].astype(str) == start_label].iloc[0]
    end_row = filtered[filtered[modality].astype(str) == end_label].iloc[0]

    return SegmentTimes(
        start_ms=float(start_row[ELAN_START_COL]),
        end_ms=float(end_row[ELAN_END_COL]),
    )
