try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas module is not installed. Please install it using 'pip install pandas'."
    ) from exc


TRACKING_COLUMNS = ["Frame", "Time (ms)", "Velocity", "Mean X", "Mean Y"]


def export_tracking_data(raw_data, path: str) -> None:
    df = pd.DataFrame(raw_data, columns=TRACKING_COLUMNS)
    df.to_csv(path, index=False)
