import os
import pandas as pd
import numpy as np

def compute_trend(close_series, threshold=0.01):
    pct_change = close_series.pct_change().shift(-1)  # shift(-1) = lookahead
    return np.where(pct_change > threshold, 1, np.where(pct_change < -threshold, -1, 0))

# Handle millisecond timestamps (if any)
def convert_if_millis(date_str):
    try:
        if date_str.isdigit() and len(date_str) >= 13:
            dt = pd.to_datetime(int(date_str), unit='ms')
        else:
            dt = pd.to_datetime(date_str, utc=True, errors='coerce')  # force UTC

        return dt.tz_localize(None) if dt.tzinfo is not None else dt
    except:
        return pd.NaT

def save_dataframe_to_csv(df: pd.DataFrame, filename: str):
    """
    Save DataFrame to a CSV file.
    """
    if df.empty:
        print("[WARN] No data to save.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[SUCCESS] Data saved to: {filename}")
