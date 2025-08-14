import os
import pandas as pd

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
