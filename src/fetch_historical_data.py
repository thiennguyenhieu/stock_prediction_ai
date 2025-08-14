import sys
import os
import time
import glob
import json
import joblib
import pandas as pd
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
from vnstock import Quote

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *
from src.fetch_general_info import fetch_all_symbols
from src.utility import save_dataframe_to_csv
from src.feature_engineering import *

# Constants
START_DATE = "2021-01-01"
END_DATE = "2025-08-12"
FINAL_OUTPUT_PATH = "data/historical_data_final.csv"
FAILED_LOG_PATH = "data/historical_failed_symbols.log"
CHUNK_PATH_PATTERN = "data/historical_data_chunk_*.csv"

# --- Data Fetching ---
def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = Quote(SOURCE_DATA, symbol).history(start=start_date, end=end_date, interval='1D')
        if df.empty:
            return pd.DataFrame()
        df[COL_TIME] = pd.to_datetime(df[COL_TIME])
        df[COL_SYMBOL] = symbol
        return df
    except Exception as e:
        print(f"[ERROR] Could not fetch {symbol}: {e}")
        return pd.DataFrame()

def merge_with_index(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    index_df = fetch_historical_data("VNINDEX", start_date, end_date)
    if index_df.empty:
        return df

    index_df = index_df.rename(columns={
        COL_OPEN: COL_INDEX_OPEN, COL_HIGH: COL_INDEX_HIGH, COL_LOW: COL_INDEX_LOW,
        COL_CLOSE: COL_INDEX_CLOSE, COL_VOLUME: COL_INDEX_VOLUME
    }).drop(columns=[COL_SYMBOL])
    index_df[COL_INDEX_PCT_CHANGE] = index_df[COL_INDEX_CLOSE].pct_change()

    return pd.merge(df, index_df, on=COL_TIME, how='left')

def process_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = fetch_historical_data(symbol, start_date, end_date)
    if df.empty:
        print(f"[WARN] Skipping {symbol} (no data)")
        return pd.DataFrame()
    return merge_with_index(df, start_date, end_date)

def fetch_all_symbols_data(symbols: list, start_date: str, end_date: str, delay_sec: float = 1.0) -> pd.DataFrame:
    all_data = []
    for i, symbol in enumerate(symbols):
        try:
            print(f"[{i+1}/{len(symbols)}] Fetching {symbol}...")
            df = process_symbol(symbol, start_date, end_date)
            if not df.empty:
                all_data.append(df)
            time.sleep(delay_sec)
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
            time.sleep(delay_sec)
    
    if not all_data:
        raise ValueError("No stock data fetched.")

    return post_process_data(pd.concat(all_data, ignore_index=True))

def post_process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([COL_SYMBOL, COL_TIME]).reset_index(drop=True)
    df = add_multi_cycle_fourier_features(df)
    df.drop(columns=[COL_TIME], inplace=True)
    df = add_technical_indicators(df)

    # Clean volume
    df = df[~df[COL_VOLUME].replace(r'^\s*$', pd.NA, regex=True).isna() & (df[COL_VOLUME] != 0)]

    df[COL_INDEX_PCT_CHANGE] = df[COL_INDEX_PCT_CHANGE].bfill()

    encoder: LabelEncoder = joblib.load(ENCODER_PATH)
    df[COL_SYMBOL] = encoder.transform(df[COL_SYMBOL])

    return df

# --- Display Support ---
def fetch_recent_price(symbol: str) -> pd.DataFrame:
    df = fetch_historical_data(symbol, (date.today() - timedelta(days=30)).isoformat(), date.today().isoformat())
    return df[[COL_TIME, COL_VOLUME, COL_CLOSE]] if not df.empty else pd.DataFrame()

# --- Main Orchestration ---
def main():
    chunk_size = 500

    if os.path.exists(FINAL_OUTPUT_PATH):
        print(f"[INFO] Final dataset already exists at {FINAL_OUTPUT_PATH}.")
        df = pd.read_csv(FINAL_OUTPUT_PATH)
    else:
        symbols = fetch_all_symbols()
        symbols = symbols[
            (symbols["exchange"] == "HSX") & 
            (symbols.iloc[:, 0].str.len() == 3)
        ].iloc[:, 0].tolist()

        total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
        existing_chunks = {
            int(os.path.basename(f).split("_")[-1].split(".")[0])
            for f in glob.glob(CHUNK_PATH_PATTERN)
        }

        for chunk_idx in sorted(set(range(1, total_chunks + 1)) - existing_chunks):
            chunk = symbols[(chunk_idx - 1) * chunk_size: chunk_idx * chunk_size]
            print(f"[INFO] Fetching chunk {chunk_idx}/{total_chunks} ({len(chunk)} symbols)...")
            try:
                df_chunk = fetch_all_symbols_data(chunk, START_DATE, END_DATE, delay_sec=5)
                save_dataframe_to_csv(df_chunk, f"data/historical_data_chunk_{chunk_idx}.csv")
            except Exception as e:
                with open(FAILED_LOG_PATH, "a") as f:
                    f.write(f"Chunk {chunk_idx} failed: {e}\n")
                print(f"[ERROR] Chunk {chunk_idx} failed: {e}")

        chunk_files = sorted(glob.glob(CHUNK_PATH_PATTERN))
        if len(chunk_files) < total_chunks:
            print("[ERROR] Not all chunks were fetched. Re-run to complete missing chunks.")
            return

        print("[INFO] Merging chunk files...")
        df = pd.concat((pd.read_csv(f) for f in chunk_files), ignore_index=True)
        save_dataframe_to_csv(df, FINAL_OUTPUT_PATH)
        print(f"[INFO] Final dataset saved to {FINAL_OUTPUT_PATH}")

    # Save metadata
    with open("data/historical_data_version.json", "w") as f:
        json.dump({
            "last_updated": date.today().isoformat(),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "total_rows": len(df),
            "symbols": df[COL_SYMBOL].nunique()
        }, f)

if __name__ == "__main__":
    main()
