import sys
import os
import joblib
import glob
import pandas as pd
import json
from datetime import date, timedelta
import time
from sklearn.preprocessing import LabelEncoder
from vnstock import Quote, Listing, Finance

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *
from src.utility import save_dataframe_to_csv
from src.stock_inference import predict_future_trend
from src.feature_engineering import add_technical_indicators

# --- Constants ---
START_DATE = "2021-01-01"
END_DATE = "2025-04-13"

# --- Financial & Price Data ---
def __fetch_financial_data(symbol: str) -> pd.DataFrame:
    df = Finance(symbol, period=COL_QUARTER, source=SOURCE_DATA).ratio(symbol, lang='en', dropna=True)
    if df.empty: return pd.DataFrame()

    df = df.reset_index(drop=True)
    df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.rename(columns={COL_TICKER: COL_SYMBOL}, inplace=True)
    return df

def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = Quote(symbol, SOURCE_DATA).history(start=start_date, end=end_date, interval='1D')
        if df.empty: return pd.DataFrame()
        df[COL_TIME] = pd.to_datetime(df[COL_TIME])
        df[COL_YEAR_REPORT] = df[COL_TIME].dt.year
        df[COL_QUARTER_REPORT] = df[COL_TIME].dt.quarter
        df[COL_SYMBOL] = symbol
        return df
    except Exception as e:
        print(f"[ERROR] Could not fetch {symbol}: {e}")
        return pd.DataFrame()

# --- Index Integration ---
def __merge_with_index_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    df_index = fetch_historical_data("VNINDEX", start_date, end_date)
    if df_index.empty: return df

    df_index = df_index.rename(columns={
        COL_OPEN: COL_INDEX_OPEN, COL_HIGH: COL_INDEX_HIGH, COL_LOW: COL_INDEX_LOW,
        COL_CLOSE: COL_INDEX_CLOSE, COL_VOLUME: COL_INDEX_VOLUME
    }).drop(columns=[COL_SYMBOL, COL_YEAR_REPORT, COL_QUARTER_REPORT])
    df_index[COL_INDEX_PCT_CHANGE] = df_index[COL_INDEX_CLOSE].pct_change()
    df_index[COL_TIME] = pd.to_datetime(df_index[COL_TIME])

    return pd.merge(df, df_index, on=COL_TIME, how='left')

# --- Data Preprocessor ---
def __process_symbol_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df_price = fetch_historical_data(symbol, start_date, end_date)
    df_fin = __fetch_financial_data(symbol)

    if df_price.empty or df_fin.empty:
        print(f"[WARN] Skipping {symbol} (insufficient data)")
        return pd.DataFrame()

    df = pd.merge(df_price, df_fin, on=[COL_YEAR_REPORT, COL_QUARTER_REPORT, COL_SYMBOL], how='left')
    existing = [col for col in KEEP_COLS if col in df.columns]
    df = df[existing]

    df = __merge_with_index_data(df, start_date, end_date)
    return df

def __fetch_stock_data(symbols: list, start_date: str, end_date: str, delay_sec: float = 1.0) -> pd.DataFrame:
    all_data = []

    for i, sym in enumerate(symbols):
        try:
            print(f"[{i + 1}/{len(symbols)}] Fetching data for: {sym}")
            df = __process_symbol_data(sym, start_date, end_date)
            all_data.append(df)
            time.sleep(delay_sec)
        except Exception as e:
            print(f"Failed to fetch {sym}: {e}")
            time.sleep(delay_sec)
            continue

    if not all_data:
        raise ValueError("No stock data was fetched.")

    df_all = pd.concat(all_data, ignore_index=True)

    df_all = df_all.sort_values([COL_SYMBOL, COL_TIME]).reset_index(drop=True)
    df_all[COL_TIME] = pd.to_datetime(df_all[COL_TIME])
    df_all[COL_MONTH] = df_all[COL_TIME].dt.month
    df_all[COL_QUARTER] = df_all[COL_TIME].dt.quarter
    df_all[COL_DAYOFWEEK] = df_all[COL_TIME].dt.dayofweek
    df_all[COL_TIME_ORDINAL] = df_all[COL_TIME].map(pd.Timestamp.toordinal)
    df_all.drop(columns=[COL_TIME], inplace=True)

    df_all = df_all[~df_all[COL_VOLUME].replace(r'^\s*$', pd.NA, regex=True).isna() & (df_all[COL_VOLUME] != 0)]
    df_all = df_all[~df_all[COL_ROE].replace(r'^\s*$', pd.NA, regex=True).isna() & (df_all[COL_ROE] != 0)]
    df_all = df_all.dropna(subset=STORED_COLS)

    encoder: LabelEncoder = joblib.load("data/symbol_encoder.pkl")
    df_all[COL_SYMBOL] = encoder.transform(df_all[COL_SYMBOL])

    return df_all

# --- Display Utilities ---
def fetch_historical_data_for_display(symbol: str) -> pd.DataFrame:
    df = fetch_historical_data(symbol, (date.today() - timedelta(days=30)).isoformat(), date.today().isoformat())
    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    return df[[COL_TIME, COL_CLOSE]]

def fetch_prediction_data_for_display(symbol: str, predict_start_date: str, forecast_interval: int) -> pd.DataFrame:
    encoder = joblib.load("data/symbol_encoder.pkl")
    start_date = pd.to_datetime(predict_start_date) - pd.DateOffset(years=1)
    start_date_str = start_date.strftime("%Y-%m-%d")

    df = __process_symbol_data(symbol, start_date_str, predict_start_date)
    if df.empty: return pd.DataFrame()

    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    df[FINANCIAL_COLS] = df[FINANCIAL_COLS].ffill()
    df = df.dropna()

    if df[COL_SYMBOL].dtype == object:
        try:
            df[COL_SYMBOL] = encoder.transform(df[COL_SYMBOL])
        except Exception as e:
            print(f"[ERROR] Encoding failed for {symbol}: {e}")
            return pd.DataFrame()

    if len(df) < 30:
        print(f"[WARN] Not enough data for {symbol}")
        return pd.DataFrame()

    try:
        trend_df = predict_future_trend(df, forecast_steps=forecast_interval)
        start = pd.to_datetime(predict_start_date)
        trend_df.insert(0, COL_TIME, [start + timedelta(days=i) for i in range(1, forecast_interval + 1)])
        return trend_df[[COL_TIME, COL_CLOSE, COL_EPS, COL_BVPS]]
    except Exception as e:
        print(f"Prediction failed: {e}")
        return pd.DataFrame()

# --- Symbol Listing ---
def fetch_all_symbols() -> pd.DataFrame:
    return Listing().symbols_by_exchange()[[COL_SYMBOL, COL_EXCHANGE, COL_ORGAN_NAME]]

# --- Entry ---
def main():
    final_output_path = "data/stock_data_final.csv"
    failed_log = "data/failed_symbols.log"
    chunk_size = 500

    if os.path.exists(final_output_path):
        df = pd.read_csv(final_output_path)
        print(f"[INFO] Final dataset already exists at {final_output_path}. Skipping processing.")
    else:
        symbols_with_info = fetch_all_symbols()
        symbols = symbols_with_info.iloc[:, 0].tolist()
        total_chunks = (len(symbols) + chunk_size - 1) // chunk_size

        existing_chunk_files = glob.glob("data/stock_data_chunk_*.csv")
        existing_chunk_indices = {
            int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in existing_chunk_files
        }

        expected_chunk_indices = set(range(1, total_chunks + 1))
        missing_chunk_indices = expected_chunk_indices - existing_chunk_indices

        if missing_chunk_indices:
            print(f"[INFO] Found {len(missing_chunk_indices)} missing chunk(s). Downloading...")

            for chunk_idx in sorted(missing_chunk_indices):
                output_path = f"data/stock_data_chunk_{chunk_idx}.csv"
                chunk = symbols[(chunk_idx - 1) * chunk_size: chunk_idx * chunk_size]
                print(f"[INFO] Fetching chunk {chunk_idx}: {len(chunk)} symbols")

                try:
                    df_chunk = __fetch_stock_data(
                        chunk,
                        start_date=START_DATE,
                        end_date=END_DATE,
                        delay_sec=5
                    )
                    save_dataframe_to_csv(df_chunk, output_path)
                    print(f"[INFO] Saved chunk to {output_path}")
                except Exception as e:
                    with open(failed_log, "a") as f:
                        f.write(f"Chunk {chunk_idx} failed: {e}\n")
                    print(f"[ERROR] Chunk {chunk_idx} failed. Logged.")

        current_chunks = glob.glob("data/stock_data_chunk_*.csv")
        if len(current_chunks) < total_chunks:
            print(f"[ERROR] Only found {len(current_chunks)} of {total_chunks} expected chunk files.")
            print("[HINT] Re-run with force=True to fetch all data from scratch.")
            return

        print("[INFO] Generating final dataset from chunk files...")
        df = pd.concat((pd.read_csv(file) for file in sorted(current_chunks)), ignore_index=True)

        nan_cols = df[STORED_COLS].columns[df[STORED_COLS].isnull().any()].tolist()
        if nan_cols:
            raise ValueError(f"Data contains NaNs in columns: {nan_cols}")

        save_dataframe_to_csv(df, final_output_path)
        print(f"[INFO] Final processed data saved to {final_output_path}")

    # Write or overwrite stock data version
    with open("data/stock_data_version.json", "w") as f:
        json.dump({
            "last_updated": date.today().isoformat(),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "total_rows": len(df),
            "symbols": df[COL_SYMBOL].nunique()
        }, f)

if __name__ == "__main__":
    main()
