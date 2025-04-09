import sys
import os
import joblib
import pandas as pd
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder
from vnstock import Quote, Listing, Finance

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *
from src.utility import save_dataframe_to_csv
from src.stock_inference import predict_future_trend

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
    print(f"[INFO] Processing {symbol}")
    df_price = fetch_historical_data(symbol, start_date, end_date)
    df_fin = __fetch_financial_data(symbol)

    if df_price.empty or df_fin.empty:
        print(f"[WARN] Skipping {symbol} (insufficient data)")
        return pd.DataFrame()

    df = pd.merge(df_price, df_fin, on=[COL_YEAR_REPORT, COL_QUARTER_REPORT, COL_SYMBOL], how='left')
    existing = [col for col in INPUT_FEATURES if col in df.columns]
    df = df[existing]

    df = __merge_with_index_data(df, start_date, end_date)
    return df

def __fetch_stock_data(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    df_all = pd.concat([__process_symbol_data(sym, start_date, end_date) for sym in symbols], ignore_index=True)
    df_all[COL_TIME] = pd.to_datetime(df_all[COL_TIME])
    df_all = df_all[~df_all[COL_ROE].replace(r'^\s*$', pd.NA, regex=True).isna()]
    df_all = df_all.sort_values([COL_SYMBOL, COL_TIME]).reset_index(drop=True)

    # Encode symbol
    encoder = LabelEncoder()
    df_all[COL_SYMBOL] = encoder.fit_transform(df_all[COL_SYMBOL])
    joblib.dump(encoder, "data/symbol_encoder.pkl")

    return df_all

# --- Display Utilities ---
def fetch_historical_data_for_display(symbol: str) -> pd.DataFrame:
    df = fetch_historical_data(symbol, (date.today() - timedelta(days=30)).isoformat(), date.today().isoformat())
    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    return df[[COL_TIME, COL_CLOSE]]

def fetch_prediction_data_for_display(symbol: str, predict_start_date: str, forecast_interval: int) -> pd.DataFrame:
    encoder = joblib.load("data/symbol_encoder.pkl")
    # Go back 1 year from prediction start date
    start_date = pd.to_datetime(predict_start_date) - pd.DateOffset(years=1)
    start_date_str = start_date.strftime("%Y-%m-%d")

    # Process symbol data using adjusted start date
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
    group = "VN30"
    limit = 15
    symbols = Listing().symbols_by_group(group)[:limit]
    print(f"[INFO] Fetching data for {group} ({len(symbols)} stocks)")

    df = __fetch_stock_data(symbols, start_date="2021-01-01", end_date=date.today().isoformat())
    save_dataframe_to_csv(df, "data/stock_data.csv")

if __name__ == "__main__":
    main()
