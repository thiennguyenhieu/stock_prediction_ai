import sys
import os
from vnstock import Quote, Listing, Finance
from datetime import date, timedelta, datetime
import pandas as pd
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *

# ------------------ Data Fetching Functions ------------------

def fetch_financial_data(symbol: str) -> pd.DataFrame:
    """
    Fetch quarterly financial ratios for a given stock symbol.
    """
    df = Finance(symbol, period='quarter', source=SOURCE_DATA).ratio(symbol, lang='en', dropna=True)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index(drop=True)
    df.columns = df.columns.droplevel(0)  # Remove MultiIndex if exists
    df.rename(columns={'ticker': 'symbol'}, inplace=True)

    return df

def fetch_historical_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical price data for a stock between start and end dates.
    """
    try:
        df = Quote(symbol, SOURCE_DATA).history(start=start_date, end=end_date, interval='1D')
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df['time'] = pd.to_datetime(df['time'])
    df['yearReport'] = df['time'].dt.year
    df['lengthReport'] = df['time'].dt.quarter
    df['symbol'] = symbol

    return df


def fetch_historical_data_for_display(symbol: str):
    start_date = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d') # display data in 1 month
    end_date = date.today().strftime('%Y-%m-%d')
    df = fetch_historical_data(symbol, start_date, end_date)
    df = df[[COL_TIME, COL_CLOSE]]
    df[COL_TIME] = pd.to_datetime(df[COL_TIME])  # ensure time is datetime

    return df


def fetch_prediction_data_for_display(symbol: str, start_date: str, forecast_interval: int) -> pd.DataFrame:
    """
    Generate fake predicted stock price data starting from `start_date`
    for the given `forecast_interval` (number of future days).
    """
    base_price = 23.60
    daily_increase = 0.10  # linear growth for fake prediction

    # Parse the input start_date string to a datetime object
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()

    # Generate future dates and fake prices
    dates = [start_dt + timedelta(days=i) for i in range(forecast_interval)]
    prices = [round(base_price + i * daily_increase, 2) for i in range(forecast_interval)]

    df_pred = pd.DataFrame({
        'time': dates,
        'close': prices
    })

    return df_pred


def process_symbol_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch and merge historical and financial data for a single symbol.
    """
    print(f"[INFO] Processing {symbol}...")

    df_price = fetch_historical_data(symbol, start_date, end_date)
    if df_price.empty:
        print(f"[WARN] No price data found for {symbol}")
        return pd.DataFrame()

    df_financial = fetch_financial_data(symbol)
    if df_financial.empty:
        print(f"[WARN] No financial data for {symbol}")
        return pd.DataFrame()

    df_merged = pd.merge(
        df_price, df_financial,
        on=['yearReport', 'lengthReport', 'symbol'],
        how='left'
    )

    existing_cols = [col for col in KEEP_COLS if col in df_merged.columns]
    df_filtered = df_merged[existing_cols]

    # Filter out rows with missing ROE
    df_filtered = df_filtered[~df_filtered[COL_ROE].replace(r'^\s*$', pd.NA, regex=True).isna()]

    return df_filtered

def fetch_stock_data(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Process a list of stock symbols and combine their financial + price data.
    """
    all_data = [process_symbol_data(symbol, start_date, end_date) for symbol in symbols]
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# ------------------ Save Utility ------------------

def save_dataframe_to_csv(df: pd.DataFrame, filename: str = "data/stock_data.csv"):
    """
    Save DataFrame to a CSV file.
    """
    if df.empty:
        print("[WARN] No data to save.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[SUCCESS] Data saved to: {filename}")

# ------------------ Main Execution ------------------

def main():
    symbol_group = 'VN30'
    limit = 15

    symbols = Listing().symbols_by_group(symbol_group)[:limit]
    start_date = '2021-01-01'
    end_date = date.today().strftime('%Y-%m-%d')

    print(f"[INFO] Fetching data for group: {symbol_group} ({len(symbols)} symbols)")
    stock_data = fetch_stock_data(symbols, start_date, end_date)

    save_dataframe_to_csv(stock_data, filename="data/stock_data.csv")


if __name__ == "__main__":
    main()
