import sys
import os
from vnstock import Quote, Listing, Finance
from datetime import date, timedelta, datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *
from src.utility import  *

# ------------------ Data Fetching Functions ------------------

def __fetch_financial_data(symbol: str) -> pd.DataFrame:
    """
    Fetch quarterly financial ratios for a given stock symbol.
    """
    df = Finance(symbol, period='quarter', source=SOURCE_DATA).ratio(symbol, lang='en', dropna=True)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index(drop=True)
    df.columns = df.columns.droplevel(0)  # Remove MultiIndex if exists
    df.rename(columns={COL_TICKER: COL_SYMBOL}, inplace=True)

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

    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    df[COL_YEAR_REPORT] = df[COL_TIME].dt.year
    df[COL_QUARTER_REPORT] = df[COL_TIME].dt.quarter
    df[COL_SYMBOL] = symbol

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
        COL_TIME: dates,
        COL_CLOSE: prices
    })

    return df_pred

def fetch_all_symbols () -> pd.DataFrame:
    df = Listing().symbols_by_exchange()
    keep_cols = [COL_SYMBOL, COL_EXCHANGE, COL_ORGAN_NAME]
    df = df[keep_cols]
    
    return df

def __process_symbol_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch and merge historical and financial data for a single symbol.
    """
    print(f"[INFO] Processing {symbol}...")

    df_price = fetch_historical_data(symbol, start_date, end_date)
    if df_price.empty:
        print(f"[WARN] No price data found for {symbol}")
        return pd.DataFrame()

    df_financial = __fetch_financial_data(symbol)
    if df_financial.empty:
        print(f"[WARN] No financial data for {symbol}")
        return pd.DataFrame()

    df_merged = pd.merge(
        df_price, df_financial,
        on=[COL_YEAR_REPORT, COL_QUARTER_REPORT, COL_SYMBOL],
        how='left'
    )

    existing_cols = [col for col in KEEP_COLS if col in df_merged.columns]
    df_merged = df_merged[existing_cols]

    return df_merged

def __fetch_stock_data(symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Process a list of stock symbols and combine their financial + price data.
    """
    list_all_data = [__process_symbol_data(symbol, start_date, end_date) for symbol in symbols]
    df_all_data = pd.concat(list_all_data, ignore_index=True) if list_all_data else pd.DataFrame()
    
    # Adding market index data to the stock dataset
    df_index_data = fetch_historical_data('VNINDEX', start_date, end_date)
    df_index_data = df_index_data.drop(columns=[COL_SYMBOL])
    df_index_data = df_index_data.drop(columns=[COL_YEAR_REPORT])
    df_index_data = df_index_data.drop(columns=[COL_QUARTER_REPORT])
    df_index_data = df_index_data.rename(columns={
        COL_OPEN: COL_INDEX_OPEN,
        COL_HIGH: COL_INDEX_HIGH,
        COL_LOW: COL_INDEX_LOW,
        COL_CLOSE: COL_INDEX_CLOSE,
        COL_VOLUME: COL_INDEX_VOLUME
    })
    # Create index % change feature
    df_index_data[COL_INDEX_PCT_CHANGE] = df_index_data[COL_INDEX_CLOSE].pct_change()

    # Make sure time is in datetime format
    df_all_data[COL_TIME] = pd.to_datetime(df_all_data[COL_TIME])
    df_index_data[COL_TIME] = pd.to_datetime(df_index_data[COL_TIME])
    
    df_merged_data = pd.merge(df_all_data, df_index_data, on=COL_TIME, how='left')
    
    # Filter out rows with missing ROE
    df_merged_data = df_merged_data[~df_merged_data[COL_ROE].replace(r'^\s*$', pd.NA, regex=True).isna()]
    
    # Sort by symbol and time (symbol will be a string, not one-hot)
    df_merged_data = df_merged_data.sort_values(by=[COL_SYMBOL, COL_TIME]).reset_index(drop=True)

    # Label encode the 'symbol' column for embedding
    label_encoder = LabelEncoder()
    df_merged_data[COL_SYMBOL] = label_encoder.fit_transform(df_merged_data[COL_SYMBOL])
    joblib.dump(label_encoder, "data/symbol_encoder.pkl")

    return df_merged_data

# ------------------ Main Execution ------------------

def main():
    symbol_group = 'VN30'
    limit = 15

    symbols = Listing().symbols_by_group(symbol_group)[:limit]
    start_date = '2021-01-01'
    end_date = date.today().strftime('%Y-%m-%d')

    print(f"[INFO] Fetching data for group: {symbol_group} ({len(symbols)} symbols)")
    stock_data = __fetch_stock_data(symbols, start_date, end_date)

    save_dataframe_to_csv(stock_data, filename="data/stock_data.csv")


if __name__ == "__main__":
    main()
