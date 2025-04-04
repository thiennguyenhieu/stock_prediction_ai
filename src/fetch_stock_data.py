from vnstock import Quote, Listing, Finance
from datetime import datetime
import pandas as pd
import os
from stock_config import KEEP_COLS, COL_ROE, SOURCE_DATA

def fetch_financial_data(symbol: str, source: str) -> pd.DataFrame:
    """Fetch quarterly financial ratios for a given stock symbol."""
    df = Finance(symbol, period='quarter', source=source).ratio(symbol, lang='en', dropna=True)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index(drop=True)
    df.columns = df.columns.droplevel(0)  # Remove MultiIndex if exists
    df.rename(columns={'ticker': 'symbol'}, inplace=True)
    
    return df


def fetch_stock_data(source: str, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch and merge stock price and financial data for multiple symbols."""
    data_frames = [
        process_symbol_data(symbol, source, start_date, end_date)
        for symbol in symbols
    ]
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()


def process_symbol_data(symbol: str, source: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Process stock history and merge it with financial ratios for one symbol."""
    print(f"Fetching data for {symbol}...")

    # Historical price data
    df_price = Quote(symbol, source).history(start=start_date, end=end_date, interval='1D')
    if df_price.empty:
        print(f"No price data found for {symbol}")
        return pd.DataFrame()

    df_price['time'] = pd.to_datetime(df_price['time'])
    df_price['yearReport'] = df_price['time'].dt.year
    df_price['lengthReport'] = df_price['time'].dt.quarter
    df_price['symbol'] = symbol

    # Financial ratio data
    df_financial = fetch_financial_data(symbol, source)
    if df_financial.empty:
        print(f"No financial data found for {symbol}")
        return pd.DataFrame()

    # Merge on year, quarter, and symbol
    df_merged = pd.merge(df_price, df_financial, on=['yearReport', 'lengthReport', 'symbol'], how='left')

    # Filter only the columns that exist in the actual DataFrame
    existing_cols = [col for col in KEEP_COLS if col in df_merged.columns]
    df_final = df_merged[existing_cols]

    # Filter out rows with missing ROE
    df_final = df_final[~df_final[COL_ROE].replace(r'^\s*$', pd.NA, regex=True).isna()]

    return df_final


def save_dataframe_to_csv(df: pd.DataFrame, filename: str = "data/stock_data.csv"):
    """Save DataFrame to CSV file."""
    if df.empty:
        print("Warning: No data to save.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# ------------------ Entry Point ------------------

if __name__ == "__main__":
    # Load VN30 symbols (limit to 15 to avoid overloading API)
    symbols = Listing().symbols_by_group('VN30')[:15]
    
    start = '2021-01-01'
    end = datetime.today().strftime('%Y-%m-%d')

    stock_data = fetch_stock_data(source=SOURCE_DATA, symbols=symbols, start_date=start, end_date=end)
    save_dataframe_to_csv(stock_data, filename="data/stock_data.csv")
