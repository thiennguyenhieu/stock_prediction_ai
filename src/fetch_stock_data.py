from vnstock import Quote, Listing, Finance
from datetime import datetime
import pandas as pd

def fetch_financial_data(symbol: str, source: str) -> pd.DataFrame:
    """
    Fetch financial ratio data for a stock symbol.
    
    Args:
        symbol (str): Stock symbol to fetch financial ratios.
        source (str): Data provider (e.g., 'VCI').

    Returns:
        pd.DataFrame: Financial ratios DataFrame for the symbol.
    """
    df_finance = Finance(symbol, period='quarter', source=source).ratio(symbol, lang='en', dropna=True).head()
    df_finance = df_finance.reset_index(drop=True)
    df_finance.columns = df_finance.columns.droplevel(0)  # Drop first level of MultiIndex
    df_finance.rename(columns={'ticker': 'symbol'}, inplace=True)
    
    return df_finance

def fetch_stock_data(source: str, symbols: set, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch stock data for multiple symbols.
    
    Args:
        source (str): Data provider (e.g., 'VCI').
        symbols (set): Set of stock symbols to fetch.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: DataFrame containing stock price history and financial ratios.
    """
    df_list = [
        process_symbol_data(symbol, source, start_date, end_date)
        for symbol in symbols
    ]
    
    # Return an empty DataFrame if no data was retrieved
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def process_symbol_data(symbol: str, source: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Process and merge historical stock and financial data for a single symbol.
    
    Args:
        symbol (str): Stock symbol to fetch data for.
        source (str): Data provider (e.g., 'VCI').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Processed DataFrame with stock data and financial ratios.
    """
    print(f"Fetching data for {symbol}...")
    
    # Fetch historical stock data
    df_history = Quote(symbol, source).history(start=start_date, end=end_date, interval='1D')
    if df_history.empty:
        print(f"No data found for {symbol}.")
        return pd.DataFrame()  # Return empty DataFrame if no stock data found
    
    # Convert 'time' to datetime and extract year and quarter
    df_history['time'] = pd.to_datetime(df_history['time'])
    df_history['yearReport'] = df_history['time'].dt.year
    df_history['lengthReport'] = df_history['time'].dt.quarter
    df_history['symbol'] = symbol  # Add symbol column
    
    # Fetch financial data
    df_finance = fetch_financial_data(symbol, source)
    
    # Merge financial data with stock data
    df_merged = pd.merge(df_history, df_finance, how='left', on=['yearReport', 'lengthReport', 'symbol'])
    
    # Keep only relevant columns
    keep_cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'ROE (%)', 'ROA (%)', 'P/E', 'P/B', 'EPS (VND)', 'BVPS (VND)']
    return df_merged[keep_cols]

def save_dataframe_to_csv(df: pd.DataFrame, filename: str = "data/stock_data.csv"):
    """
    Saves a DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): File path to save the CSV (default: 'data/stock_data.csv').
    """
    if df.empty:
        print("Warning: No data to save.")
        return
    
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Example usage
if __name__ == "__main__":  
    # Get the symbols, and keep only the first 15
    symbols = Listing().symbols_by_group('VN30')[:15]
    
    # Fetch the stock data and save the data to CSV
    # !!! Due to the limit on the number of requests to the server, use one of them at once !!! 
    #trained_stock_data = fetch_stock_data('VCI', symbols, '2022-01-01', '2025-01-01')
    #save_dataframe_to_csv(trained_stock_data, f"data/trained_stock_data.csv")  
    test_stock_data = fetch_stock_data('VCI', symbols, '2025-01-01', datetime.today().strftime('%Y-%m-%d'))
    save_dataframe_to_csv(test_stock_data, f"data/test_stock_data.csv")
