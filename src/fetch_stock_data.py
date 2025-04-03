from vnstock import Quote
import pandas as pd

def fetch_stock_data(source: str, symbols: set, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch stock data for multiple symbols.

    Args:
        source (str): Data provider (e.g., 'VCI').
        symbols (set): Set of stock symbols to fetch.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing stock price history, or an empty DataFrame if no data is found.
    """
    df_list = []

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        df = Quote(symbol, source).history(start=start_date, end=end_date, interval='1D')

        if df.empty:
            print(f"No data found for {symbol} in the given date range.")
        else:
            df_list.append(df)

    if not df_list:
        print("No stock data retrieved.")
        return pd.DataFrame()  # Return empty DataFrame if no data was fetched

    return pd.concat(df_list, ignore_index=True)

def save_dataframe_to_csv(df: pd.DataFrame, filename: str = "data/stock_data.csv"):
    """
    Saves a DataFrame to a fixed CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The file path to save the CSV (default: 'data/stock_data.csv').
    """
    if df.empty:
        print("Warning: No data to save.")
        return
    
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Example usage
if __name__ == "__main__":
    start_date = '2022-01-01'
    end_date = '2025-01-01'
    source = 'VCI'
    symbols = {'VNM', 'BSR', 'ACB', 'TCB'}  # Use a set for uniqueness

    stock_data = fetch_stock_data(source, symbols, start_date, end_date)
    filename = f"data/{start_date}_{end_date}_stock_data.csv"
    save_dataframe_to_csv(stock_data, filename)
