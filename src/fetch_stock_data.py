from vnstock import Quote, Listing

def fetch_stock_data(symbol: str, source: str, start_date: str, end_date: str, save_csv: bool = False):
    """
    Fetch historical stock data for a given stock symbol.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'VIC', 'VNM').
        source (str): Data provider (e.g., 'VCI').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        save_csv (bool, optional): Whether to save data as a CSV file. Defaults to False.

    Returns:
        DataFrame or None: Dataframe containing stock price history, or None on failure.
    """
    try:
        df = Quote(symbol, source).history(start=start_date, end=end_date, interval='1D')

        if df.empty:
            print(f"No data found for {symbol} in the given date range.")
            return None

        if save_csv:
            filename = f"data/{symbol}_stock_data.csv"
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")

        return df

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def fetch_all_symbols(source: str, symbols: set, start_date: str, end_date: str, save_csv: bool = False):
    """
    Fetch stock data for multiple symbols.

    Args:
        source (str): Data provider (e.g., 'VCI').
        symbols (set): Set of stock symbols to fetch.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        save_csv (bool, optional): Whether to save data as CSV files. Defaults to False.
    """
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        fetch_stock_data(symbol, source, start_date, end_date, save_csv)

# Example usage
if __name__ == "__main__":
    start_date = '2022-01-01'
    end_date = '2025-01-01'
    source = 'VCI'
    symbols = {'VNM', 'BSR', 'ACB', 'TCB'}  # Use a set for uniqueness
    fetch_all_symbols(source, symbols, start_date, end_date, save_csv=True)