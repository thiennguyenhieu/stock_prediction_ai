import vnstock
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date, save_csv=False):
    """
    Fetch historical stock data for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'VIC', 'VNM').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        save_csv (bool): Whether to save data as a CSV file.
    
    Returns:
        pd.DataFrame: Dataframe containing stock price history.
    """
    df = vnstock.stock_historical_data(ticker, start_date, end_date)
    
    if df is not None and not df.empty:
        print(df.head())  # Print first few rows
        
        if save_csv:
            filename = f"data/{ticker}_stock_data.csv"
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
    else:
        print(f"No data found for {ticker} in the given date range.")
    
    return df

# Example usage
if __name__ == "__main__":
    stock_data = fetch_stock_data("VNM", "2024-01-01", "2024-03-31", save_csv=True)