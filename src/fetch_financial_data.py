import sys
import os
import pandas as pd
from vnstock import Finance

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *
from src.utility import save_dataframe_to_csv

START_DATE = "2021-01-01"
END_DATE = "2025-04-20"

# --- Price Data ---
def __fetch_financial_data(symbol: str) -> pd.DataFrame:
    df = Finance(symbol, period=COL_QUARTER, source=SOURCE_DATA).ratio(symbol, lang='en', dropna=True)
    if df.empty: return pd.DataFrame()

    df = df.reset_index(drop=True)
    df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.rename(columns={COL_TICKER: COL_SYMBOL}, inplace=True)
    return df