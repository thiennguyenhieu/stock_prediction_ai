import sys, os
import pandas as pd
from vnstock import Listing
from vnstock import Company
from vnstock import Screener

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *

# --- Symbol Listing ---
def fetch_all_symbols() -> pd.DataFrame:
    return Listing().symbols_by_exchange()[[COL_SYMBOL, COL_EXCHANGE, COL_ORGAN_NAME]]

# --- Company overview ---
def fetch_company_overview(symbol: str) -> str:
    overview = Company(SOURCE_DATA, symbol).overview()[[COL_ICB_NAME]]
    if overview.empty:
        return ""
    return str(overview.iloc[0, 0])

# --- Company overview ---
def fetch_dividend(symbol: str) -> pd.DataFrame:
    df_div = Company('TCBS', symbol).dividends().head(4)
    df_div["cash_dividend_percentage"] = (df_div["cash_dividend_percentage"] * 100).round(2).astype(str) + "%"
    
    issue_map = {
        "cash": VI_STRINGS["cash"],
        "share": VI_STRINGS["share"],
    }
    df_div["issue_method"] = df_div["issue_method"].map(issue_map).fillna(df_div["issue_method"])

    rename_map = {
        "exercise_date": VI_STRINGS["exercise_date"],
        "cash_year": VI_STRINGS["cash_year"],
        "cash_dividend_percentage": VI_STRINGS["cash_dividend_percentage"],
        "issue_method": VI_STRINGS["issue_method"],
    }

    df_div = df_div.rename(columns=rename_map)
    df_div.reset_index(drop=True)

    return df_div

# --- Filtering ---
def filter_by_params(query_params: dict) -> pd.DataFrame:
    screener = Screener()

    screener_df = screener.stock(params=query_params, limit=1700)

    print("Available columns:", screener_df.columns.tolist())
    return screener_df

# ----- CLI -----
if __name__ == "__main__":
    out = filter_by_params({"exchangeName": "HOSE,HNX,UPCOM"})
    print(out)