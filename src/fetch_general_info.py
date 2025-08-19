import pandas as pd
from vnstock import Listing
from vnstock import Company

from src.constants import *

# --- Symbol Listing ---
def fetch_all_symbols() -> pd.DataFrame:
    return Listing().symbols_by_exchange()[[COL_SYMBOL, COL_EXCHANGE, COL_ORGAN_NAME]]

# --- Company overview ---
def fetch_company_overview(symbol: str) -> pd.DataFrame:
    overview = Company(SOURCE_DATA, symbol).overview()[[COL_ISSUE_SHARE, COL_ICB_NAME]]
    return overview.iloc[0, 0], overview.iloc[0, 1]

# --- Company overview ---
def fetch_dividend(symbol: str) -> pd.DataFrame:
    return Company('TCBS', symbol).dividends().head(5)