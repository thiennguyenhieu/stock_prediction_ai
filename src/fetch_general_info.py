import sys, os
import pandas as pd
from vnstock import Listing
from vnstock import Company
from vnstock import Screener

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *
from src.utility import save_dataframe_to_csv

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
def filter_by_valuation() -> pd.DataFrame:
    """
    Filter valuation stocks based on criteria:
    - Market cap > 500 tỷ VND
    - Revenue growth (5y) > 0
    - EPS growth (5y) > 0
    - Gross margin > 25%
    - ROE > 15%
    - Debt/Equity < 100%
    - P/E < 10
    """
    screener = Screener()
    df = screener.stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=1000000)

    # Convert to numeric
    for col in ["market_cap", "revenue_growth_5y", "eps_growth_5y",
                "gross_margin", "roe", "doe", "pe"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize ratios (ROE, gross margin might be % already)
    roe = df["roe"].where(df["roe"] <= 100, df["roe"] / 100)
    gross_margin = df["gross_margin"].where(df["gross_margin"] <= 100, df["gross_margin"] / 100)
    doe_ratio = df["doe"].where(df["doe"] < 10, df["doe"] / 100)  # handle % scale

    mask = (
        (df["market_cap"] > 500) &             # tỷ VND
        (df["revenue_growth_5y"] > 0) &
        (df["eps_growth_5y"] > 0) &
        (gross_margin > 25) &
        (roe > 15) &
        (doe_ratio < 1.0) &                    # < 100%
        (df["pe"] > 0) & (df["pe"] < 10)
    )

    filtered = df.loc[mask].copy()

    cols = [
        "ticker", "exchange", "industry",
        "market_cap", "pe", "pb",
        "roe", "gross_margin", "doe",
        "revenue_growth_5y", "eps_growth_5y"
    ]
    existing = [c for c in cols if c in filtered.columns]

    # Sort by P/E ascending (cheaper first), then ROE descending (higher first)
    out = filtered[existing].sort_values(by=["pe", "roe"], ascending=[True, False])

    # Rename to Vietnamese
    rename_map = {
        "ticker": "Mã cổ phiếu",
        "exchange": "Sàn giao dịch",
        "industry": "Ngành",
        "market_cap": "Vốn hóa (tỷ VND)",
        "pe": "P/E",
        "pb": "P/B",
        "roe": "ROE (%)",
        "gross_margin": "Biên LN gộp (%)",
        "doe": "Nợ/VCSH",
        "revenue_growth_5y": "Tăng trưởng DT 5 năm (%)",
        "eps_growth_5y": "Tăng trưởng EPS 5 năm (%)"
    }

    return out.rename(columns=rename_map)

def filter_by_growth() -> pd.DataFrame:
    """
    Growth screen:
      - market_cap > 500 (tỷ VND)
      - eps_ttm_growth1_year > 20 (%)
      - last_quarter_profit_growth > 20 (%)
      - last_quarter_revenue_growth > 20 (%)
      - roe > 15 (%, robust to ratio vs % scale)
      - P/E < 15
    """

    screener = Screener()
    df = screener.stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=1000000)
    
    # Ensure required columns exist
    required = [
        "market_cap",
        "eps_ttm_growth1_year",
        "last_quarter_profit_growth",
        "last_quarter_revenue_growth",
        "roe",
        "pe"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    # Cast to numeric
    num_cols = [
        "market_cap",
        "eps_ttm_growth1_year",
        "last_quarter_profit_growth",
        "last_quarter_revenue_growth",
        "roe",
        "pe"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize ROE to percent scale if it's in ratio form (e.g., 0.18 -> 18)
    roe = df["roe"].copy()
    if roe.dropna().le(1.5).mean() > 0.8:
        roe = roe * 100.0

    # Build mask
    mask = (
        (df["market_cap"] > 500) &
        (df["eps_ttm_growth1_year"] > 20) &
        (df["last_quarter_profit_growth"] > 20) &
        (df["last_quarter_revenue_growth"] > 20) &
        (roe > 15) &
        (df["pe"] > 0) & (df["pe"] < 15)
    )

    out = df.loc[mask].copy()

    # Choose and rename output columns for clarity
    cols = [
        "ticker", "exchange", "industry",
        "market_cap",
        "eps_ttm_growth1_year",
        "last_quarter_profit_growth",
        "last_quarter_revenue_growth",
        "roe", "pe", "pb"
    ]
    existing = [c for c in cols if c in out.columns]

    # Helpful sort: strongest recent growth first, then ROE
    sort_cols = [c for c in ["last_quarter_profit_growth",
                             "last_quarter_revenue_growth",
                             "eps_ttm_growth1_year", "roe", "pe", "pb"] if c in out.columns]
    if sort_cols:
        out = out[existing].sort_values(by=sort_cols, ascending=False)
    else:
        out = out[existing]

    rename_map = {
        "ticker": "Mã cổ phiếu",
        "exchange": "Sàn giao dịch",
        "industry": "Ngành",
        "market_cap": "Vốn hóa (tỷ VND)",
        "eps_ttm_growth1_year": "Tăng trưởng EPS 12T (%)",
        "last_quarter_profit_growth": "Tăng trưởng LN quý gần nhất (%)",
        "last_quarter_revenue_growth": "Tăng trưởng DT quý gần nhất (%)",
        "roe": "ROE (%)",
        "pe": "P/E",
        "pb": "P/B"
    }

    df_vn = out.rename(columns=rename_map)

    return df_vn

# ----- CLI -----
if __name__ == "__main__":
    out_val = filter_by_valuation()
    print(out_val)
    out_growth = filter_by_growth()
    print(out_growth)