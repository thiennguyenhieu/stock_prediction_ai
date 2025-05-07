import sys
import os
import glob
import json
import joblib
import pandas as pd
from datetime import date, datetime
from sklearn.preprocessing import LabelEncoder
from vnstock import Finance

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *
from src.utility import save_dataframe_to_csv

ENCODER_PATH = "data/symbol_encoder.pkl"
FINAL_OUTPUT_PATH = "data/financial_data_final.csv"
FAILED_LOG_PATH = "data/financial_failed_symbols.log"
CHUNK_PATH_PATTERN = "data/financial_data_chunk_*.csv"
KEEP_COLUMNS = [
    "symbol", "year", "quarter",
    "EPS (VND)", "BVPS (VND)",
    "P/E", "P/B", "P/S",
    "Net Profit Margin (%)", "ROE (%)", "ROA (%)", "ROIC (%)",
    "EBIT Margin (%)", "Gross Profit Margin (%)",
    "Financial Leverage", "Interest Coverage",
    "Net Cash Flows from Operating Activities before BIT",
    "Net Cash Flows from Investing Activities",
    "Net increase/decrease in cash and cash equivalents",
    "Revenue (Bn. VND)", "Revenue YoY (%)", "Net Profit For the Year",
    "Profit before tax", "Cost of Sales", "Gross Profit",
    "Operating Profit/Loss", "Selling Expenses", "General & Admin Expenses",
    "Fixed Asset-To-Equity", "Market Capital (Bn. VND)",
    "Depreciation and Amortisation", "Outstanding Share (Mil. Shares)",
    "Debt/Equity", "Owners' Equity/Charter Capital",
    "Current Ratio", "Quick Ratio", "Cash Ratio",
    "Attribute to parent company (Bn. VND)"
]

def fetch_financial_single_symbol(symbol: str) -> pd.DataFrame:
    try:
        encoder: LabelEncoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        raise RuntimeError(f"[FATAL] Failed to load symbol encoder: {e}")
    
    # Fetch financial data
    finance_api = Finance(symbol, period=COL_QUARTER, source=SOURCE_DATA)
    df_ratio = finance_api.ratio(symbol, lang='en', dropna=True)
    df_income = finance_api.income_statement(symbol, lang='en', dropna=True)
    df_cashflow = finance_api.cash_flow(symbol, lang='en', dropna=True)

    if df_ratio.empty or df_income.empty or df_cashflow.empty:
        raise ValueError("One or more components are empty.")

    for df in [df_ratio, df_income, df_cashflow]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)

    # --- Merge and preprocess ---
    df_ratio = df_ratio.rename(columns={"ticker": COL_SYMBOL, "yearReport": COL_YEAR, "lengthReport": COL_QUARTER})
    df_income = df_income.rename(columns={"ticker": COL_SYMBOL, "yearReport": COL_YEAR, "lengthReport": COL_QUARTER})
    df_cashflow = df_cashflow.rename(columns={"ticker": COL_SYMBOL, "yearReport": COL_YEAR, "lengthReport": COL_QUARTER})

    # Merge on standardized keys
    df_merge = df_ratio.merge(df_income, on=[COL_SYMBOL, COL_YEAR, COL_QUARTER], how="inner") \
                    .merge(df_cashflow, on=[COL_SYMBOL, COL_YEAR, COL_QUARTER], how="inner")
    
    # Filter only useful columns
    df_merge = df_merge[[col for col in KEEP_COLUMNS if col in df_merge.columns]]

    # Sort by date
    df_merge.sort_values(by=[COL_SYMBOL, COL_YEAR, COL_QUARTER], inplace=True)

    # Drop rows with missing essential targets (e.g., EPS or BVPS)
    df_merge.dropna(subset=[COL_EPS, COL_BVPS], inplace=True)

    # Reset index
    df_merge.reset_index(drop=True, inplace=True)

    df_merge[COL_SYMBOL] = encoder.transform(df_merge[COL_SYMBOL])

    return df_merge

def fetch_financial_data(symbols: list) -> pd.DataFrame:
    all_data = []
    
    for i, symbol in enumerate(symbols):
        try:
            print(f"[{i+1}/{len(symbols)}] Fetching {symbol} at {datetime.now().strftime('%H:%M:%S')}")

            df_symbol = fetch_financial_single_symbol(symbol)
            all_data.append(df_symbol)

        except Exception as e:
            with open(FAILED_LOG_PATH, "a") as log_file:
                log_file.write(f"[{symbol}] Failed at {datetime.now()}: {e}\n")
            print(f"[ERROR] {symbol} skipped: {e}")

    if not all_data:
        raise ValueError("No valid data collected.")

    return pd.concat(all_data, ignore_index=True)

def main():
    chunk_size = 500

    if os.path.exists(FINAL_OUTPUT_PATH):
        print(f"[INFO] Final dataset already exists at {FINAL_OUTPUT_PATH}.")
        df = pd.read_csv(FINAL_OUTPUT_PATH)
    else:
        symbols = fetch_all_symbols()
        symbols = symbols[(symbols["exchange"] == "HSX") & (symbols.iloc[:, 0].str.len() == 3)].iloc[:, 0].tolist()

        total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
        existing_chunks = {
            int(os.path.basename(f).split("_")[-1].split(".")[0])
            for f in glob.glob(CHUNK_PATH_PATTERN)
        }

        for chunk_idx in sorted(set(range(1, total_chunks + 1)) - existing_chunks):
            chunk = symbols[(chunk_idx - 1) * chunk_size: chunk_idx * chunk_size]
            print(f"[INFO] Processing chunk {chunk_idx}/{total_chunks} with {len(chunk)} symbols...")
            try:
                df_chunk = fetch_financial_data(chunk)
                save_dataframe_to_csv(df_chunk, f"data/financial_data_chunk_{chunk_idx}.csv")
            except Exception as e:
                with open(FAILED_LOG_PATH, "a") as f:
                    f.write(f"[Chunk {chunk_idx}] failed: {e}\n")
                print(f"[ERROR] Chunk {chunk_idx} failed: {e}")

        chunk_files = sorted(glob.glob(CHUNK_PATH_PATTERN))
        if len(chunk_files) < total_chunks:
            print("[ERROR] Not all chunks were fetched. Please re-run the script to complete missing chunks.")
            return

        df = pd.concat((pd.read_csv(f) for f in chunk_files), ignore_index=True)
        save_dataframe_to_csv(df, FINAL_OUTPUT_PATH)
        print(f"[INFO] Final dataset saved to {FINAL_OUTPUT_PATH}")

    with open("data/financial_data_version.json", "w") as f:
        json.dump({
            "last_updated": date.today().isoformat(),
            "total_rows": len(df),
            "symbols": df[COL_SYMBOL].nunique()
        }, f)

if __name__ == "__main__":
    main()

