import sys
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from vnstock import Finance

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *

def fetch_financial_single_symbol(symbol: str) -> pd.DataFrame:
    finance_api = Finance(SOURCE_DATA, symbol, COL_QUARTER)

    # 1) Fetch in parallel (I/O bound)
    def _ratio():      return finance_api.ratio(symbol, lang='en', dropna=True)
    def _income():     return finance_api.income_statement(symbol, lang='en', dropna=True)
    def _cashflow():   return finance_api.cash_flow(symbol, lang='en', dropna=True)

    with ThreadPoolExecutor(max_workers=3) as ex:
        df_ratio, df_income, df_cashflow = ex.map(lambda f: f(), (_ratio, _income, _cashflow))

    if df_ratio.empty or df_income.empty or df_cashflow.empty:
        raise ValueError("One or more components are empty.")

    # 2) Normalize columns quickly (avoid looping over droplevel if not needed)
    def _flatten_cols(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        return df

    df_ratio    = _flatten_cols(df_ratio).rename(columns={"ticker": COL_SYMBOL, "yearReport": COL_YEAR, "lengthReport": COL_QUARTER})
    df_income   = _flatten_cols(df_income).rename(columns={"ticker": COL_SYMBOL, "yearReport": COL_YEAR, "lengthReport": COL_QUARTER})
    df_cashflow = _flatten_cols(df_cashflow).rename(columns={"ticker": COL_SYMBOL, "yearReport": COL_YEAR, "lengthReport": COL_QUARTER})

    # 3) Keep only the columns you actually need BEFORE joining
    cols_to_keep = [
        COL_SYMBOL, COL_YEAR, COL_QUARTER,
        COL_REVENUE, COL_REVENUE_YOY,
        COL_ATTRIBUTE, COL_ATTRIBUTE_YOY,
        COL_ROE, COL_EPS, COL_BVPS, COL_PE
    ]
    def _prune(df):
        existing = [c for c in cols_to_keep if c in df.columns]
        return df[existing].copy()

    df_ratio, df_income, df_cashflow = map(_prune, (df_ratio, df_income, df_cashflow))

    # 4) Ensure merge keys are consistent & fast
    #    (avoid object dtype if they are numeric)
    for df in (df_ratio, df_income, df_cashflow):
        if COL_YEAR in df.columns:
            df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce").astype("Int64")
        if COL_QUARTER in df.columns:
            df[COL_QUARTER] = pd.to_numeric(df[COL_QUARTER], errors="coerce").astype("Int64")
        # symbol as string (small cardinality, safe)
        if COL_SYMBOL in df.columns:
            df[COL_SYMBOL] = df[COL_SYMBOL].astype("string")

    # 5) Join on an index instead of repeated merge
    def _set_key_index(df):
        return df.set_index([COL_SYMBOL, COL_YEAR, COL_QUARTER])

    r, i, c = map(_set_key_index, (df_ratio, df_income, df_cashflow))
    df_merge = r.join(i, how="inner", lsuffix="_r", rsuffix="_i").join(c, how="inner", rsuffix="_c").reset_index()

    # 6) Compute TTM efficiently:
    #    sort ONCE ascending to make rolling "past 4 quarters" correct
    df_merge = df_merge.sort_values([COL_SYMBOL, COL_YEAR, COL_QUARTER], ascending=[True, True, True], kind="mergesort")

    # If multiple symbols can appear, groupby + rolling; for a single symbol it still works
    # Includes current quarter: window=4 on the current and prev 3 rows
    df_merge[COL_EPS_TTM] = (
        df_merge
        .groupby(COL_SYMBOL, sort=False)[COL_EPS]
        .rolling(window=4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # 7) Final presentation: newest first (reverse without a second sort)
    #    We sorted by (year, quarter) ascending—reverse per symbol:
    df_merge = (
        df_merge
        .sort_values([COL_SYMBOL, COL_YEAR, COL_QUARTER], ascending=[True, True, True], kind="mergesort")
        .groupby(COL_SYMBOL, sort=False, group_keys=False)
        .apply(lambda g: g.iloc[::-1])
        .reset_index(drop=True)
    )

    # 8) Keep final columns (guard against missing)
    final_cols = [
        COL_YEAR, COL_QUARTER,
        COL_REVENUE, COL_REVENUE_YOY,
        COL_ATTRIBUTE, COL_ATTRIBUTE_YOY,
        COL_ROE, COL_EPS, COL_EPS_TTM, COL_BVPS, COL_PE
    ]
    df_merge = df_merge[[c for c in final_cols if c in df_merge.columns]]

    return df_merge