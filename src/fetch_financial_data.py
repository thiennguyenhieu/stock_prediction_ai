import sys
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from vnstock import Finance

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *

def fetch_income_statement(symbol: str) -> pd.DataFrame:
    finance_api = Finance(SOURCE_DATA, symbol, COL_QUARTER)
    
    df_income = finance_api.income_statement(symbol, lang='vi', dropna=True)
    
    cols_to_drop = [
        "CP",
        "Cổ tức đã nhận",
        "Chi phí thuế TNDN hoãn lại",
        "Cổ đông thiểu số",
        "Cổ đông của Công ty mẹ",
        "Lãi cơ bản trên cổ phiếu",
    ]
    df_income = df_income.drop(columns=cols_to_drop, errors="ignore")
    
    df_income.reset_index(drop=True)

    df_income = df_income.sort_values(by=["Năm", "Kỳ"], ascending=[True, True])
    df_income = df_income.tail(8)

    return df_income

def fetch_ratio(symbol: str) -> pd.DataFrame:
    finance_api = Finance(SOURCE_DATA, symbol, COL_QUARTER)

    df_ratio = finance_api.ratio(symbol, lang='vi', dropna=True)
    df_ratio.columns = df_ratio.columns.droplevel(0)
    
    cols_to_drop = [
        "CP",
        "Số CP lưu hành (Triệu CP)"
    ]
    df_ratio = df_ratio.drop(columns=cols_to_drop, errors="ignore")
    
    df_ratio.reset_index(drop=True)

    df_ratio = df_ratio.sort_values(by=["Năm", "Kỳ"], ascending=[True, True])
    df_ratio = df_ratio.tail(8)

    return df_ratio