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
    
    df_income = finance_api.income_statement(lang='vi', dropna=True)

    cols_income_keep = [
        "Năm",                           # Năm báo cáo
        "Kỳ",                            # Quý/Năm
        "Doanh thu thuần",               # Phi tài chính
        "Doanh thu (đồng)",              # Nếu nguồn dùng tên này
        "Chi phí quản lý DN",
        "LN trước thuế",
        "Lợi nhuận sau thuế của Cổ đông công ty mẹ (đồng)",
    ]
    df_income = df_income.loc[:, [c for c in cols_income_keep if c in df_income.columns]]
    
    df_income.reset_index(drop=True)

    df_income = df_income.sort_values(by=["Năm", "Kỳ"], ascending=[True, True])
    df_income = df_income.tail(4)

    return df_income

def fetch_ratio(symbol: str) -> pd.DataFrame:
    finance_api = Finance(SOURCE_DATA, symbol, COL_QUARTER)

    df_ratio = finance_api.ratio(lang='vi', dropna=True)
    df_ratio.columns = df_ratio.columns.droplevel(0)
    
    cols_ratio_keep = [
        "Năm",
        "Kỳ",
        "EPS (VND)", 
        "BVPS (VND)",
        "P/E", 
        "P/B",
        "ROE (%)", 
        "ROA (%)",
        "Số CP lưu hành (Triệu CP)",
    ]
    df_ratio = df_ratio.loc[:, [c for c in cols_ratio_keep if c in df_ratio.columns]]
    
    df_ratio.reset_index(drop=True)

    df_ratio = df_ratio.sort_values(by=["Năm", "Kỳ"], ascending=[True, True])
    df_ratio = df_ratio.tail(4)

    return df_ratio

def fetch_balance_sheet(symbol: str) -> pd.DataFrame:
    finance_api = Finance(SOURCE_DATA, symbol, COL_QUARTER)

    df_balance_sheet = finance_api.balance_sheet(lang='vi', dropna=True)

    cols_balance_keep = [
        "Năm",
        "Kỳ",
        "TỔNG CỘNG TÀI SẢN (đồng)",
        "NỢ PHẢI TRẢ (đồng)",
        "VỐN CHỦ SỞ HỮU (đồng)",
    ]
    df_balance_sheet = df_balance_sheet.loc[:, [c for c in cols_balance_keep if c in df_balance_sheet.columns]]
    
    df_balance_sheet.reset_index(drop=True)

    df_balance_sheet = df_balance_sheet.sort_values(by=["Năm", "Kỳ"], ascending=[True, True])
    df_balance_sheet = df_balance_sheet.tail(4)

    return df_balance_sheet