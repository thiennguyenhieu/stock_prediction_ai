import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *

def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

def signed_expm1(x):
    return np.sign(x) * (np.expm1(np.abs(x)))

def add_multi_cycle_fourier_features(df):
    """
    Adds Fourier-based sine/cosine features for multiple seasonal cycles:
    - daily, weekly, quarterly, yearly

    Requirements:
    - df must contain a datetime column named COL_TIME (or convertible to datetime)

    For each cycle, the following features will be added:
    - sin_{name}: sine transformation of the time ordinal over the cycle period
    - cos_{name}: cosine transformation of the time ordinal over the cycle period
    """
    df = df.copy()

    # Ensure COL_TIME is datetime and compute ordinal
    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    df[COL_TIME_ORDINAL] = df[COL_TIME].map(pd.Timestamp.toordinal)

    # Define cycles and their periods in days
    cycles = {
        "day": 1,
        "week": 7,
        "quarter": 91.25,   # Approximation of a financial quarter
        "year": 365.25      # Includes leap year adjustment
    }

    # Add sine and cosine features for each cycle
    for name, period in cycles.items():
        df[f"sin_{name}"] = np.sin(2 * np.pi * df[COL_TIME_ORDINAL] / period)
        df[f"cos_{name}"] = np.cos(2 * np.pi * df[COL_TIME_ORDINAL] / period)

    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Moving Averages and EMAs ---
    df[COL_MA_10] = df[COL_CLOSE].rolling(window=10).mean()
    df[COL_MA_30] = df[COL_CLOSE].rolling(window=30).mean()
    df[COL_EMA_10] = df[COL_CLOSE].ewm(span=10, adjust=False).mean()
    df[COL_EMA_30] = df[COL_CLOSE].ewm(span=30, adjust=False).mean()

    # --- Bollinger Bands ---
    rolling_mean = df[COL_CLOSE].rolling(window=20).mean()
    rolling_std = df[COL_CLOSE].rolling(window=20).std()
    df[COL_BB_UPPER] = rolling_mean + 2 * rolling_std
    df[COL_BB_LOWER] = rolling_mean - 2 * rolling_std

    # --- MACD ---
    ema12 = df[COL_CLOSE].ewm(span=12, adjust=False).mean()
    ema26 = df[COL_CLOSE].ewm(span=26, adjust=False).mean()
    df[COL_MACD] = ema12 - ema26
    df[COL_MACD_SIGNAL] = df[COL_MACD].ewm(span=9, adjust=False).mean()

    # --- Momentum, Volatility, Returns ---
    df[COL_MOMENTUM_10] = df[COL_CLOSE] - df[COL_CLOSE].shift(10)
    df[COL_VOLATILITY_20] = df[COL_CLOSE].rolling(window=20).std()
    df[COL_RETURN_1D] = df[COL_CLOSE].pct_change(1)
    df[COL_RETURN_5D] = df[COL_CLOSE].pct_change(5)

    # --- RSI (Smoothed) ---
    delta = df[COL_CLOSE].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df[COL_RSI_14] = 100 - (100 / (1 + rs))

    # --- Clean/Fill: Replace 0s → NaN → forward fill ---
    cols_to_clean = [
        COL_MA_10,
        COL_MA_30,
        COL_MOMENTUM_10,
        COL_VOLATILITY_20,
        COL_RETURN_1D,
        COL_RETURN_5D,
        COL_RSI_14,
        COL_EMA_10,
        COL_EMA_30,
        COL_BB_UPPER,
        COL_BB_LOWER,
        COL_MACD,
        COL_MACD_SIGNAL
    ]

    for col in cols_to_clean:
        df[col].replace(0, pd.NA, inplace=True)
        df[col].fillna(method='ffill', inplace=True)

    # Drop any row where feature column is still NaN
    df = df.dropna(subset=cols_to_clean)

    return df
