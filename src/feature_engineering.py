import pandas as pd
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators such as MA, RSI, momentum, volatility, etc.
    Requires 'close' and 'volume' columns to be present in the dataframe.
    """
    df = df.copy()

    # Moving Averages
    df[COL_MA_10] = df[COL_CLOSE].rolling(window=10).mean()
    df[COL_MA_30] = df[COL_CLOSE].rolling(window=30).mean()

    # Momentum
    df[COL_MOMENTUM_10] = df[COL_CLOSE] - df[COL_CLOSE].shift(10)

    # Price-to-Volume Ratio
    df[COL_PRICE_VOL_RATIO] = df[COL_CLOSE] / (df[COL_VOLUME] + 1e-6)

    # Volatility (Standard Deviation)
    df[COL_VOLATILITY_20] = df[COL_CLOSE].rolling(window=20).std()

    # Returns
    df[COL_RETURN_1D] = df[COL_CLOSE].pct_change(1)
    df[COL_RETURN_5D] = df[COL_CLOSE].pct_change(5)

    # Relative Strength Index (RSI) - simple version
    delta = df[COL_CLOSE].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df[COL_RSI_14] = 100 - (100 / (1 + rs))

    # Drop rows with NaNs caused by rolling/shift operations
    df = df.dropna()

    return df
