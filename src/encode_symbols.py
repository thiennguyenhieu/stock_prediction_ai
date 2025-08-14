import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import LabelEncoder

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fetch_general_info import fetch_all_symbols

def encode_symbols(df: pd.DataFrame, symbol_col: str = None, output_path: str = "data/symbol_encoder.pkl") -> LabelEncoder:
    if symbol_col is None:
        symbol_col = df.columns[0]  # default to the first column

    encoder = LabelEncoder()
    encoder.fit(df[symbol_col])
    joblib.dump(encoder, output_path)
    return encoder

if __name__ == "__main__":
    symbols_with_info = fetch_all_symbols()  # Return [symbol, exchange, organ name]
    encode_symbols(symbols_with_info)
    print("Symbol encoder saved to 'data/symbol_encoder.pkl'")
