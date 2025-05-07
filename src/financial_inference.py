import os
import joblib
import pandas as pd
from src.stock_config import *
from src.fetch_financial_data import fetch_financial_single_symbol

# --- Config ---
VERSION_TAG = "v1_lgbm_eps_bvps"
MODEL_DIR = os.path.join("models", "lgbm_eps_bvps", VERSION_TAG)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler_X.pkl")
MODEL_EPS_PATH = os.path.join(MODEL_DIR, "model_eps.pkl")
MODEL_BVPS_PATH = os.path.join(MODEL_DIR, "model_bvps.pkl")

META_COLS = [COL_SYMBOL, COL_YEAR, COL_QUARTER]
TARGET_COLS = [COL_EPS, COL_BVPS]

# --- Load Artifacts ---
scaler = joblib.load(SCALER_PATH)
model_eps = joblib.load(MODEL_EPS_PATH)
model_bvps = joblib.load(MODEL_BVPS_PATH)

# Load training feature list (in order) from scaler
if hasattr(scaler, 'feature_names_in_'):
    FEATURES = list(scaler.feature_names_in_)
else:
    raise ValueError("Scaler is missing 'feature_names_in_' — ensure it was fitted with feature names.")

# --- Prediction Function ---
def get_eps_bvps_prediction(symbol: str) -> pd.Series:
    df = fetch_financial_single_symbol(symbol)

    if df.empty or df.shape[0] < 1:
        raise ValueError(f"No data available for symbol: {symbol}")

    # Add missing columns with 0.0, and ensure correct order
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df = df[FEATURES]

    # Scale and predict
    X_scaled = scaler.transform(df)
    eps = model_eps.predict(X_scaled)[-1]
    bvps = model_bvps.predict(X_scaled)[-1]

    return pd.Series({COL_EPS: eps, COL_BVPS: bvps})
