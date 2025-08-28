# src/historical_inference_v2_returns.py
# Inference for 14-day direct multi-horizon model trained on RETURNS.
# - Loads with compile=False (no custom loss needed)
# - Rebuilds features, predicts returns vector, inverse-transforms,
#   then reconstructs PRICE path by compounding from the last observed close.
# - Optional near-term scale anchoring (kept light; often unnecessary with returns)

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import date
from tensorflow.keras.models import load_model

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *
from src.fetch_historical_data import process_symbol, post_process_data

# ----- CONFIG -----
VERSION_TAG = "v2_returns_h14_noleak"   # must match training folder above
MODEL_DIR = os.path.join("models", "cnn_lstm_close_regression", VERSION_TAG)

ANCHOR_STRENGTH = 0.0  # returns-based models usually don't need this; set to >0 to enable scale anchoring

# ----- HELPERS -----
def _ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if COL_TIME not in out.columns:
        for cand in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
            if cand in out.columns:
                out = out.rename(columns={cand: COL_TIME})
                break
        else:
            if np.issubdtype(out.index.dtype, np.datetime64):
                out = out.reset_index().rename(columns={"index": COL_TIME})
            if COL_TIME not in out.columns:
                out[COL_TIME] = np.arange(len(out), dtype=np.int64)
    out = out.sort_values(COL_TIME).reset_index(drop=True)
    return out

def _build_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_time_column(df)
    # parity with training: delta + lags
    df["delta_close"] = df[COL_CLOSE].diff()
    for lag in range(1, 31):
        df[f"lag_close_{lag}"] = df[COL_CLOSE].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

# ----- LOADING -----
def load_artifacts():
    model = load_model(os.path.join(MODEL_DIR, "cnn_lstm_forecast_model.keras"), compile=False)
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    scaler_ret = joblib.load(os.path.join(MODEL_DIR, "scaler_ret.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return model, scaler_X, scaler_ret, metadata

# ----- CORE PREDICTION -----
def predict_close_price_series(raw_df: pd.DataFrame, forecast_steps: int = 14, debug: bool = False) -> pd.DataFrame:
    model, scaler_X, scaler_ret, metadata = load_artifacts()

    input_features = metadata["input_features"]
    H = int(metadata["forecast_horizon"])
    seq_len = int(metadata["input_seq_len"])
    if forecast_steps > H:
        raise ValueError(f"Requested {forecast_steps} steps, but model horizon is {H}.")

    df = _build_features_for_inference(raw_df.copy())

    # feature presence
    missing = [c for c in input_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input features for inference: {missing}")
    if len(df) < seq_len:
        raise ValueError(f"Not enough rows for prediction sequence. Need {seq_len}, have {len(df)}")

    # scale inputs
    scaled = df.copy()
    scaled[input_features] = scaler_X.transform(scaled[input_features])
    X_seq = scaled[input_features].iloc[-seq_len:].values.astype(np.float32).reshape(1, seq_len, len(input_features))

    # forward pass -> scaled returns vector
    y_rets_scaled, _ = model.predict(X_seq, verbose=0)      # shape: (1, H)
    # inverse-transform returns
    y_rets = scaler_ret.inverse_transform(y_rets_scaled)[0] # (H,)

    # reconstruct price path
    last_close = float(raw_df[COL_CLOSE].iloc[-1])
    last_logp  = np.log(last_close)
    log_path   = last_logp + np.cumsum(y_rets[:forecast_steps])
    y_close    = np.exp(log_path)

    # OPTIONAL: light scale anchoring (usually unnecessary with returns)
    if ANCHOR_STRENGTH and ANCHOR_STRENGTH > 0:
        eps = 1e-8
        scale_t1 = last_close / max(y_close[0], eps)
        scale_t1 = float(np.clip(scale_t1, 0.6, 1.6))
        decay = np.linspace(1.0, 0.0, forecast_steps, dtype=np.float32)
        decay = np.power(decay, 0.7)
        scales = np.power(scale_t1, ANCHOR_STRENGTH * decay)
        y_close = y_close * scales
        y_close = np.maximum(y_close, 0.0)

    return pd.DataFrame({COL_CLOSE: y_close})

# ----- PUBLIC API -----
def get_close_prediction(symbol: str, interval: int = 14) -> pd.DataFrame:
    start_date = (date.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")

    df = process_symbol(symbol, start_date, today)
    if df is None or df.empty:
        print(f"[WARN] Empty data for {symbol}")
        return pd.DataFrame()

    df = post_process_data(df)
    df = _ensure_time_column(df)

    if len(df) < 240:
        print(f"[WARN] Not enough data for {symbol}. Need more rows for seq_len & features.")
        return pd.DataFrame()

    try:
        forecast = predict_close_price_series(df, forecast_steps=interval, debug=False)
        dates = pd.date_range(start=date.today() + BDay(1), periods=interval, freq=BDay())
        forecast.insert(0, COL_TIME, dates)
        return forecast
    except Exception as e:
        print(f"[ERROR] Prediction failed for {symbol}: {e}")
        return pd.DataFrame()

# ----- CLI -----
if __name__ == "__main__":
    out = get_close_prediction("SHB", interval=14)
    print(out)
