# inference_cnn_lstm_v2.py
# Inference for CNN‑BiLSTM v2 (14‑day horizon, leak‑fixed training)
# - Loads artifacts from models/cnn_lstm_close_regression/v2_close_regression_leakfix
# - Rebuilds simple features used during training (delta_close, lag_close_1..30) if missing
# - Single‑shot multi‑horizon prediction, proper inverse transform (MinMax + expm1)
# - Returns ONLY [time, close] for UI compatibility
# - Optional ±7% per‑day cap applied after denorm (enabled by default)

import os
import json
import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import tensorflow as tf
from tensorflow.keras.models import load_model

# Project imports
from src.constants import COL_TIME, COL_CLOSE
from src.fetch_historical_data import process_symbol, post_process_data

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
VERSION_TAG = "v2_close_regression_leakfix"
MODEL_DIR = os.path.join("models", "cnn_lstm_close_regression", VERSION_TAG)

# ---------------------------------------------------------------------
# Artifacts loader
# ---------------------------------------------------------------------
def load_artifacts(model_dir: str = MODEL_DIR):
    model = load_model(os.path.join(model_dir, "cnn_lstm_forecast_model.keras"))
    scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return model, scaler_X, scaler_y, metadata

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _inv_minmax_expm1(vec: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse MinMax for a 1D vector (H,), then expm1. Returns shape (H,).
    """
    v = vec.reshape(-1, 1)
    inv = scaler.inverse_transform(v)
    inv = np.expm1(inv)
    return inv.ravel()

def _cap_daily_moves_series(pred_close: np.ndarray, anchor_last_close: float, cap: float = 0.07) -> np.ndarray:
    """
    Enforce per-step cap on relative change vs previous step.
    pred_close is in price space; anchor is last real close before forecast.
    """
    if pred_close.size == 0:
        return pred_close
    out = np.empty_like(pred_close, dtype=float)
    prev = float(anchor_last_close)
    lo_mul, hi_mul = 1.0 - cap, 1.0 + cap
    for t in range(len(pred_close)):
        unclamped = float(pred_close[t])
        clamped = min(max(unclamped, prev * lo_mul), prev * hi_mul)
        out[t] = clamped
        prev = clamped
    return out

# ---------------------------------------------------------------------
# Core inference (single-shot multi-horizon)
# ---------------------------------------------------------------------
def predict_close_price_series(
    raw_df: pd.DataFrame,
    forecast_steps: int = 14,
    apply_cap: bool = True,
    cap_pct: float = 0.07,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame with a single column 'close' (length = forecast_steps).
    """
    model, scaler_X, scaler_y, metadata = load_artifacts()
    input_features = metadata["input_features"]
    seq_len = int(metadata["input_seq_len"])

    df = raw_df.copy()
    df[COL_TIME] = pd.to_datetime(df[COL_TIME])

    # Rebuild the simple engineered features used in training (if missing)
    if "delta_close" not in df.columns:
        df["delta_close"] = df[COL_CLOSE].diff()
    for lag in range(1, 31):
        col = f"lag_close_{lag}"
        if col not in df.columns:
            df[col] = df[COL_CLOSE].shift(lag)

    df.dropna(inplace=True)

    # Validations
    missing = [f for f in input_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing input features required by the model: {missing}")
    if len(df) < seq_len:
        raise ValueError(f"Not enough rows for prediction sequence. Needed: {seq_len}, Got: {len(df)}")

    # Build input window (1, seq_len, n_features) with the same scaler_X used in training
    X_scaled_full = scaler_X.transform(df[input_features].to_numpy(dtype=np.float32))
    X_seq = X_scaled_full[-seq_len:]
    X_input = X_seq.reshape(1, seq_len, len(input_features))

    # Forward pass: model outputs (1, H) for close and (1, H) for direction
    y_close_scaled, _y_dir_prob = model.predict(X_input, verbose=0)
    y_close_scaled = y_close_scaled.reshape(-1)[:forecast_steps]  # (H,)

    # Inverse target scaling (MinMax) then expm1 to price space
    y_close = _inv_minmax_expm1(y_close_scaled, scaler_y)  # (H,)

    # Optional ±7% cap (relative to previous step, starting at last real close)
    last_close = float(df[COL_CLOSE].iloc[-1])
    if apply_cap:
        y_close = _cap_daily_moves_series(y_close, anchor_last_close=last_close, cap=cap_pct)

    return pd.DataFrame({COL_CLOSE: y_close})

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_close_prediction(symbol: str, interval: int = 14, apply_cap: bool = True, cap_pct: float = 0.07) -> pd.DataFrame:
    """
    Convenience entrypoint: fetches recent data, runs v2 inference,
    and returns ONLY [time, close] for the next `interval` business days.
    """
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=2)

    df = process_symbol(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return pd.DataFrame(columns=[COL_TIME, COL_CLOSE])

    df = post_process_data(df)
    if df is None or df.empty:
        return pd.DataFrame(columns=[COL_TIME, COL_CLOSE])

    try:
        pred_df = predict_close_price_series(
            df,
            forecast_steps=interval,
            apply_cap=apply_cap,
            cap_pct=cap_pct,
            debug=False,
        )

        # Forecast dates: next business days after the last real date in df
        last_dt = pd.to_datetime(df[COL_TIME].iloc[-1])
        forecast_dates = pd.date_range(start=last_dt + BDay(1), periods=len(pred_df), freq=BDay())

        pred_df.insert(0, COL_TIME, forecast_dates)
        return pred_df[[COL_TIME, COL_CLOSE]].reset_index(drop=True)

    except Exception as e:
        print(f"[ERROR] v2 inference failed for {symbol}: {e}")
        return pd.DataFrame(columns=[COL_TIME, COL_CLOSE])
