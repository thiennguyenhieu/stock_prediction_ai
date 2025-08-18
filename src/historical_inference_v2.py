# Updated inference script for CNN-LSTM Return Forecasting (H=14, ±7% cap)
import os
import json
import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import date
from tensorflow.keras.models import load_model
from src.stock_config import *
from src.fetch_historical_data import process_symbol, post_process_data

# --- Constants ---
VERSION_TAG = "v2_regression_H14"  # match training tag
MODEL_DIR = os.path.join("models", "cnn_lstm_close_regression", VERSION_TAG)

# daily cap (±7%) in log-return space
MAX_DAILY_CHANGE = 0.07
MAX_LOG_RET = float(np.log1p(MAX_DAILY_CHANGE))

# --- Load model, scaler, and metadata ---
def load_artifacts():
    model = load_model(os.path.join(MODEL_DIR, "cnn_lstm_forecast_model.keras"))
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return model, scaler_X, metadata

# --- Predict close price series (multi-step in one forward pass) ---
def get_close_prediction(symbol: str, interval: int) -> pd.DataFrame:
    # trained for 14 steps; allow any interval <= 14
    if interval > 14:
        raise ValueError("Model was trained for 14 steps. Please request interval <= 14.")

    start_date = (date.today() - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")

    df = process_symbol(symbol, start_date, today)
    if df.empty:
        return pd.DataFrame()

    df = post_process_data(df)
    if len(df) < 30:
        print(f"[WARN] Not enough data for {symbol}")
        return pd.DataFrame()

    try:
        forecast = predict_close_price_series(df, forecast_steps=interval, debug=False)
        forecast_dates = pd.date_range(start=date.today() + BDay(1), periods=interval, freq=BDay())
        forecast.insert(0, COL_TIME, forecast_dates)
        return forecast
    except Exception as e:
        print(f"[ERROR] Prediction failed for {symbol}: {e}")
        return pd.DataFrame()

def predict_close_price_series(raw_df: pd.DataFrame, forecast_steps: int = 14, debug: bool = False) -> pd.DataFrame:
    model, scaler_X, metadata = load_artifacts()
    input_features = metadata["input_features"]
    seq_len = metadata["input_seq_len"]
    horizon = metadata["forecast_horizon"]

    if forecast_steps > horizon:
        raise ValueError(f"Requested {forecast_steps} steps, but model horizon is {horizon}.")

    # --- Feature Engineering for Inference (mirror training) ---
    for lag in range(1, 31):
        raw_df[f"lag_close_{lag}"] = raw_df[COL_CLOSE].shift(lag)
    raw_df["delta_close"] = raw_df[COL_CLOSE].diff()
    raw_df.dropna(inplace=True)

    # last observed close (base price for reconstruction)
    last_close = float(raw_df[COL_CLOSE].iloc[-1])

    # ensure we have all features
    missing = [f for f in input_features if f not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    if len(raw_df) < seq_len:
        raise ValueError(f"Not enough rows for prediction sequence. Needed: {seq_len}, Got: {len(raw_df)}")

    # scale features only (no y-scaling)
    scaled_df = raw_df.copy()
    scaled_df[input_features] = scaler_X.transform(scaled_df[input_features])
    sequence = scaled_df[input_features].iloc[-seq_len:].values.astype(np.float32)

    # Single forward pass predicts the next H **returns** (not prices)
    X_input = sequence.reshape(1, seq_len, len(input_features))
    y_ret_pred = model.predict(X_input, verbose=0)  # shape: (1, H)

    # clip per-day returns to ±log(1.07) just like training
    y_ret_pred = np.clip(y_ret_pred, -MAX_LOG_RET, MAX_LOG_RET)

    # take only the number of steps requested
    ret_path = y_ret_pred[0, :forecast_steps]  # (forecast_steps,)

    # reconstruct price path from log-returns:
    # P_{t+k} = P_t * exp(sum_{j=1..k} r_j)
    cum_ret = np.cumsum(ret_path)
    price_path = last_close * np.exp(cum_ret)

    if debug:
        print(f"Base price: {last_close:.4f}")
        print(f"Pred returns (first 5): {ret_path[:5]}")
        print(f"Pred prices  (first 5): {price_path[:5]}")

    # output as DataFrame (include returns for debugging/analysis if you wish)
    pred_df = pd.DataFrame({
        "predicted_return": ret_path,
        COL_CLOSE: price_path
    })
    return pred_df
