# Updated inference script for CNN-LSTM Close Forecasting
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import date
from tensorflow.keras.models import load_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *
from src.fetch_historical_data import process_symbol, post_process_data

# --- Constants ---
VERSION_TAG = "v1_close_regression"
MODEL_DIR = os.path.join("models", "cnn_lstm_close_regression", VERSION_TAG)

# --- Load model, scalers, and metadata ---
def load_artifacts():
    model = load_model(os.path.join(MODEL_DIR, "cnn_lstm_forecast_model.keras"))
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return model, scaler_X, scaler_y, metadata

# --- Predict close price series ---
def get_close_prediction(symbol: str, interval: int) -> pd.DataFrame:
    start_date = (date.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
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

def predict_close_price_series(raw_df: pd.DataFrame, forecast_steps: int = 5, debug: bool = False) -> pd.DataFrame:
    model, scaler_X, scaler_y, metadata = load_artifacts()
    input_features = metadata["input_features"]
    seq_len = metadata["input_seq_len"]

    # --- Feature Engineering for Inference (RAW space) ---
    df_raw = raw_df.copy()
    for lag in range(1, 31):
        df_raw[f"lag_close_{lag}"] = df_raw[COL_CLOSE].shift(lag)
    df_raw["delta_close"] = df_raw[COL_CLOSE].diff()
    df_raw.dropna(inplace=True)

    missing = [f for f in input_features if f not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    if len(df_raw) < seq_len:
        raise ValueError(f"Not enough rows: need {seq_len}, have {len(df_raw)}")

    # --- Make scaled window for the model ---
    df_scaled = df_raw.copy()
    df_scaled[input_features] = scaler_X.transform(df_scaled[input_features])
    sequence = df_scaled[input_features].iloc[-seq_len:].values.astype(np.float32)

    # We'll also keep the last UNscaled feature row to update lags/delta each step
    last_raw_row = df_raw[input_features].iloc[-1].copy()

    predictions = []
    for step in range(forecast_steps):
        X_input = sequence.reshape(1, seq_len, len(input_features))
        y_close_scaled, _ = model.predict(X_input, verbose=0)  # y_close_scaled shape: (1, 1) for v1 head step0
        # Inverse target scaling to RAW close (for output)
        y_close_log = scaler_y.inverse_transform(y_close_scaled)[0, 0]
        y_pred_raw = float(np.expm1(y_close_log))
        predictions.append(y_pred_raw)

        # ----- Update lag features in RAW space -----
        # shift lag_close_k: lag_k <- lag_{k-1}, and set lag_close_1 = predicted_close
        for lag in range(30, 1, -1):
            col = f"lag_close_{lag}"
            prev_col = f"lag_close_{lag-1}"
            if col in last_raw_row and prev_col in last_raw_row:
                last_raw_row[col] = last_raw_row[prev_col]
        if "lag_close_1" in last_raw_row:
            last_raw_row["lag_close_1"] = y_pred_raw

        # recompute delta_close = lag1 - lag2 if both exist
        if "delta_close" in last_raw_row and "lag_close_1" in last_raw_row and "lag_close_2" in last_raw_row:
            last_raw_row["delta_close"] = last_raw_row["lag_close_1"] - last_raw_row["lag_close_2"]

        # scale the UPDATED raw row to build next sequence row
        new_row_scaled = scaler_X.transform(pd.DataFrame([last_raw_row], columns=input_features))[0]
        sequence = np.vstack([sequence[1:], new_row_scaled.astype(np.float32)])

        if debug:
            print(f"step {step+1}: pred={y_pred_raw:.4f}, lag1(raw)={last_raw_row.get('lag_close_1', None)}")

    return pd.DataFrame(predictions, columns=[COL_CLOSE])

if __name__ == "__main__":
    print(get_close_prediction("SHB", 14))
    print(get_close_prediction("DRI", 14))
    print(get_close_prediction("HPG", 14))