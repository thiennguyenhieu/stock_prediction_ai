# Updated inference script for CNN-LSTM Close Forecasting
import os
import json
import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import date
from tensorflow.keras.models import load_model
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
    target_col = metadata["target_col"]
    seq_len = metadata["input_seq_len"]

    # --- Feature Engineering for Inference ---
    for lag in range(1, 31):
        raw_df[f"lag_close_{lag}"] = raw_df[COL_CLOSE].shift(lag)
    raw_df["delta_close"] = raw_df[COL_CLOSE].diff()
    raw_df.dropna(inplace=True)

    if debug:
        print("\nForecasting", forecast_steps, "steps")
        print("Input shape:", (seq_len, len(input_features)))
        print("Target:", target_col)
        print("Scaler y range:", scaler_y.data_min_, "→", scaler_y.data_max_)

    missing = [f for f in input_features if f not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    if len(raw_df) < seq_len:
        raise ValueError(f"Not enough rows for prediction sequence. Needed: {seq_len}, Got: {len(raw_df)}")

    scaled_df = raw_df.copy()
    scaled_df[input_features] = scaler_X.transform(scaled_df[input_features])
    sequence = scaled_df[input_features].iloc[-seq_len:].values.astype(np.float32)

    predictions = []
    for step in range(forecast_steps):
        X_input = sequence.reshape(1, seq_len, len(input_features))
        y_close, y_dir = model.predict(X_input, verbose=0)

        forecast_value = float(np.expm1(scaler_y.inverse_transform(y_close)[0, 0]))
        predictions.append(forecast_value)

        if debug:
            print(f"\nStep {step + 1}")
            print("  y_scaled:", y_close[0, 0], y_dir[0, 0])
            print("  forecast_close:", forecast_value)

        next_input = sequence[-1].copy()
        if target_col in input_features:
            idx = input_features.index(target_col)
            next_input[idx] = y_close[0, 0]
        sequence = np.vstack([sequence[1:], next_input])

    pred_df = pd.DataFrame(predictions, columns=[COL_CLOSE])
    return pred_df
