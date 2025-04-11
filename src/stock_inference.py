import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.stock_config import *
from src.feature_engineering import add_technical_indicators

# --- Constants ---
MODEL_DIR = "models/cnn_lstm_stock_model"

# --- Load model, scalers, and metadata ---
def load_artifacts():
    model = load_model(os.path.join(MODEL_DIR, "cnn_lstm_stock_model.keras"))
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return model, scaler_X, scaler_y, metadata

# --- Predict trend for future steps ---
def predict_future_trend(raw_df: pd.DataFrame, forecast_steps: int = 5, debug: bool = False) -> pd.DataFrame:
    model, scaler_X, scaler_y, metadata = load_artifacts()
    input_features = metadata["input_features"]
    target_cols = metadata["target_cols"]
    seq_len = metadata["input_seq_len"]

    # --- Time-based + technical feature engineering ---
    if COL_TIME in raw_df.columns:
        raw_df[COL_TIME] = pd.to_datetime(raw_df[COL_TIME])
        raw_df[COL_MONTH] = raw_df[COL_TIME].dt.month
        raw_df[COL_QUARTER] = raw_df[COL_TIME].dt.quarter
        raw_df[COL_DAYOFWEEK] = raw_df[COL_TIME].dt.dayofweek
        raw_df[COL_TIME_ORDINAL] = raw_df[COL_TIME].map(pd.Timestamp.toordinal)

    raw_df = add_technical_indicators(raw_df)

    # --- Validate input features ---
    missing = [f for f in input_features if f not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    raw_df = raw_df.dropna()
    if len(raw_df) < seq_len:
        raise ValueError(f"Not enough data to form sequence (need {seq_len}, got {len(raw_df)})")

    # --- Scale input features ---
    scaled_df = raw_df.copy()
    scaled_df[input_features] = scaler_X.transform(scaled_df[input_features])
    sequence = scaled_df[input_features].iloc[-seq_len:].values.astype(np.float32)

    predictions = []

    if debug:
        print(f"\nForecasting {forecast_steps} steps")
        print(f"Targets: {target_cols}")
        print(f"Input shape: {sequence.shape}")
        print("scaler_y range:", scaler_y.data_min_, "→", scaler_y.data_max_)

    # --- Forecast loop ---
    for step in range(forecast_steps):
        X_input = sequence.reshape(1, seq_len, len(input_features))
        y_scaled = model.predict(X_input, verbose=0)
        y = scaler_y.inverse_transform(y_scaled)[0]
        predictions.append(y)

        if debug:
            print(f"\nStep {step}")
            print("Scaled:", y_scaled)
            print("Decoded:", y)

        # Update sequence autoregressively using prediction
        next_input = sequence[-1].copy()
        for j, col in enumerate(target_cols):
            if col in input_features:
                col_idx = input_features.index(col)
                next_input[col_idx] = y_scaled[0, j]  # insert scaled prediction into input

        sequence = np.vstack([sequence[1:], next_input])

    # --- Format result ---
    pred_df = pd.DataFrame(predictions, columns=target_cols)
    pred_df.index.name = "Step"

    if debug:
        print("\nFinal Predictions:")
        print(pred_df)

    return pred_df