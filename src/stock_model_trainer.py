import numpy as np
import pandas as pd
import sys
import os
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *

INPUT_SEQ_LEN = 30
EPOCHS = 50
BATCH_SIZE = 32

# --- Create Sequences ---
def create_sequences(data, input_seq_len, feature_cols, target_cols):
    X, y = [], []
    for i in range(len(data) - input_seq_len):
        x_seq = data.iloc[i:i+input_seq_len][feature_cols].astype(np.float32).values
        y_vals = data.iloc[i+input_seq_len][target_cols].astype(np.float32).values
        X.append(x_seq)
        y.append(y_vals)
    return np.array(X), np.array(y)

# --- Build Model ---
def build_cnn_lstm_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_dim)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --- Plot predictions ---
def plot_predictions(y_true, y_pred, label, n=100):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:n], label=f"True {label}", linewidth=2)
    plt.plot(y_pred[:n], label=f"Predicted {label}", linestyle="--")
    plt.title(f"{label} – True vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main ---
def main():
    df = pd.read_csv("data/stock_data.csv")

    # Remove COL_TIME from input features
    input_features = [col for col in INPUT_FEATURES if col != COL_TIME]

    # Add time-based features
    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    df[COL_MONTH] = df[COL_TIME].dt.month
    df[COL_QUARTER] = df[COL_TIME].dt.quarter
    df[COL_DAYOFWEEK] = df[COL_TIME].dt.dayofweek
    df[COL_TIME_ORDINAL] = df[COL_TIME].map(pd.Timestamp.toordinal)
    input_features += [COL_MONTH, COL_QUARTER, COL_DAYOFWEEK, COL_TIME_ORDINAL]

    # Fill or drop missing values
    df[COL_DIVIDEND_YIELD].fillna(0, inplace=True)
    df[COL_INDEX_PCT_CHANGE].fillna(method='ffill', inplace=True)

    # Drop rows with any remaining NaNs
    df = df.dropna(subset=input_features + TARGET_COLS)

    # Check for NaNs before scaling
    if df[input_features + TARGET_COLS].isnull().values.any():
        raise ValueError("🚨 Data contains NaNs. Please clean the dataset before training.")

    # 1. Scale input features
    scaler_X = MinMaxScaler()
    df[input_features] = scaler_X.fit_transform(df[input_features])

    # 2. Scale target columns — ⚠️ this part is critical
    scaler_y = MinMaxScaler()
    scaler_y.fit(df[TARGET_COLS].values)               # ✅ Fit on original
    df[TARGET_COLS] = scaler_y.transform(df[TARGET_COLS].values)  # ✅ Then transform

    # Create sequences
    X, y = create_sequences(df, INPUT_SEQ_LEN, input_features, TARGET_COLS)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    # Convert to float32 to avoid TensorFlow dtype issues
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)

    # Train model
    model = build_cnn_lstm_model(input_shape=(INPUT_SEQ_LEN, len(input_features)), output_dim=len(TARGET_COLS))
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es])

    # Predict and inverse scale
    predictions = model.predict(X_val)
    predictions_inv = scaler_y.inverse_transform(predictions)
    y_val_inv = scaler_y.inverse_transform(y_val)

    # Plot predictions
    for i, col in enumerate(TARGET_COLS):
        print(f"\n--- MAE for {col} ---")
        mae = mean_absolute_error(y_val_inv[:, i], predictions_inv[:, i])
        print(f"{col}: {mae:.4f}")
        plot_predictions(y_val_inv[:, i], predictions_inv[:, i], label=col, n=100)

    # Save model and artifacts
    os.makedirs("models/cnn_lstm_stock_model", exist_ok=True)
    model.save("models/cnn_lstm_stock_model/cnn_lstm_stock_model.keras")
    print("📊 scaler_y min:", scaler_y.data_min_)
    print("📊 scaler_y max:", scaler_y.data_max_)
    joblib.dump(scaler_X, 'models/cnn_lstm_stock_model/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/cnn_lstm_stock_model/scaler_y.pkl')
    with open("models/cnn_lstm_stock_model/metadata.json", "w") as f:
        json.dump({
            "input_features": input_features,
            "target_cols": TARGET_COLS,
            "input_seq_len": INPUT_SEQ_LEN
        }, f)

    print("\n✅ Model and artifacts saved successfully.")

if __name__ == "__main__":
    main()
