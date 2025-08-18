# CNN-LSTM Multi-Step Return Forecasting (H=14, per-day cap ±7%)
import numpy as np
import pandas as pd
import joblib
import sys
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, LayerNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)

INPUT_SEQ_LEN = 180
FORECAST_HORIZON = 14
EPOCHS = 50
BATCH_SIZE = 32
VERSION_TAG = "v2_regression_H14"
MODEL_OUTPUT_DIR = os.path.join("models", "cnn_lstm_close_regression", VERSION_TAG)

# daily cap constants
MAX_DAILY_CHANGE = 0.07
MAX_LOG_RET = float(np.log1p(MAX_DAILY_CHANGE))

class SequenceGenerator(Sequence):
    """
    Yields:
      X: (batch, T, F) features
      y: (batch, H) next-H log-returns (clipped to ±log(1.07))
    """
    def __init__(self, df, input_seq_len, feature_cols, target_cols, batch_size=32):
        self.df = df
        self.input_seq_len = input_seq_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols  # ['target_ret_t1', ..., 'target_ret_tH']
        self.batch_size = batch_size

        self.data = df[feature_cols + target_cols].values.astype(np.float32)
        self.n_features = len(feature_cols)
        self.indices = np.arange(len(self.data) - input_seq_len)

        # for evaluation (price reconstruction)
        self.close_raw = df["close_raw"].values.astype(np.float32)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(batch_indices), self.input_seq_len, self.n_features), dtype=np.float32)
        y = np.zeros((len(batch_indices), len(self.target_cols)), dtype=np.float32)

        for i, start_idx in enumerate(batch_indices):
            X[i] = self.data[start_idx:start_idx + self.input_seq_len, :self.n_features]
            label_row = start_idx + self.input_seq_len
            y[i] = self.data[label_row, self.n_features:]

        return X, y

def build_model(input_shape, forecast_horizon):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    attention = Dense(1, activation='tanh')(x)
    attention_weights = tf.nn.softmax(attention, axis=1)
    context = tf.reduce_sum(x * attention_weights, axis=1)

    context = Dense(64, activation='relu')(context)
    context = Dropout(0.2)(context)
    context = LayerNormalization()(context)

    forecast_output = Dense(forecast_horizon, name="forecast_close")(context)  # returns

    model = Model(inputs, forecast_output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=Huber(),
        metrics=["mae"]
    )
    return model

def main():
    df = pd.read_csv('data/historical_data_final.csv')

    # keep raw close for price reconstruction
    df["close_raw"] = df[COL_CLOSE].astype(float)

    # build log-return targets (clipped)
    logp = np.log(df[COL_CLOSE].astype(float))
    for i in range(1, FORECAST_HORIZON + 1):
        r = logp.shift(-i) - logp
        df[f"target_ret_t{i}"] = np.clip(r, -MAX_LOG_RET, MAX_LOG_RET)

    # simple features
    df["delta_close"] = df[COL_CLOSE].diff()
    for lag in range(1, 31):
        df[f"lag_close_{lag}"] = df[COL_CLOSE].shift(lag)

    df.dropna(inplace=True)

    # targets & features
    target_cols = [f"target_ret_t{i}" for i in range(1, FORECAST_HORIZON + 1)]
    input_features = [c for c in df.columns if c not in target_cols]

    # robust scale features only
    scaler_X = RobustScaler()
    df[input_features] = scaler_X.fit_transform(df[input_features])

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # temporal split
    train_size = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:train_size].copy(), df.iloc[train_size:].copy()

    train_gen = SequenceGenerator(df_train, INPUT_SEQ_LEN, input_features, target_cols, batch_size=BATCH_SIZE)
    val_gen   = SequenceGenerator(df_val,   INPUT_SEQ_LEN, input_features, target_cols, batch_size=BATCH_SIZE)

    model = build_model((INPUT_SEQ_LEN, len(input_features)), FORECAST_HORIZON)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[ReduceLROnPlateau(factor=0.5, patience=10), EarlyStopping(patience=20, restore_best_weights=True)],
        verbose=1
    )

    # save curves
    with open(os.path.join(MODEL_OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump({k: [float(vv) for vv in v] for k, v in history.history.items()}, f, indent=4)

    # plot regression curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Train Return MAE')
    plt.plot(history.history['val_mae'], label='Val Return MAE')
    plt.title("Forecast Return MAE (per-step, clipped ±7%)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "training_history.png")); plt.close()

    # --- Evaluate: reconstruct price paths; cap predicted returns to ±log(1.07)
    y_true_ret, y_pred_ret, base_prices = [], [], []

    for i in range(len(val_gen)):
        batch_indices = val_gen.indices[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        X_batch, y_batch = val_gen[i]
        preds = model.predict(X_batch, verbose=0)

        y_true_ret.append(y_batch)
        y_pred_ret.append(np.clip(preds, -MAX_LOG_RET, MAX_LOG_RET))
        bp = val_gen.close_raw[batch_indices + INPUT_SEQ_LEN - 1]
        base_prices.append(bp)

    y_true_ret = np.concatenate(y_true_ret, axis=0)
    y_pred_ret = np.concatenate(y_pred_ret, axis=0)
    base_prices = np.concatenate(base_prices, axis=0)

    def returns_to_prices(base, rets):
        cum = np.cumsum(rets, axis=1)
        return (base[:, None] * np.exp(cum)).astype(np.float32)

    y_true_close = returns_to_prices(base_prices, y_true_ret)
    y_pred_close = returns_to_prices(base_prices, y_pred_ret)

    mae = float(round(mean_absolute_error(y_true_close.flatten(), y_pred_close.flatten()), 4))
    with open(os.path.join(MODEL_OUTPUT_DIR, "forecast_mae.json"), "w") as f:
        json.dump({"forecast_close_mae": mae}, f, indent=4)

    with open(os.path.join(MODEL_OUTPUT_DIR, "training_summary.txt"), "w") as f:
        f.write(f"Best Train Return MAE: {min(history.history['mae']):.4f}\n")
        f.write(f"Best Val Return MAE: {min(history.history['val_mae']):.4f}\n")
        f.write(f"Reconstructed Price MAE: {mae:.4f}\n")

    # save model & scaler
    model.save(os.path.join(MODEL_OUTPUT_DIR, "cnn_lstm_forecast_model.keras"))
    joblib.dump(scaler_X, os.path.join(MODEL_OUTPUT_DIR, "scaler_X.pkl"))

    with open(os.path.join(MODEL_OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "input_features": input_features,
            "target_cols": target_cols,
            "input_seq_len": INPUT_SEQ_LEN,
            "forecast_horizon": FORECAST_HORIZON,
            "version": VERSION_TAG,
            "target_type": "log_returns",
            "max_daily_change": MAX_DAILY_CHANGE
        }, f, indent=4)

    # quick plots at a few horizons (price space)
    for i in [1, 7, 14]:
        if i <= y_true_close.shape[1]:
            plt.figure(figsize=(10, 4))
            plt.plot(y_true_close[:, i - 1], label=f'True Price t+{i}')
            plt.plot(y_pred_close[:, i - 1], label=f'Predicted Price t+{i}')
            plt.title(f"Reconstructed Close Price (t+{i}) - cap ±7%/day")
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f"forecast_t{i}.png")); plt.close()

if __name__ == "__main__":
    main()
