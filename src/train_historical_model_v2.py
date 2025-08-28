# src/train_historical_model_v2_returns.py
# CNN-LSTM for 14-day forecasting using MULTI-STEP LOG-RETURNS as target.
# - Fits scalers on TRAIN ONLY (no leakage)
# - Targets: ret_t1..ret_tH (StandardScaler)
# - Direction labels from returns > 0 (same horizon)
# - Saves raw-space MAE per horizon by reconstructing prices from returns

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, LayerNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# ----- Project imports -----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *

# GPU growth
for dev in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(dev, True)
    except Exception:
        pass

# ----- Config -----
INPUT_SEQ_LEN = 180
FORECAST_HORIZON = 14
EPOCHS = 50
BATCH_SIZE = 32
VERSION_TAG = "v2_returns_h14_noleak"
MODEL_OUTPUT_DIR = os.path.join("models", "cnn_lstm_close_regression", VERSION_TAG)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ----- Data Generator -----
class SequenceGenerator(Sequence):
    """
    Generates (X, y) pairs for multi-step forecasting.
    y has two heads:
      - 'forecast_close': here means returns target (ret_t1..ret_tH) scaled -> shape (batch, H)
      - 'direction_classifier': binary (ret > 0) -> shape (batch, H)
    """
    def __init__(self, df_scaled, input_seq_len, feature_cols, ret_cols, dir_cols, batch_size=32):
        self.df = df_scaled
        self.input_seq_len = input_seq_len
        self.feature_cols = feature_cols
        self.ret_cols = ret_cols
        self.dir_cols = dir_cols
        self.batch_size = batch_size

        self.n_features = len(feature_cols)
        # indices produce windows ending at t and targets t+1..t+H
        self.indices = np.arange(len(self.df) - input_seq_len - len(ret_cols) + 1)

        sel = self.feature_cols + self.ret_cols + self.dir_cols
        self.data = self.df[sel].values.astype(np.float32)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        bi = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(bi), self.input_seq_len, self.n_features), dtype=np.float32)
        y_ret = np.zeros((len(bi), len(self.ret_cols)), dtype=np.float32)
        y_dir = np.zeros((len(bi), len(self.dir_cols)), dtype=np.float32)

        for i, s in enumerate(bi):
            X[i] = self.data[s:s + self.input_seq_len, :self.n_features]
            y_ret[i] = self.data[s + self.input_seq_len:s + self.input_seq_len + len(self.ret_cols), self.n_features:self.n_features + len(self.ret_cols)]
            y_dir[i] = self.data[s + self.input_seq_len:s + self.input_seq_len + len(self.dir_cols), self.n_features + len(self.ret_cols):]

        return X, {"forecast_close": y_ret, "direction_classifier": y_dir}

# ----- Model -----
def build_model(input_shape, forecast_horizon):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    # simple attention
    att = Dense(1, activation='tanh')(x)
    att_w = tf.nn.softmax(att, axis=1)
    context = tf.reduce_sum(x * att_w, axis=1)

    context = Dense(64, activation='relu')(context)
    context = Dropout(0.2)(context)
    context = LayerNormalization()(context)

    out_ret = Dense(forecast_horizon, name="forecast_close")(context)  # returns
    out_dir = Dense(forecast_horizon, name="direction_classifier", activation="sigmoid")(context)

    model = Model(inputs, [out_ret, out_dir])
    model.compile(
        optimizer=Adam(1e-3),
        loss={"forecast_close": Huber(), "direction_classifier": "binary_crossentropy"},
        loss_weights={"forecast_close": 1.0, "direction_classifier": 0.5},
        metrics={"forecast_close": "mae", "direction_classifier": "accuracy"},
    )
    return model

# ----- Training pipeline -----
def main():
    df = pd.read_csv('data/historical_data_final.csv')
    df = df.sort_values(COL_TIME).reset_index(drop=True)

    # keep a raw close column for later reconstruction
    df["close_raw"] = df[COL_CLOSE].astype(float)
    df["logp"] = np.log(df[COL_CLOSE].astype(float))

    H = FORECAST_HORIZON

    # Multi-step returns (log space): log(C_{t+h}) - log(C_t)
    ret_cols = []
    for h in range(1, H + 1):
        col = f"ret_t{h}"
        df[col] = df["logp"].shift(-h) - df["logp"]
        ret_cols.append(col)

    # Direction labels per horizon (returns > 0)
    dir_cols = []
    for h in range(1, H + 1):
        col = f"direction_t{h}"
        df[col] = (df[f"ret_t{h}"] > 0).astype(int)
        dir_cols.append(col)

    # Feature engineering (parity with your previous setup)
    df["delta_close"] = df[COL_CLOSE].diff()
    for lag in range(1, 31):
        df[f"lag_close_{lag}"] = df[COL_CLOSE].shift(lag)

    # Drop NA from shifts/targets
    df = df.dropna().reset_index(drop=True)

    # Input features exclude all targets/labels
    input_features = [c for c in df.columns if c not in ["close_raw", "logp"] + ret_cols + dir_cols]

    # Chronological split (train-only fit)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()
    df_val   = df.iloc[train_size:].copy()

    # Fit scalers on TRAIN ONLY
    scaler_X = MinMaxScaler()
    scaler_ret = StandardScaler()

    df_train[input_features] = scaler_X.fit_transform(df_train[input_features])
    df_val[input_features]   = scaler_X.transform(df_val[input_features])

    df_train[ret_cols] = scaler_ret.fit_transform(df_train[ret_cols])
    df_val[ret_cols]   = scaler_ret.transform(df_val[ret_cols])

    # Save scalers
    joblib.dump(scaler_X, os.path.join(MODEL_OUTPUT_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_ret, os.path.join(MODEL_OUTPUT_DIR, "scaler_ret.pkl"))

    # Generators
    train_gen = SequenceGenerator(df_train, INPUT_SEQ_LEN, input_features, ret_cols, dir_cols, batch_size=BATCH_SIZE)
    val_gen   = SequenceGenerator(df_val,   INPUT_SEQ_LEN, input_features, ret_cols, dir_cols, batch_size=BATCH_SIZE)

    # Model
    model = build_model((INPUT_SEQ_LEN, len(input_features)), H)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[ReduceLROnPlateau(factor=0.5, patience=10), EarlyStopping(patience=20, restore_best_weights=True)],
        verbose=1
    )

    # Save training curves (note: losses/mae are on scaled returns)
    with open(os.path.join(MODEL_OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump({k: [float(vv) for vv in v] for k, v in history.history.items()}, f, indent=2)

    # ----- Offline evaluation in RAW PRICE space -----
    # We reconstruct price paths using the last close at the end of each input window.

    # Recreate the same window indices used by the val generator
    idxs = np.arange(len(df_val) - INPUT_SEQ_LEN - H + 1)
    n_batches = int(np.ceil(len(idxs) / BATCH_SIZE))

    y_true_paths = []
    y_pred_paths = []
    y_true_dir_all = []
    y_pred_dir_logits_all = []

    for b in range(n_batches):
        bi = idxs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        if len(bi) == 0:
            continue

        # X batch (scaled)
        X = np.zeros((len(bi), INPUT_SEQ_LEN, len(input_features)), dtype=np.float32)

        # collect ground truth returns (scaled) and directions for these windows
        true_rets_scaled = np.zeros((len(bi), H), dtype=np.float32)
        true_dirs = np.zeros((len(bi), H), dtype=np.float32)
        last_closes = np.zeros((len(bi),), dtype=np.float32)

        for i, s in enumerate(bi):
            rows = df_val.iloc[s:s + INPUT_SEQ_LEN]
            X[i] = rows[input_features].values.astype(np.float32)

            # ground truth returns for this window (scaled)
            tgt_rows = df_val.iloc[s + INPUT_SEQ_LEN:s + INPUT_SEQ_LEN + H]
            true_rets_scaled[i] = tgt_rows[ret_cols].values.astype(np.float32).ravel()
            true_dirs[i] = tgt_rows[dir_cols].values.astype(np.float32).ravel()

            # last close at t (end of input window)
            last_closes[i] = float(df_val.iloc[s + INPUT_SEQ_LEN - 1]["close_raw"])

        # predict scaled returns + direction logits
        pred_rets_scaled, pred_dir_logits = model.predict(X, verbose=0)

        # inverse-transform returns
        true_rets = scaler_ret.inverse_transform(true_rets_scaled)
        pred_rets = scaler_ret.inverse_transform(pred_rets_scaled)

        # reconstruct prices from returns
        # log path: log(C_t) + cumsum(ret_t+1..t+H)
        last_logp = np.log(last_closes)
        true_log_paths = last_logp[:, None] + np.cumsum(true_rets, axis=1)
        pred_log_paths = last_logp[:, None] + np.cumsum(pred_rets, axis=1)
        true_prices = np.exp(true_log_paths)
        pred_prices = np.exp(pred_log_paths)

        y_true_paths.append(true_prices)
        y_pred_paths.append(pred_prices)
        y_true_dir_all.append(true_dirs)
        y_pred_dir_logits_all.append(pred_dir_logits)

    if len(y_true_paths):
        y_true_paths = np.vstack(y_true_paths)
        y_pred_paths = np.vstack(y_pred_paths)
        y_true_dir_all = np.vstack(y_true_dir_all)
        y_pred_dir_logits_all = np.vstack(y_pred_dir_logits_all)

        # Overall raw MAE across all horizons
        mae_raw_overall = float(round(mean_absolute_error(y_true_paths.flatten(), y_pred_paths.flatten()), 4))

        # MAE per horizon
        mae_by_step = {f"t+{i+1}": float(round(mean_absolute_error(y_true_paths[:, i], y_pred_paths[:, i]), 4))
                       for i in range(H)}

        # Direction accuracy
        y_pred_dir_bin = (y_pred_dir_logits_all >= 0.5).astype(int)
        acc_overall = float(round(accuracy_score(y_true_dir_all.flatten(), y_pred_dir_bin.flatten()), 4))
        acc_by_step = {f"t+{i+1}": float(round(accuracy_score(y_true_dir_all[:, i], y_pred_dir_bin[:, i]), 4))
                       for i in range(H)}
    else:
        mae_raw_overall = None
        mae_by_step = {}
        acc_overall = None
        acc_by_step = {}

    # Save eval artifacts
    with open(os.path.join(MODEL_OUTPUT_DIR, "forecast_mae.json"), "w") as f:
        json.dump({"forecast_close_mae": mae_raw_overall}, f, indent=2)

    with open(os.path.join(MODEL_OUTPUT_DIR, "forecast_mae_by_step.json"), "w") as f:
        json.dump(mae_by_step, f, indent=2)

    with open(os.path.join(MODEL_OUTPUT_DIR, "direction_accuracy.json"), "w") as f:
        json.dump({"overall_accuracy": acc_overall}, f, indent=2)

    with open(os.path.join(MODEL_OUTPUT_DIR, "direction_accuracy_by_step.json"), "w") as f:
        json.dump(acc_by_step, f, indent=2)

    # Save model & metadata
    model.save(os.path.join(MODEL_OUTPUT_DIR, "cnn_lstm_forecast_model.keras"))
    with open(os.path.join(MODEL_OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "input_features": input_features,
            "target_kind": "log_returns",
            "ret_cols": ret_cols,
            "dir_cols": dir_cols,
            "input_seq_len": INPUT_SEQ_LEN,
            "forecast_horizon": FORECAST_HORIZON,
            "version": VERSION_TAG,
            "scalers_fit_scope": "train_only",
            "scalers": {"X": "MinMaxScaler", "returns": "StandardScaler"}
        }, f, indent=2)

    # Quick diagnostic plots in raw space (t+1,t+7,t+14)
    if len(y_true_paths):
        for i in [1, 7, 14]:
            if i <= y_true_paths.shape[1]:
                plt.figure(figsize=(10, 4))
                plt.plot(y_true_paths[:, i - 1], label=f"True t+{i}")
                plt.plot(y_pred_paths[:, i - 1], label=f"Pred t+{i}")
                plt.title(f"Close Forecast (t+{i}) – RAW")
                plt.legend(); plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f"forecast_t{i}_raw.png"))
                plt.close()

if __name__ == "__main__":
    main()
