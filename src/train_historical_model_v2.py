# Transformer-based Quantile Regression + Direction Classifier (v2)
# - 14-day horizon (t+1..t+14)
# - Transformer encoder with sinusoidal positional encoding
# - Quantile regression heads (p10, p50, p90) using pinball loss
# - Direction classifier with focal loss
# - Purged chronological split, early stopping, LR scheduling
# - Saves artifacts compatible with v1 (forecast_mae.json, direction_accuracy*.json, history, plots, metadata)

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------------------------------
# Project-specific constants (expecting src/constants.py as in v1)
# ------------------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *  # expects COL_CLOSE, etc., like in v1

# ------------------------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------------------------
INPUT_SEQ_LEN = 180
FORECAST_HORIZON = 14
EPOCHS = 80
BATCH_SIZE = 32
D_MODEL = 128
NUM_HEADS = 4
FF_DIM = 256
DROPOUT = 0.15
LR = 1e-3
VERSION_TAG = "v2_transformer_close_quantile"
MODEL_OUTPUT_DIR = os.path.join("models", "transformer_close_regression", VERSION_TAG)

for device in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass

# ------------------------------------------------------------------------------------
# Data Sequence (same contract as v1, but returns targets for 3 quantile heads + dir)
# ------------------------------------------------------------------------------------
class SequenceGenerator(Sequence):
    def __init__(self, df, input_seq_len, feature_cols, target_col, forecast_horizon, batch_size=32):
        self.df = df
        self.input_seq_len = input_seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        dir_cols = [f"direction_t{i}" for i in range(1, forecast_horizon + 1)]
        self.data = df[feature_cols + [target_col] + dir_cols].values.astype(np.float32)
        self.n_features = len(feature_cols)
        self.indices = np.arange(len(self.data) - input_seq_len - forecast_horizon + 1)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(batch_indices), self.input_seq_len, self.n_features), dtype=np.float32)
        y_close = np.zeros((len(batch_indices), self.forecast_horizon), dtype=np.float32)
        y_dir = np.zeros((len(batch_indices), self.forecast_horizon), dtype=np.float32)

        for i, start_idx in enumerate(batch_indices):
            X[i] = self.data[start_idx:start_idx + self.input_seq_len, :self.n_features]
            y_close[i] = self.data[start_idx + self.input_seq_len:start_idx + self.input_seq_len + self.forecast_horizon, self.n_features]
            y_dir[i] = self.data[start_idx + self.input_seq_len:start_idx + self.input_seq_len + self.forecast_horizon, self.n_features + 1]

        # same target for all quantile heads; p50 is the central forecast used for MAE
        return X, {
            "forecast_p10": y_close,
            "forecast_p50": y_close,
            "forecast_p90": y_close,
            "direction_classifier": y_dir
        }

# ------------------------------------------------------------------------------------
# Losses: Pinball (quantile) and Focal for classification
# ------------------------------------------------------------------------------------
def pinball_loss(q):
    q = tf.constant(q, dtype=tf.float32)
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss

# Standard sigmoid focal loss for binary classification
# y_true, y_pred shapes: (batch, horizon)
def focal_loss(gamma=2.0, alpha=0.25):
    gamma = tf.constant(gamma, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    def loss(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        fl = alpha * tf.pow(1 - pt, gamma) * ce
        return tf.reduce_mean(fl)
    return loss

# ------------------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ------------------------------------------------------------------------------------
def sinusoidal_position_encoding(seq_len, d_model):
    def get_angles(pos, i, d_model):
        return pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = get_angles(
        np.arange(seq_len)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    # apply sin to even indices; cos to odd indices
    sines = np.sin(angle_rads[:, 0::2])
    coses = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = coses
    return tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)  # (1, seq_len, d_model)

# ------------------------------------------------------------------------------------
# Transformer Encoder Block
# ------------------------------------------------------------------------------------
def transformer_encoder(x, num_heads=NUM_HEADS, key_dim=32, ff_dim=FF_DIM, dropout=DROPOUT, name_prefix="enc"):
    # Multi-Head Self-Attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name=f"{name_prefix}_mha")(x, x)
    attn_output = Dropout(dropout, name=f"{name_prefix}_attn_drop")(attn_output)
    x = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(x + attn_output)

    # Feed-Forward
    ffn = Dense(ff_dim, activation='relu', name=f"{name_prefix}_ffn1")(x)
    ffn = Dropout(dropout, name=f"{name_prefix}_ffn_drop")(ffn)
    ffn = Dense(x.shape[-1], name=f"{name_prefix}_ffn2")(ffn)
    x = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x + ffn)
    return x

# ------------------------------------------------------------------------------------
# Model Builder
# ------------------------------------------------------------------------------------
def build_model(input_shape, forecast_horizon):
    inputs = Input(shape=input_shape)

    # Project inputs to d_model
    x = Dense(D_MODEL, name="proj_in")(inputs)

    # Add sinusoidal positional encoding
    pe = sinusoidal_position_encoding(INPUT_SEQ_LEN, D_MODEL)
    x = x + pe  # broadcasting to (batch, seq_len, d_model)

    # Stacked transformer encoders
    x = transformer_encoder(x, key_dim=D_MODEL // NUM_HEADS, name_prefix="enc1")
    x = transformer_encoder(x, key_dim=D_MODEL // NUM_HEADS, name_prefix="enc2")
    x = transformer_encoder(x, key_dim=D_MODEL // NUM_HEADS, name_prefix="enc3")

    # Attention-style pooling over time
    # Learn attention weights over the sequence and get context vector
    attn_scores = Dense(1, activation=None, name="pool_attn_scores")(x)  # (batch, seq_len, 1)
    attn_weights = tf.nn.softmax(attn_scores, axis=1)
    context = tf.reduce_sum(attn_weights * x, axis=1)  # (batch, d_model)

    # Regularize & normalize
    context = Dropout(DROPOUT, name="head_dropout")(context)
    context = LayerNormalization(epsilon=1e-6, name="head_ln")(context)

    # Regression heads (quantiles)
    p10 = Dense(64, activation='relu', name="p10_fc1")(context)
    p10 = Dropout(DROPOUT, name="p10_drop")(p10)
    out_p10 = Dense(forecast_horizon, name="forecast_p10")(p10)

    p50 = Dense(64, activation='relu', name="p50_fc1")(context)
    p50 = Dropout(DROPOUT, name="p50_drop")(p50)
    out_p50 = Dense(forecast_horizon, name="forecast_p50")(p50)

    p90 = Dense(64, activation='relu', name="p90_fc1")(context)
    p90 = Dropout(DROPOUT, name="p90_drop")(p90)
    out_p90 = Dense(forecast_horizon, name="forecast_p90")(p90)

    # Direction classifier head
    dhead = Dense(64, activation='relu', name="dir_fc1")(context)
    dhead = Dropout(DROPOUT, name="dir_drop")(dhead)
    out_dir = Dense(forecast_horizon, activation="sigmoid", name="direction_classifier")(dhead)

    model = Model(inputs, [out_p10, out_p50, out_p90, out_dir])

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss={
            "forecast_p10": pinball_loss(0.1),
            "forecast_p50": pinball_loss(0.5),
            "forecast_p90": pinball_loss(0.9),
            "direction_classifier": focal_loss(gamma=2.0, alpha=0.25),
        },
        loss_weights={
            "forecast_p10": 0.5,
            "forecast_p50": 1.0,  # central forecast emphasized
            "forecast_p90": 0.5,
            "direction_classifier": 1.0,  # stronger weight vs v1
        },
        metrics={
            "forecast_p50": "mae",
            "direction_classifier": "accuracy",
        },
    )
    return model

# ------------------------------------------------------------------------------------
# Main training routine
# ------------------------------------------------------------------------------------

def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # ---------------------------- Load & prepare data ----------------------------
    df = pd.read_csv('data/historical_data_final.csv')

    # Future direction labels (1 if future close > current close)
    for i in range(1, FORECAST_HORIZON + 1):
        df[f"direction_t{i}"] = (df[COL_CLOSE].shift(-i) > df[COL_CLOSE]).astype(int)

    # Example additional basic features used in v1
    df["delta_close"] = df[COL_CLOSE].diff()
    for lag in range(1, 31):
        df[f"lag_close_{lag}"] = df[COL_CLOSE].shift(lag)

    df.dropna(inplace=True)

    target_col = COL_CLOSE
    dir_cols = [f"direction_t{i}" for i in range(1, FORECAST_HORIZON + 1)]
    # Feature set: everything except target + dir labels
    input_features = [c for c in df.columns if c not in [target_col] + dir_cols]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    df[input_features] = scaler_X.fit_transform(df[input_features])
    df[target_col] = np.log1p(df[target_col])  # stabilize heavy tails
    df[target_col] = scaler_y.fit_transform(df[[target_col]])

    # Chronological split with a small purge gap to avoid leakage around the split
    purge = max(FORECAST_HORIZON, 7)
    split_idx = int(len(df) * 0.8)
    train_end = split_idx - purge
    df_train = df.iloc[:train_end]
    df_val = df.iloc[split_idx:]

    train_gen = SequenceGenerator(df_train, INPUT_SEQ_LEN, input_features, target_col, FORECAST_HORIZON, batch_size=BATCH_SIZE)
    val_gen = SequenceGenerator(df_val, INPUT_SEQ_LEN, input_features, target_col, FORECAST_HORIZON, batch_size=BATCH_SIZE)

    # ---------------------------- Build & train model ----------------------------
    model = build_model((INPUT_SEQ_LEN, len(input_features)), FORECAST_HORIZON)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            EarlyStopping(patience=16, restore_best_weights=True, verbose=1),
        ],
        verbose=1,
    )

    # Save training history
    with open(os.path.join(MODEL_OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump({k: [float(vv) for vv in v] for k, v in history.history.items()}, f, indent=4)

    # Plot training curves (central quantile + direction)
    plt.figure(figsize=(10,4))
    plt.plot(history.history.get('forecast_p50_mae', []), label='Train p50 MAE')
    plt.plot(history.history.get('val_forecast_p50_mae', []), label='Val p50 MAE')
    plt.title('Central Forecast (p50) MAE')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'training_mae_p50.png'))
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(history.history.get('direction_classifier_accuracy', []), label='Train Dir Acc')
    plt.plot(history.history.get('val_direction_classifier_accuracy', []), label='Val Dir Acc')
    plt.title('Direction Classifier Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'training_dir_acc.png'))
    plt.close()

    # ---------------------------- Evaluate on validation ----------------------------
    y_true_close, y_pred_p50 = [], []
    y_true_dir, y_pred_dir = [], []

    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        preds = model.predict(X_batch, verbose=0)
        # preds: [p10, p50, p90, dir]
        y_true_close.append(y_batch["forecast_p50"])  # same as true close
        y_pred_p50.append(preds[1])
        y_true_dir.append(y_batch["direction_classifier"])
        y_pred_dir.append(preds[3])

    y_true_close = np.concatenate(y_true_close, axis=0)
    y_pred_p50 = np.concatenate(y_pred_p50, axis=0)
    y_true_dir = np.concatenate(y_true_dir, axis=0)
    y_pred_dir = np.concatenate(y_pred_dir, axis=0)

    # Denormalize price back to original scale
    y_true_close_denorm = np.expm1(scaler_y.inverse_transform(y_true_close))
    y_pred_p50_denorm = np.expm1(scaler_y.inverse_transform(y_pred_p50))

    mae = float(round(mean_absolute_error(y_true_close_denorm.flatten(), y_pred_p50_denorm.flatten()), 4))
    y_pred_dir_binary = (y_pred_dir >= 0.5).astype(int)
    acc_overall = float(round(accuracy_score(y_true_dir.flatten(), y_pred_dir_binary.flatten()), 4))

    with open(os.path.join(MODEL_OUTPUT_DIR, "forecast_mae.json"), "w") as f:
        json.dump({"forecast_close_mae": mae}, f, indent=4)

    with open(os.path.join(MODEL_OUTPUT_DIR, "direction_accuracy.json"), "w") as f:
        json.dump({"overall_accuracy": acc_overall}, f, indent=4)

    # Step-wise direction accuracy
    acc_by_step = {}
    for i in range(FORECAST_HORIZON):
        step_acc = float(round(accuracy_score(y_true_dir[:, i], y_pred_dir_binary[:, i]), 4))
        acc_by_step[f"t+{i+1}"] = step_acc
    with open(os.path.join(MODEL_OUTPUT_DIR, "direction_accuracy_by_step.json"), "w") as f:
        json.dump(acc_by_step, f, indent=4)

    # Training summary (mirrors v1 fields but with p50 MAE)
    best_train_mae = float(np.min(history.history.get('forecast_p50_mae', [np.nan])))
    best_val_mae = float(np.min(history.history.get('val_forecast_p50_mae', [np.nan])))
    best_train_dir = float(np.max(history.history.get('direction_classifier_accuracy', [np.nan])))
    best_val_dir = float(np.max(history.history.get('val_direction_classifier_accuracy', [np.nan])))

    with open(os.path.join(MODEL_OUTPUT_DIR, "training_summary.txt"), "w") as f:
        f.write(f"Best Train Forecast MAE (p50): {best_train_mae:.4f}\n")
        f.write(f"Best Val Forecast MAE (p50): {best_val_mae:.4f}\n")
        f.write(f"Best Train Dir Acc: {best_train_dir:.4f}\n")
        f.write(f"Best Val Dir Acc: {best_val_dir:.4f}\n")

    # Save model & scalers
    model.save(os.path.join(MODEL_OUTPUT_DIR, "transformer_quantile_dir_model.keras"))
    joblib.dump(scaler_X, os.path.join(MODEL_OUTPUT_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_OUTPUT_DIR, "scaler_y.pkl"))

    # Save metadata
    with open(os.path.join(MODEL_OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "input_features": input_features,
            "target_col": target_col,
            "input_seq_len": INPUT_SEQ_LEN,
            "forecast_horizon": FORECAST_HORIZON,
            "version": VERSION_TAG,
            "model": "Transformer + Quantiles (p10/p50/p90) + Direction (focal loss)",
        }, f, indent=4)

    # Diagnostic plots for a few horizons
    for i in [1, 7, 14]:
        if i <= y_true_close_denorm.shape[1]:
            plt.figure(figsize=(10, 4))
            plt.plot(y_true_close_denorm[:, i - 1], label=f'True t+{i}')
            plt.plot(y_pred_p50_denorm[:, i - 1], label=f'Pred p50 t+{i}')
            plt.title(f"Close Price Forecast (t+{i})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f"forecast_t{i}.png"))
            plt.close()

    print("\n=== v2 training complete ===")
    print(f"Validation p50 MAE (denorm): {mae}")
    print(f"Validation Direction Acc (overall): {acc_overall}")
    print(f"Per-step direction acc: {acc_by_step}")


if __name__ == "__main__":
    main()
