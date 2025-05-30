# Optimized CNN-LSTM Regression Model for Multi-Step Close Price Forecasting
import numpy as np
import pandas as pd
import joblib
import sys
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
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
FORECAST_HORIZON = 5
EPOCHS = 50
BATCH_SIZE = 32
VERSION_TAG = "v1_close_regression"
MODEL_OUTPUT_DIR = os.path.join("models", "cnn_lstm_close_regression", VERSION_TAG)

class SequenceGenerator(Sequence):
    def __init__(self, df, input_seq_len, feature_cols, target_col, forecast_horizon, batch_size=32):
        self.df = df
        self.input_seq_len = input_seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.data = df[feature_cols + [target_col] + [f"direction_t{i}" for i in range(1, forecast_horizon + 1)]].values.astype(np.float32)
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

        return X, {"forecast_close": y_close, "direction_classifier": y_dir}

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

    forecast_output = Dense(forecast_horizon, name="forecast_close")(context)
    direction_output = Dense(forecast_horizon, name="direction_classifier", activation="sigmoid")(context)

    model = Model(inputs, [forecast_output, direction_output])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            "forecast_close": Huber(),
            "direction_classifier": "binary_crossentropy"
        },
        loss_weights={
            "forecast_close": 1.0,
            "direction_classifier": 0.5
        },
        metrics={
            "forecast_close": "mae",
            "direction_classifier": "accuracy"
        }
    )
    return model

def main():
    df = pd.read_csv('data/historical_data_final.csv')

    for i in range(1, FORECAST_HORIZON + 1):
        df[f"direction_t{i}"] = (df[COL_CLOSE].shift(-i) > df[COL_CLOSE]).astype(int)

    df["delta_close"] = df[COL_CLOSE].diff()

    for lag in range(1, 31):
        df[f"lag_close_{lag}"] = df[COL_CLOSE].shift(lag)

    df.dropna(inplace=True)

    target_col = COL_CLOSE
    input_features = [col for col in df.columns if col not in [target_col] + [f"direction_t{i}" for i in range(1, FORECAST_HORIZON + 1)]]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    df[input_features] = scaler_X.fit_transform(df[input_features])
    df[target_col] = np.log1p(df[target_col])
    df[target_col] = scaler_y.fit_transform(df[[target_col]])

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    train_size = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:train_size], df.iloc[train_size:]

    train_gen = SequenceGenerator(df_train, INPUT_SEQ_LEN, input_features, target_col, FORECAST_HORIZON)
    val_gen = SequenceGenerator(df_val, INPUT_SEQ_LEN, input_features, target_col, FORECAST_HORIZON)

    model = build_model((INPUT_SEQ_LEN, len(input_features)), FORECAST_HORIZON)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            ReduceLROnPlateau(factor=0.5, patience=10),
            EarlyStopping(patience=20, restore_best_weights=True)
        ],
        verbose=1
    )

    with open(os.path.join(MODEL_OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump({k: [float(vv) for vv in v] for k, v in history.history.items()}, f, indent=4)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(history.history['forecast_close_mae'], label='Train Forecast MAE')
    axs[0].plot(history.history['val_forecast_close_mae'], label='Val Forecast MAE')
    axs[0].set_title("Forecast Close MAE")
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(history.history['direction_classifier_accuracy'], label='Train Dir Acc')
    axs[1].plot(history.history['val_direction_classifier_accuracy'], label='Val Dir Acc')
    axs[1].set_title("Direction Classifier Accuracy")
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "training_history.png"))
    plt.close()

    y_true_close, y_pred_close = [], []
    y_true_dir, y_pred_dir = [], []
    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        preds = model.predict(X_batch)
        y_true_close.append(y_batch["forecast_close"])
        y_pred_close.append(preds[0])
        y_true_dir.append(y_batch["direction_classifier"])
        y_pred_dir.append(preds[1])

    y_true_close = np.expm1(scaler_y.inverse_transform(np.concatenate(y_true_close, axis=0)))
    y_pred_close = np.expm1(scaler_y.inverse_transform(np.concatenate(y_pred_close, axis=0)))
    y_true_dir = np.concatenate(y_true_dir, axis=0)
    y_pred_dir = np.concatenate(y_pred_dir, axis=0)
    y_pred_dir_binary = (y_pred_dir >= 0.5).astype(int)

    mae = float(round(mean_absolute_error(y_true_close.flatten(), y_pred_close.flatten()), 4))
    acc = float(round(accuracy_score(y_true_dir.flatten(), y_pred_dir_binary.flatten()), 4))

    with open(os.path.join(MODEL_OUTPUT_DIR, "forecast_mae.json"), "w") as f:
        json.dump({"forecast_close_mae": mae}, f, indent=4)

    with open(os.path.join(MODEL_OUTPUT_DIR, "direction_accuracy.json"), "w") as f:
        json.dump({"overall_accuracy": acc}, f, indent=4)

    acc_by_step = {f"t+{i+1}": float(round(accuracy_score(y_true_dir[:, i], y_pred_dir_binary[:, i]), 4)) for i in range(FORECAST_HORIZON)}
    with open(os.path.join(MODEL_OUTPUT_DIR, "direction_accuracy_by_step.json"), "w") as f:
        json.dump(acc_by_step, f, indent=4)

    with open(os.path.join(MODEL_OUTPUT_DIR, "training_summary.txt"), "w") as f:
        f.write(f"Best Train Forecast MAE: {min(history.history['forecast_close_mae']):.4f}\n")
        f.write(f"Best Val Forecast MAE: {min(history.history['val_forecast_close_mae']):.4f}\n")
        f.write(f"Best Train Dir Acc: {max(history.history['direction_classifier_accuracy']):.4f}\n")
        f.write(f"Best Val Dir Acc: {max(history.history['val_direction_classifier_accuracy']):.4f}\n")

    model.save(os.path.join(MODEL_OUTPUT_DIR, "cnn_lstm_forecast_model.keras"))
    joblib.dump(scaler_X, os.path.join(MODEL_OUTPUT_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_OUTPUT_DIR, "scaler_y.pkl"))

    with open(os.path.join(MODEL_OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "input_features": input_features,
            "target_col": target_col,
            "input_seq_len": INPUT_SEQ_LEN,
            "forecast_horizon": FORECAST_HORIZON,
            "version": VERSION_TAG
        }, f)

    for i in [1, 3, 5]:
        if i <= y_true_close.shape[1]:
            plt.figure(figsize=(10, 4))
            plt.plot(y_true_close[:, i - 1], label=f'True t+{i}')
            plt.plot(y_pred_close[:, i - 1], label=f'Predicted t+{i}')
            plt.title(f"Close Price Forecast (t+{i})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f"forecast_t{i}.png"))
            plt.close()

if __name__ == "__main__":
    main()
