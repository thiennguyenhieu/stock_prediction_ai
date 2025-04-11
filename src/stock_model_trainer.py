import numpy as np
import pandas as pd
import joblib
import sys
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, LayerNormalization, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import Huber
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import Adam

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *
from src.utility import save_dataframe_to_csv
from src.feature_engineering import add_technical_indicators

# --- Constants ---
INPUT_SEQ_LEN = 90
EPOCHS = 50
BATCH_SIZE = 32
VERSION_TAG = "v1_19Apr2025"
MODEL_OUTPUT_DIR = os.path.join("models", "cnn_lstm_stock_model", VERSION_TAG)

# --- Sequence Generator ---
class SequenceGenerator(Sequence):
    def __init__(self, df, input_seq_len, feature_cols, target_cols, batch_size=32):
        self.input_seq_len = input_seq_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.batch_size = batch_size
        self.all_cols = feature_cols + [col for col in self.target_cols if col not in feature_cols]
        self.data = df[self.all_cols].values.astype(np.float32)
        self.n_features = len(feature_cols)
        self.n_targets = len(self.target_cols)
        self.indices = np.arange(len(self.data) - input_seq_len)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((len(batch_indices), self.input_seq_len, self.n_features), dtype=np.float32)
        y = np.zeros((len(batch_indices), self.n_targets), dtype=np.float32)
        for i, start_idx in enumerate(batch_indices):
            end_idx = start_idx + self.input_seq_len
            X[i] = self.data[start_idx:end_idx, :self.n_features]
            y[i] = self.data[end_idx, [self.all_cols.index(col) for col in self.target_cols]]
        return X, {
            "out_close": y[:, 0],
            "out_eps": y[:, 1],
            "out_bvps": y[:, 2]
        }

# --- Model Architecture ---
def build_head(base, name):
    h = Dense(32, activation='relu')(base)
    h = Dropout(0.2)(h)
    h = LayerNormalization()(h)
    return Dense(1, name=name)(h)

def build_multitask_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)

    
    x = Dense(64, activation='relu')(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)

    shared = Dense(64, activation='relu')(x)

    out_close = build_head(shared, "out_close")
    lstm_eps = Dense(32, activation='relu')(x)
    lstm_eps = BatchNormalization()(lstm_eps)
    out_eps = build_head(lstm_eps, "out_eps")

    lstm_bvps = Dense(32, activation='relu')(x)
    lstm_bvps = BatchNormalization()(lstm_bvps)
    out_bvps = build_head(lstm_bvps, "out_bvps")   

    model = Model(inputs, [out_close, out_eps, out_bvps])
    huber = Huber()
    lr_schedule = CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=1000)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss={
            "out_close": huber,
            "out_eps": huber,
            "out_bvps": huber
        },
        loss_weights={
            "out_close": 1.0,
            "out_eps": 0.5,
            "out_bvps": 0.5
        },
        metrics={
            "out_close": "mae",
            "out_eps": "mae",
            "out_bvps": "mae"
        }
    )
    return model

# --- Other Utilities ---
def handle_log_transformation(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: np.nan if x <= 0 else np.log1p(x))
    return df

def smooth_predictions(pred, window=5):
    return pd.DataFrame(pred).rolling(window=window, min_periods=1).mean().values

def plot_predictions(y_true, y_pred, target_cols, tag):
    os.makedirs(f"{MODEL_OUTPUT_DIR}/predictions", exist_ok=True)
    for i, col in enumerate(target_cols):
        plt.figure(figsize=(10, 4))
        plt.plot(y_true[:, i], label='True')
        plt.plot(y_pred[:, i], label='Predicted')
        plt.title(f'{tag} – {col}')
        plt.xlabel('Samples')
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{MODEL_OUTPUT_DIR}/predictions/{tag}_{col}.png")
        plt.close()

# --- Main Execution ---
def main():
    final_output_path = 'data/stock_data_final.csv'
    if not os.path.exists(final_output_path):
        print("[ERROR] Dataset not found")
        return

    df = pd.read_csv(final_output_path)
    df = add_technical_indicators(df)
    df = handle_log_transformation(df, [COL_EPS, COL_BVPS])
    df = df.dropna(subset=TARGET_COLS)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    y_log = df[TARGET_COLS].copy()  # log+scale targets for regression
    scaler_X = MinMaxScaler()
    scaler_close = MinMaxScaler()
    scaler_eps = MinMaxScaler()
    scaler_bvps = MinMaxScaler()

    df[INPUT_FEATURES] = scaler_X.fit_transform(df[INPUT_FEATURES])
    df[COL_CLOSE] = scaler_close.fit_transform(np.log1p(y_log[[COL_CLOSE]]))
    df[COL_EPS] = scaler_eps.fit_transform(np.log1p(y_log[[COL_EPS]]))
    df[COL_BVPS] = scaler_bvps.fit_transform(np.log1p(y_log[[COL_BVPS]]))

    df_train, df_val = train_test_split(df, test_size=0.2, shuffle=False, random_state=42)
    train_gen = SequenceGenerator(df_train, INPUT_SEQ_LEN, INPUT_FEATURES, [COL_CLOSE, COL_EPS, COL_BVPS])
    val_gen = SequenceGenerator(df_val, INPUT_SEQ_LEN, INPUT_FEATURES, [COL_CLOSE, COL_EPS, COL_BVPS])

    model = build_multitask_model((INPUT_SEQ_LEN, len(INPUT_FEATURES)))
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[
        ReduceLROnPlateau(factor=0.5, patience=10),
        EarlyStopping(patience=20, restore_best_weights=True)
    ], verbose=1)

    y_close = []
    y_eps = []
    y_bvps = []
    for i in range(len(val_gen)):
        _, y_batch = val_gen[i]
        y_close.append(y_batch["out_close"])
        y_eps.append(y_batch["out_eps"])
        y_bvps.append(y_batch["out_bvps"])

    y_true = np.column_stack([
        np.concatenate(y_close),
        np.concatenate(y_eps),
        np.concatenate(y_bvps)
    ])
    y_pred_raw = np.concatenate(model.predict(val_gen), axis=1)
    y_pred_smooth = smooth_predictions(y_pred_raw, window=5)

    y_true_inv = np.column_stack([
        np.expm1(scaler_close.inverse_transform(y_true[:, [0]])),
        np.expm1(scaler_eps.inverse_transform(y_true[:, [1]])),
        np.expm1(scaler_bvps.inverse_transform(y_true[:, [2]]))
    ])

    y_pred_inv = np.column_stack([
        np.expm1(scaler_close.inverse_transform(y_pred_smooth[:, [0]])),
        np.expm1(scaler_eps.inverse_transform(y_pred_smooth[:, [1]])),
        np.expm1(scaler_bvps.inverse_transform(y_pred_smooth[:, [2]]))
    ])

    # --- Evaluate Regression ---
    mae_result = {
        COL_CLOSE: float(round(mean_absolute_error(y_true_inv[:, 0], y_pred_inv[:, 0]), 4)),
        COL_EPS: float(round(mean_absolute_error(y_true_inv[:, 1], y_pred_inv[:, 1]), 4)),
        COL_BVPS: float(round(mean_absolute_error(y_true_inv[:, 2], y_pred_inv[:, 2]), 4))
    }
    json.dump(mae_result, open(f"{MODEL_OUTPUT_DIR}/regression_mae.json", "w"), indent=4)

    # --- Save Model and Artifacts ---
    model.save(os.path.join(MODEL_OUTPUT_DIR, "cnn_lstm_multitask_model.keras"))

    joblib.dump(scaler_X, os.path.join(MODEL_OUTPUT_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_close, os.path.join(MODEL_OUTPUT_DIR, "scaler_close.pkl"))
    joblib.dump(scaler_eps, os.path.join(MODEL_OUTPUT_DIR, "scaler_eps.pkl"))
    joblib.dump(scaler_bvps, os.path.join(MODEL_OUTPUT_DIR, "scaler_bvps.pkl"))

    with open(os.path.join(MODEL_OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "input_features": INPUT_FEATURES,
            "target_cols": TARGET_COLS,
            "input_seq_len": INPUT_SEQ_LEN,
            "version": VERSION_TAG
        }, f)

    print("Model and preprocessing artifacts saved successfully.")

    plot_predictions(y_true_inv, y_pred_inv, TARGET_COLS, tag="final")

    # --- Plot Training Loss ---
    plt.figure(figsize=(10, 4))
    for i, metric in enumerate(history.history):
        plt.plot(history.history[metric], label=metric)
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{MODEL_OUTPUT_DIR}/training_history.png")
    plt.close()

if __name__ == "__main__":
    main()
