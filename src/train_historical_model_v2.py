# 14-day CLOSE forecasting via multi-step *vol-normalized 1-day log-returns* (per-symbol safe)
# Implements:
# - Per-horizon 1-day returns targets (not cumulative)
# - Vol normalization (per symbol) with 20d sigma, no sqrt(h)
# - Horizon-weighted Huber for magnitude + sign-loss on returns for direction
# - Class balance report per horizon (class_balance.json)
# - Per-horizon decision thresholds tau_h learned on validation (dir_thresholds.json)
# - Per-horizon affine calibration in raw price space (calibration.json)

import os, sys, json, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, LayerNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam

# ----- Project imports -----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *  # expects COL_CLOSE="close", COL_TIME_ORDINAL="time_ordinal", etc.

# GPU mem growth
for d in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(d, True)
    except Exception: pass

# ===== Config =====
INPUT_SEQ_LEN = 180
FORECAST_HORIZON = 14
EPOCHS = 50
BATCH_SIZE = 32
VERSION_TAG = "v2_volnorm1d_signloss_hw_h14"
MODEL_OUTPUT_DIR = os.path.join("models", "cnn_lstm_close_regression", VERSION_TAG)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ===== Windowing =====
def build_valid_starts(df: pd.DataFrame, seq_len: int, horizon: int) -> np.ndarray:
    starts = []
    for _, g in df.groupby("symbol", sort=False):
        s0, s1 = g.index.min(), g.index.max()
        max_start = s1 - (seq_len + horizon) + 1
        for s in range(s0, max(s0-1, max_start) + 1):
            starts.append(s)
    return np.array(starts, dtype=np.int64)

class SequenceGenerator(Sequence):
    """
    Y target is a vector of length H with *normalized 1-day* returns:
    nret1d_t{h} = (logP_{t+h} - logP_{t+h-1}) / vol20_{anchor}
    """
    def __init__(self, df_scaled: pd.DataFrame, L: int, feature_cols: list,
                 tgt_cols: list, valid_starts: np.ndarray, batch_size: int = 32):
        self.df = df_scaled; self.L = L
        self.feature_cols = feature_cols; self.tgt_cols = tgt_cols
        self.valid_starts = np.asarray(valid_starts, dtype=np.int64); self.batch = batch_size
        sel = self.feature_cols + self.tgt_cols
        self.data = self.df[sel].values.astype(np.float32)
        self.n_feat = len(self.feature_cols); self.H = len(self.tgt_cols)
    def __len__(self): return int(np.ceil(len(self.valid_starts) / self.batch))
    def __getitem__(self, idx):
        idxs = self.valid_starts[idx*self.batch:(idx+1)*self.batch]
        B = len(idxs)
        X     = np.zeros((B, self.L, self.n_feat), np.float32)
        y_ret = np.zeros((B, self.H),               np.float32)  # normalized 1-day returns
        for i, s in enumerate(idxs):
            X[i] = self.data[s:s+self.L, :self.n_feat]
            anchor = s + self.L - 1
            y_ret[i] = self.data[anchor, self.n_feat:self.n_feat+self.H]
        return X, y_ret

# ===== Losses =====
def horizon_weighted_huber(weights, delta=1.0):
    w = tf.constant(weights, dtype=tf.float32)  # shape (H,)
    def _loss(y_true, y_pred):
        err = tf.abs(y_true - y_pred)
        hub = tf.where(err < delta, 0.5*tf.square(err), delta*(err - 0.5*delta))
        hub = tf.reduce_mean(hub * w, axis=1)  # weighted mean over horizon
        return tf.reduce_mean(hub)             # mean over batch
    return _loss

def sign_loss(lambda_margin=0.0):
    """
    sign_loss = mean(max(0, - sign(y_true) * y_pred - m))
    If m>0, requires positive margin; here default 0.
    Works directly on normalized returns (sign invariant to scaling).
    """
    m = tf.constant(lambda_margin, dtype=tf.float32)
    def _loss(y_true, y_pred):
        s = tf.sign(y_true)            # {-1, 0, +1}; zeros contribute zero loss if pred>=0
        # max(0, -s*y_pred - m)
        t = -(s * y_pred) - m
        return tf.reduce_mean(tf.nn.relu(t))
    return _loss

def combined_loss(weights_huber, alpha_sign=0.5, delta=1.0):
    hw = horizon_weighted_huber(weights_huber, delta=delta)
    sl = sign_loss(0.0)
    def _loss(y_true, y_pred):
        return hw(y_true, y_pred) + alpha_sign * sl(y_true, y_pred)
    return _loss

# ===== Model =====
def build_model(input_shape, H):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.35)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.35)(x)

    # attention pooling
    att = Dense(1, activation='tanh')(x)
    att_w = tf.nn.softmax(att, axis=1)
    context = tf.reduce_sum(x * att_w, axis=1)

    context = Dense(96, activation='relu')(context)
    context = Dropout(0.25)(context)
    context = LayerNormalization()(context)

    # single head emits H normalized 1-day returns
    out_nret1d = Dense(H, name="forecast_nret1d")(context)

    model = Model(inputs, out_nret1d)
    return model

# ===== Training =====
def main():
    df = pd.read_csv('data/historical_data_final.csv')

    H = FORECAST_HORIZON
    df["close_raw"] = pd.to_numeric(df[COL_CLOSE], errors="coerce").astype(float)

    # base log price and 1d log return
    df["logp"] = df.groupby("symbol")["close_raw"].transform(np.log)
    df["r1"]   = df.groupby("symbol")["logp"].diff()
    # 20d vol of 1d returns (per symbol)
    df["vol20"] = (
        df.groupby("symbol")["r1"]
          .transform(lambda s: s.rolling(20, min_periods=10).std())
          .clip(lower=1e-6)
    )

    # --- Build per-horizon 1-day returns and normalized targets ---
    # raw_1d_ret_t{h} = logP_{t+h} - logP_{t+h-1}
    # nret1d_t{h}     = raw_1d_ret_t{h} / vol20_{anchor}
    nret1d_cols, raw1d_cols, dir_cols = [], [], []
    for h in range(1, H+1):
        rcol = f"raw1d_ret_t{h}"
        # shift(-h) - shift(-(h-1))
        df[rcol] = df.groupby("symbol")["logp"].shift(-h) - df.groupby("symbol")["logp"].shift(-(h-1))
        raw1d_cols.append(rcol)
        ncol = f"nret1d_t{h}"
        df[ncol] = df[rcol] / df["vol20"]
        nret1d_cols.append(ncol)
        dcol = f"direction_t{h}"
        df[dcol] = (df[rcol] > 0).astype("int8")
        dir_cols.append(dcol)

    # --- Basic supervised features: delta + lags (raw space), plus your existing engineered features preserved by post_process_data upstream ---
    df["delta_close"] = df.groupby("symbol")["close_raw"].diff()
    for lag in range(1, 31):
        df[f"lag_close_{lag}"] = df.groupby("symbol")["close_raw"].shift(lag)

    # drop NAs
    df = df.dropna().reset_index(drop=True)

    # === Class balance per horizon (save for inspection) ===
    class_balance = {}
    for h in range(1, H+1):
        p = float(df[f"direction_t{h}"].mean())
        class_balance[f"t+{h}"] = {"pos_frac": p, "neg_frac": 1.0 - p}
    with open(os.path.join(MODEL_OUTPUT_DIR, "class_balance.json"), "w") as f:
        json.dump(class_balance, f, indent=2)

    # inputs = everything except helpers and targets
    exclude = {"close_raw", "logp", "r1", "vol20"} | set(raw1d_cols) | set(nret1d_cols) | set(dir_cols)
    input_features = [c for c in df.columns if c not in exclude]

    # per-symbol time split
    parts_tr, parts_va = [], []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values(COL_TIME_ORDINAL)
        cut = int(len(g) * 0.8)
        parts_tr.append(g.iloc[:cut]); parts_va.append(g.iloc[cut:])
    df_tr = pd.concat(parts_tr, ignore_index=True)
    df_va = pd.concat(parts_va, ignore_index=True)

    # scalers
    scaler_X   = MinMaxScaler()
    scaler_nrt = StandardScaler()
    df_tr[input_features] = scaler_X.fit_transform(df_tr[input_features])
    df_va[input_features] = scaler_X.transform(df_va[input_features])
    df_tr[nret1d_cols] = scaler_nrt.fit_transform(df_tr[nret1d_cols])
    df_va[nret1d_cols] = scaler_nrt.transform(df_va[nret1d_cols])
    joblib.dump(scaler_X,  os.path.join(MODEL_OUTPUT_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_nrt, os.path.join(MODEL_OUTPUT_DIR, "scaler_nret1d.pkl"))

    # valid starts
    train_starts = build_valid_starts(df_tr, INPUT_SEQ_LEN, H)
    val_starts   = build_valid_starts(df_va, INPUT_SEQ_LEN, H)

    # generators
    train_gen = SequenceGenerator(df_tr, INPUT_SEQ_LEN, input_features, nret1d_cols, train_starts, BATCH_SIZE)
    val_gen   = SequenceGenerator(df_va, INPUT_SEQ_LEN, input_features, nret1d_cols, val_starts,   BATCH_SIZE)

    # model + losses
    model = build_model((INPUT_SEQ_LEN, len(input_features)), H)
    # horizon weights (emphasize early steps much more)
    w = np.exp(- (np.arange(1, H+1) - 1) / 3.0).astype(np.float32)  # fast decay
    model.compile(
        optimizer=Adam(1e-3),
        loss=combined_loss(w, alpha_sign=0.6, delta=1.0),  # 0.6 weight for sign-loss
        metrics=["mae"],
    )

    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[ReduceLROnPlateau(factor=0.5, patience=8),
                   EarlyStopping(patience=16, restore_best_weights=True)],
        verbose=1
    )
    with open(os.path.join(MODEL_OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()}, f, indent=2)

    # ===== Validation eval in RAW PRICE + calibration fit + per-h threshold fit =====
    idxs = val_starts; n_batches = int(np.ceil(len(idxs)/BATCH_SIZE))
    Xcols = input_features; Hh = H
    y_true_paths, y_pred_paths = [], []
    # per-sample anchors for de-normalization
    last_close_all, last_vol20_all = [], []
    # also store normalized 1d return predictions/labels for threshold search
    nret1d_true_all, nret1d_pred_all = [], []

    for b in range(n_batches):
        bi = idxs[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        if len(bi) == 0: continue
        B = len(bi)
        X = np.zeros((B, INPUT_SEQ_LEN, len(Xcols)), np.float32)
        nret_true_sc = np.zeros((B, Hh), np.float32)
        last_close = np.zeros((B,), np.float32)
        last_vol20 = np.zeros((B,), np.float32)
        for i, s in enumerate(bi):
            X[i] = df_va.iloc[s:s+INPUT_SEQ_LEN][Xcols].values.astype(np.float32)
            anchor = s + INPUT_SEQ_LEN - 1
            nret_true_sc[i] = df_va.iloc[anchor][nret1d_cols].values.astype(np.float32)
            last_close[i]   = float(df_va.iloc[anchor]["close_raw"])
            last_vol20[i]   = float(df_va.iloc[anchor]["vol20"])

        nret_pred_sc = model.predict(X, verbose=0)      # (B,H)
        # inverse normalized 1d returns (back to normalized space first -> StandardScaler inverse)
        nret_true = scaler_nrt.inverse_transform(nret_true_sc)
        nret_pred = scaler_nrt.inverse_transform(nret_pred_sc)

        # store for threshold search
        nret1d_true_all.append(nret_true)
        nret1d_pred_all.append(nret_pred)

        # convert to raw 1-day returns using last vol20 (per sample, shared across H)
        scale = (last_vol20[:, None])  # 1-day, no sqrt(h)
        raw1d_true = nret_true * scale
        raw1d_pred = nret_pred * scale

        # integrate raw 1-day returns to log-path & price path
        last_logp = np.log(last_close)
        true_log_paths = last_logp[:, None] + np.cumsum(raw1d_true, axis=1)
        pred_log_paths = last_logp[:, None] + np.cumsum(raw1d_pred, axis=1)

        y_true_paths.append(np.exp(true_log_paths))
        y_pred_paths.append(np.exp(pred_log_paths))
        last_close_all.append(last_close)
        last_vol20_all.append(last_vol20)

    if y_true_paths:
        y_true_paths = np.vstack(y_true_paths)  # (N,H)
        y_pred_paths = np.vstack(y_pred_paths)

        # raw price MAE (before calibration)
        mae_by_h = {f"t+{i+1}": float(mean_absolute_error(y_true_paths[:, i], y_pred_paths[:, i])) for i in range(H)}
        with open(os.path.join(MODEL_OUTPUT_DIR, "forecast_mae_by_step_precal.json"), "w") as f:
            json.dump(mae_by_h, f, indent=2)

        # ===== Per-horizon affine calibration: y ≈ a_h * y_hat + b_h =====
        a, b = [], []
        for i in range(H):
            x = y_pred_paths[:, i]; y = y_true_paths[:, i]
            if np.allclose(x.std(), 0):
                a_i, b_i = 1.0, 0.0
            else:
                a_i, b_i = np.polyfit(x, y, 1)
            a.append(float(a_i)); b.append(float(b_i))
        with open(os.path.join(MODEL_OUTPUT_DIR, "calibration.json"), "w") as f:
            json.dump({"a": a, "b": b}, f, indent=2)

        # post-calibration MAE
        y_pred_cal = (y_pred_paths * np.array(a)[None, :]) + np.array(b)[None, :]
        mae_by_h_cal = {f"t+{i+1}": float(mean_absolute_error(y_true_paths[:, i], y_pred_cal[:, i])) for i in range(H)}
        with open(os.path.join(MODEL_OUTPUT_DIR, "forecast_mae_by_step.json"), "w") as f:
            json.dump(mae_by_h_cal, f, indent=2)
        mae_overall = float(mean_absolute_error(y_true_paths.flatten(), y_pred_cal.flatten()))
    else:
        mae_overall = None

    with open(os.path.join(MODEL_OUTPUT_DIR, "forecast_mae.json"), "w") as f:
        json.dump({"forecast_close_mae": mae_overall}, f, indent=2)

    # ===== Direction thresholds per horizon (validation) =====
    nret1d_true_all = np.vstack(nret1d_true_all) if nret1d_true_all else np.zeros((0, H), np.float32)
    nret1d_pred_all = np.vstack(nret1d_pred_all) if nret1d_pred_all else np.zeros((0, H), np.float32)
    dir_thresholds = []
    if len(nret1d_true_all) > 0:
        for i in range(H):
            y = nret1d_true_all[:, i]
            p = nret1d_pred_all[:, i]
            # search candidate thresholds around 0 within small range to correct bias
            # candidates = percentiles of p and 0 for robustness
            cands = np.unique(np.percentile(p, [0,5,10,20,30,40,50,60,70,80,90,95,100]).tolist() + [0.0])
            best_tau, best_acc = 0.0, -1.0
            y_sign = (y > 0).astype(np.int8)
            for tau in cands:
                pred_sign = (p - tau > 0).astype(np.int8)
                acc = float((pred_sign == y_sign).mean())
                if acc > best_acc:
                    best_acc, best_tau = acc, float(tau)
            dir_thresholds.append(best_tau)
    with open(os.path.join(MODEL_OUTPUT_DIR, "dir_thresholds.json"), "w") as f:
        json.dump({"tau": dir_thresholds}, f, indent=2)

    # Save model & metadata
    model.save(os.path.join(MODEL_OUTPUT_DIR, "cnn_lstm_forecast_model.keras"))
    with open(os.path.join(MODEL_OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "input_features": input_features,
            "target_kind": "vol_normalized_1d_returns",
            "nret1d_cols": nret1d_cols,
            "raw1d_cols": raw1d_cols,
            "dir_cols": dir_cols,  # kept for analysis only
            "input_seq_len": INPUT_SEQ_LEN,
            "forecast_horizon": FORECAST_HORIZON,
            "version": VERSION_TAG,
            "scalers_fit_scope": "train_only",
            "scalers": {"X": "MinMaxScaler", "normalized_1d_returns": "StandardScaler"}
        }, f, indent=2)

if __name__ == "__main__":
    main()
