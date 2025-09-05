import os
import sys
import json
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay
from tensorflow.keras.models import load_model

# project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import COL_CLOSE, COL_TIME_ORDINAL
from src.fetch_historical_data import process_symbol, post_process_data

# =========================
# CONFIG: edit these
# =========================
SYMBOLS = ["DRI", "DCM", "SHB"]
FORECAST_HORIZON = 14
SEQ_LEN = 180
LAG_WARMUP = 30                     # because we add lag_close_1..30
REQUIRED_RAW_ROWS = SEQ_LEN + LAG_WARMUP
ROLL_STEP_DAYS = 5                  # ordinal-day spacing between anchors
YEARS_HISTORY = 3
OUT_ROOT = "backtests/by_model"

# List any model folders you want to backtest, e.g. your latest v2 variants
# Add as many as you like; each one is evaluated independently.
MODEL_SPECS = [
    {
        "name": "v2_volnorm_focal_hw_h14",
        "model_dir": "models/cnn_lstm_close_regression/v2_volnorm_focal_hw_h14"
    },
]

# =========================
# Feature builders (match training)
# =========================
def build_features_returns(df: pd.DataFrame) -> pd.DataFrame:
    """For plain returns-trained models: delta + lag_close_1..30; dropna."""
    out = df.copy()
    out["delta_close"] = out[COL_CLOSE].diff()
    for lag in range(1, 31):
        out[f"lag_close_{lag}"] = out[COL_CLOSE].shift(lag)
    out = out.dropna().reset_index(drop=True)
    return out

def build_features_volnorm(df: pd.DataFrame) -> pd.DataFrame:
    """For vol-normalized returns models: adds vol20 + delta + lags; dropna."""
    out = df.copy()
    out["logp"] = np.log(out[COL_CLOSE].astype(float))
    out["r1"]   = out["logp"].diff()
    out["vol20"]= out["r1"].rolling(20, min_periods=10).std().clip(lower=1e-6)
    out["delta_close"] = out[COL_CLOSE].diff()
    for lag in range(1, 31):
        out[f"lag_close_{lag}"] = out[COL_CLOSE].shift(lag)
    out = out.dropna().reset_index(drop=True)
    return out

# =========================
# Model loader & predictor factory (auto-detects model kind)
# =========================
def make_predictor(model_dir: str):
    """
    Returns a callable: predict_fn(df_processed: pd.DataFrame, forecast_steps: int) -> np.ndarray[H]
    Auto-detects model kind by presence of scaler_ret.pkl vs scaler_nret.pkl.
    """
    meta_path = os.path.join(model_dir, "metadata.json")
    model_path = os.path.join(model_dir, "cnn_lstm_forecast_model.keras")
    if not (os.path.exists(meta_path) and os.path.exists(model_path)):
        raise FileNotFoundError(f"Missing model or metadata in {model_dir}")

    meta = json.load(open(meta_path, "r"))
    input_features = meta["input_features"]
    H = int(meta["forecast_horizon"])
    L = int(meta["input_seq_len"])

    model = load_model(model_path, compile=False)
    scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))

    is_volnorm = os.path.exists(os.path.join(model_dir, "scaler_nret.pkl"))
    if is_volnorm:
        scaler_nrt = joblib.load(os.path.join(model_dir, "scaler_nret.pkl"))
        calib_path = os.path.join(model_dir, "calibration.json")
        calib = json.load(open(calib_path, "r")) if os.path.exists(calib_path) else {"a": None, "b": None}

        def predict_fn(df_processed: pd.DataFrame, forecast_steps: int = H) -> np.ndarray:
            if forecast_steps > H:
                raise ValueError(f"Requested {forecast_steps} > model horizon {H}")
            df = build_features_volnorm(df_processed)
            missing = [c for c in input_features if c not in df.columns]
            if missing:
                raise ValueError(f"Missing input features: {missing}")
            if len(df) < L:
                raise ValueError(f"Not enough rows for prediction sequence. Need {L}, have {len(df)}")

            scaled = df.copy()
            scaled[input_features] = scaler_X.transform(scaled[input_features])
            X_seq = scaled[input_features].iloc[-L:].values.astype(np.float32).reshape(1, L, len(input_features))

            y_nret_sc, _ = model.predict(X_seq, verbose=0)   # (1,H)
            y_nret = scaler_nrt.inverse_transform(y_nret_sc)[0]  # (H,)

            last_close = float(df[COL_CLOSE].iloc[-1])
            last_vol20 = float(df["vol20"].iloc[-1])
            scale = (last_vol20 * np.sqrt(np.arange(1, H+1, dtype=np.float32)))
            y_ret = y_nret[:forecast_steps] * scale[:forecast_steps]

            logp0 = np.log(last_close); y_close = np.exp(logp0 + np.cumsum(y_ret))
            # optional per-horizon calibration
            if calib.get("a") and calib.get("b"):
                a = np.asarray(calib["a"], dtype=np.float32)[:forecast_steps]
                b = np.asarray(calib["b"], dtype=np.float32)[:forecast_steps]
                y_close = (y_close * a) + b
                y_close = np.maximum(y_close, 0.0)
            return y_close

        return predict_fn, H, L, True

    else:
        # plain returns model
        scaler_ret = joblib.load(os.path.join(model_dir, "scaler_ret.pkl"))

        def predict_fn(df_processed: pd.DataFrame, forecast_steps: int = H) -> np.ndarray:
            if forecast_steps > H:
                raise ValueError(f"Requested {forecast_steps} > model horizon {H}")
            df = build_features_returns(df_processed)
            missing = [c for c in input_features if c not in df.columns]
            if missing:
                raise ValueError(f"Missing input features: {missing}")
            if len(df) < L:
                raise ValueError(f"Not enough rows for prediction sequence. Need {L}, have {len(df)}")

            scaled = df.copy()
            scaled[input_features] = scaler_X.transform(scaled[input_features])
            X_seq = scaled[input_features].iloc[-L:].values.astype(np.float32).reshape(1, L, len(input_features))

            y_rets_sc, _ = model.predict(X_seq, verbose=0)       # (1,H)
            y_ret = scaler_ret.inverse_transform(y_rets_sc)[0]    # (H,)

            last_close = float(df[COL_CLOSE].iloc[-1])
            logp0 = np.log(last_close); y_close = np.exp(logp0 + np.cumsum(y_ret[:forecast_steps]))
            return y_close

        return predict_fn, H, L, False

# =========================
# Backtest plumbing (ordinal only)
# =========================
def build_anchors_ord(df: pd.DataFrame, horizon: int, min_gap_days: int):
    anchors = []
    ords = df[COL_TIME_ORDINAL].to_numpy()
    start_i = max(REQUIRED_RAW_ROWS - 1, 0)
    end_i = len(df) - horizon
    for i in range(start_i, end_i):
        if not anchors or (ords[i] - ords[anchors[-1]] >= min_gap_days):
            anchors.append(i)
    return anchors

def next_future_window(df: pd.DataFrame, start_idx: int, horizon: int):
    fut = df.iloc[start_idx + 1 : start_idx + 1 + horizon][[COL_TIME_ORDINAL, COL_CLOSE]].copy()
    return fut if len(fut) == horizon else None

def backtest_one_model(model_spec: dict):
    name = model_spec["name"]
    model_dir = model_spec["model_dir"]
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    predict_fn, H_model, L_model, is_volnorm = make_predictor(model_dir)
    H = min(FORECAST_HORIZON, H_model)
    L = L_model

    all_rows = []

    for sym in SYMBOLS:
        print(f"[INFO] {name}: backtesting {sym}...")
        end_date = pd.Timestamp.today().normalize()
        start_date = (end_date - pd.DateOffset(years=YEARS_HISTORY)).strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        raw = process_symbol(sym, start_date, end_str)
        if raw is None or raw.empty:
            print(f"[WARN] Empty data for {sym}"); continue

        df = post_process_data(raw)
        df = df.sort_values(COL_TIME_ORDINAL).reset_index(drop=True)
        if len(df) < REQUIRED_RAW_ROWS:
            print(f"[WARN] Not enough rows for {sym}: {len(df)} < {REQUIRED_RAW_ROWS}")
            continue

        anchors = build_anchors_ord(df, horizon=H, min_gap_days=ROLL_STEP_DAYS)
        if not anchors:
            print(f"[WARN] No anchors for {sym}"); continue

        for a_idx in anchors:
            hist = df.iloc[:a_idx + 1].copy()
            # safety
            if len(hist) < REQUIRED_RAW_ROWS:
                continue

            anchor_ord = int(hist[COL_TIME_ORDINAL].iloc[-1])
            anchor_close = float(hist[COL_CLOSE].iloc[-1])

            fut = next_future_window(df, a_idx, H)
            if fut is None:
                continue

            truth = fut[COL_CLOSE].to_numpy(float)      # (H,)
            baseline = np.full(H, anchor_close, float)

            try:
                pred = predict_fn(hist, forecast_steps=H)  # (H,)
            except Exception as e:
                print(f"[{name} ERR] {sym} @{anchor_ord}: {e}")
                continue

            for h in range(1, H+1):
                t_ord = int(fut[COL_TIME_ORDINAL].iloc[h-1])
                y = float(truth[h-1]); p = float(pred[h-1]); b = float(baseline[h-1])
                rows = {
                    "model": name, "symbol": sym,
                    "anchor_ord": anchor_ord, "target_ord": t_ord, "h": h,
                    "truth": y, "pred": p, "baseline_pred": b,
                    "mae": abs(y - p), "baseline_mae": abs(y - b),
                    "dir": int(p - anchor_close > 0),
                    "truth_dir": int(y - anchor_close > 0),
                }
                all_rows.append(rows)

    if not all_rows:
        print(f"[WARN] No rows for model {name}")
        return

    per_h = pd.DataFrame(all_rows)
    # summaries
    def agg(df):
        return pd.Series({
            "count": len(df),
            "mae": df["mae"].mean(),
            "baseline_mae": df["baseline_mae"].mean(),
            "dir_acc": (df["dir"] == df["truth_dir"]).mean(),
            "baseline_dir_acc": (df["baseline_pred"] == df["truth"]).mean()  # not used; we compute baseline sign below
        })
    overall = per_h.groupby("h", as_index=False).apply(
        lambda d: pd.Series({
            "count": len(d),
            "mae": d["mae"].mean(),
            "baseline_mae": d["baseline_mae"].mean(),
            "dir_acc": (d["dir"] == d["truth_dir"]).mean(),
            "baseline_dir_acc": ( (d["baseline_pred"] - d.groupby(["symbol","anchor_ord"])["truth"].transform(lambda s: s.iloc[0]) > 0).astype(int) == d["truth_dir"] ).mean()
        })
    ).reset_index(drop=True)

    per_symbol = per_h.groupby(["symbol","h"], as_index=False).apply(
        lambda d: pd.Series({
            "count": len(d),
            "mae": d["mae"].mean(),
            "baseline_mae": d["baseline_mae"].mean(),
            "dir_acc": (d["dir"] == d["truth_dir"]).mean(),
            "baseline_dir_acc": ( (d["baseline_pred"] - d.groupby("anchor_ord")["truth"].transform(lambda s: s.iloc[0]) > 0).astype(int) == d["truth_dir"] ).mean()
        })
    ).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_h.to_csv(os.path.join(out_dir, f"per_anchor_h_{ts}.csv"), index=False)
    overall.to_csv(os.path.join(out_dir, f"overall_by_h_{ts}.csv"), index=False)
    per_symbol.to_csv(os.path.join(out_dir, f"per_symbol_by_h_{ts}.csv"), index=False)

    print(f"\n=== {name} — Overall (MAE / DirAcc) by horizon ===")
    cols = ["h","count","baseline_mae","mae","baseline_dir_acc","dir_acc"]
    print(overall[cols].round(4).to_string(index=False))

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for spec in MODEL_SPECS:
        backtest_one_model(spec)

if __name__ == "__main__":
    main()
