# Inference for 14-day model trained on *vol-normalized* returns.
# Loads calibration (a_h, b_h) to adjust raw price path per horizon.
import os, sys, json, joblib, numpy as np, pandas as pd
from pandas.tseries.offsets import BDay
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import *                      # COL_CLOSE, COL_TIME_ORDINAL, etc.
from src.fetch_historical_data import process_symbol, post_process_data

VERSION_TAG = "v2_volnorm_focal_hw_h14"          # must match training tag above
MODEL_DIR = os.path.join("models", "cnn_lstm_close_regression", VERSION_TAG)

def _build_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    # df already has COL_TIME_ORDINAL from post_process_data
    # parity with training: delta + lags + vol features
    df = df.copy()
    # 1d log return & 20d vol for de-normalization
    df["logp"] = np.log(df[COL_CLOSE].astype(float))
    df["r1"]   = df["logp"].diff()
    df["vol20"]= df["r1"].rolling(20, min_periods=10).std().clip(lower=1e-6)

    df["delta_close"] = df[COL_CLOSE].diff()
    for lag in range(1, 31):
        df[f"lag_close_{lag}"] = df[COL_CLOSE].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df

def load_artifacts():
    model = load_model(os.path.join(MODEL_DIR, "cnn_lstm_forecast_model.keras"), compile=False)
    scaler_X   = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
    scaler_nrt = joblib.load(os.path.join(MODEL_DIR, "scaler_nret.pkl"))
    meta = json.load(open(os.path.join(MODEL_DIR, "metadata.json"), "r"))
    calib_path = os.path.join(MODEL_DIR, "calibration.json")
    calib = json.load(open(calib_path, "r")) if os.path.exists(calib_path) else {"a": None, "b": None}
    return model, scaler_X, scaler_nrt, meta, calib

def predict_close_price_series(raw_df: pd.DataFrame, forecast_steps: int = 14, debug: bool = False) -> pd.DataFrame:
    model, scaler_X, scaler_nrt, meta, calib = load_artifacts()
    input_features = meta["input_features"]
    H = int(meta["forecast_horizon"])
    L = int(meta["input_seq_len"])
    if forecast_steps > H:
        raise ValueError(f"Requested {forecast_steps} steps, but model horizon is {H}.")

    df = _build_features_for_inference(raw_df.copy())
    missing = [c for c in input_features if c not in df.columns]
    if missing: raise ValueError(f"Missing input features: {missing}")
    if len(df) < L: raise ValueError(f"Not enough rows for prediction sequence. Need {L}, have {len(df)}")

    scaled = df.copy()
    scaled[input_features] = scaler_X.transform(scaled[input_features])
    X_seq = scaled[input_features].iloc[-L:].values.astype(np.float32).reshape(1, L, len(input_features))

    # forward pass -> normalized returns vector
    y_nret_sc, _ = model.predict(X_seq, verbose=0)          # (1,H)
    y_nret = scaler_nrt.inverse_transform(y_nret_sc)[0]      # (H,)

    # de-normalize to raw returns using last vol20 and sqrt(h)
    last_close = float(df[COL_CLOSE].iloc[-1])               # feature-aligned last close
    last_vol20 = float(df["vol20"].iloc[-1])
    scale = (last_vol20 * np.sqrt(np.arange(1, H+1, dtype=np.float32)))
    y_ret = y_nret[:forecast_steps] * scale[:forecast_steps]

    # reconstruct price path
    logp0 = np.log(last_close)
    log_path = logp0 + np.cumsum(y_ret)
    y_close = np.exp(log_path)

    # apply per-h calibration if present
    if calib.get("a") and calib.get("b"):
        a = np.asarray(calib["a"], dtype=np.float32)[:forecast_steps]
        b = np.asarray(calib["b"], dtype=np.float32)[:forecast_steps]
        y_close = (y_close * a) + b
        y_close = np.maximum(y_close, 0.0)

    return pd.DataFrame({COL_CLOSE: y_close})

# Public API (keeps ordinals; no COL_TIME output)
def get_close_prediction(symbol: str, interval: int = 14) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    df = process_symbol(symbol, start, end)
    if df is None or df.empty:
        print(f"[WARN] Empty data for {symbol}"); return pd.DataFrame()
    df = post_process_data(df)  # -> builds COL_TIME_ORDINAL; drops COL_TIME
    if len(df) < 240:
        print(f"[WARN] Not enough data for {symbol}"); return pd.DataFrame()

    fc = predict_close_price_series(df, forecast_steps=interval, debug=False)

    # attach ordinal dates derived from last feature row
    last_ord = int(df[COL_TIME_ORDINAL].iloc[-1])
    last_ts  = pd.Timestamp.fromordinal(last_ord)
    dates = pd.bdate_range(start=last_ts + BDay(1), periods=interval, freq=BDay())
    fc.insert(0, COL_TIME_ORDINAL, dates.map(pd.Timestamp.toordinal))
    return fc

if __name__ == "__main__":
    print(get_close_prediction("ABT", 14))
