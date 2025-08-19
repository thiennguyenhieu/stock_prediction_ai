import numpy as np
import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fetch_historical_data import process_symbol, post_process_data
from src.historical_inference_v1 import predict_close_price_series
from src.constants import COL_CLOSE, COL_TIME  # if COL_TIME exists

EPS = 1e-8  # for stable % metrics


def walk_forward_backtest(symbol: str, months: int = 6, horizon: int = 14,
                          min_history: int = 240, max_windows: int | None = None,
                          verbose: bool = True):
    """
    Walk-forward backtest for the CNN-LSTM model.

    Args:
        symbol (str): Stock ticker (e.g., "ACB").
        months (int): How many recent months of data to evaluate (default=6).
        horizon (int): Forecast horizon used by the model (default=14).
        min_history (int): Minimum rows required in the history window before a forecast.
                           Keeps model feature-engineering safe (lags/sequence length).
        max_windows (int|None): Optional cap on #evaluation windows for speed.
        verbose (bool): Print progress.

    Returns:
        dict: metrics at {1,7,14} with MAE, MAPE, DirectionAcc, N (sample count).
    """
    # --- Load & prep data (last `months`) ---
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(months=months)

    if verbose:
        print(f"[DATA] Range: {start.date()} → {end.date()} ({months} months)")

    df = process_symbol(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    df = post_process_data(df).reset_index(drop=True)

    if len(df) < max(min_history + horizon, 200):
        raise ValueError(f"Not enough data for backtest: {len(df)} rows "
                         f"(need ≥ {max(min_history + horizon, 200)}).")

    # --- Prepare accumulators ---
    steps_eval = [1, 7, 14]
    steps_eval = [s for s in steps_eval if s <= horizon]

    errors = {s: [] for s in steps_eval}        # absolute price error
    mape   = {s: [] for s in steps_eval}        # percent error
    dacc   = {s: [] for s in steps_eval}        # direction accuracy vs last hist price

    # --- Sliding daily windows over the last N months ---
    # We will evaluate from index i where 'hist' ends at i-1 and we forecast next `horizon` days.
    # Ensure 'hist' length >= min_history.
    start_idx = max(min_history, 1)
    end_idx_exclusive = len(df) - horizon + 1  # last start idx that allows horizon steps

    # Focus on the "last months" region but still slide daily
    # (If you want strictly last months, start_idx is fine because df is already clipped to last N months)
    window_indices = range(start_idx, end_idx_exclusive)

    if max_windows is not None:
        # take only the last `max_windows` windows for speed
        window_indices = list(window_indices)[-max_windows:]

    if verbose:
        print(f"[EVAL] Windows: {len(list(window_indices))} | Horizon: {horizon}")
        print(f"[EVAL] Steps: {steps_eval}")

    for k, idx in enumerate(window_indices, start=1):
        hist = df.iloc[:idx].copy()                       # history up to idx-1
        true_future = df.iloc[idx: idx + horizon][COL_CLOSE].to_numpy()  # next horizon closes

        last_close = float(hist[COL_CLOSE].iloc[-1])

        # Run model forecast on the 'hist' only
        try:
            pred_df = predict_close_price_series(hist, forecast_steps=horizon, debug=False)
        except Exception as e:
            if verbose:
                print(f"[WARN] forecast failed at idx={idx}: {e}")
            continue

        if COL_CLOSE not in pred_df.columns:
            if verbose:
                print(f"[WARN] prediction missing {COL_CLOSE} at idx={idx} (skipping)")
            continue

        pred_future = pred_df[COL_CLOSE].to_numpy()
        if len(pred_future) < horizon or len(true_future) < horizon:
            # Safety check
            continue

        # Collect metrics at steps t+1, t+7, t+14
        for s in steps_eval:
            y_true = float(true_future[s - 1])
            y_pred = float(pred_future[s - 1])

            # absolute error
            errors[s].append(abs(y_true - y_pred))
            # percentage error
            mape[s].append(abs(y_true - y_pred) / max(EPS, abs(y_true)))
            # direction: compare sign of change from last_close
            true_up = (y_true - last_close) > 0
            pred_up = (y_pred - last_close) > 0
            dacc[s].append(1.0 if (true_up == pred_up) else 0.0)

        if verbose and k % 50 == 0:
            print(f"  processed {k} windows...")

    # --- Aggregate ---
    metrics = {}
    for s in steps_eval:
        n = len(errors[s])
        if n > 0:
            metrics[s] = {
                "MAE": float(np.mean(errors[s])),
                "MAPE": float(np.mean(mape[s]) * 100.0),
                "DirectionAcc": float(np.mean(dacc[s])),
                "N": int(n),
            }
        else:
            metrics[s] = {}

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest for CNN-LSTM model")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol (e.g., ACB)")
    parser.add_argument("--months", type=int, default=6, help="Number of months to backtest (default=6)")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon (default=14)")
    parser.add_argument("--min_history", type=int, default=240, help="Minimum history length for each window (default=240)")
    parser.add_argument("--max_windows", type=int, default=None, help="Optional cap on number of windows")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = parser.parse_args()

    verbose = not args.quiet
    if verbose:
        print(f"Running walk-forward backtest for {args.symbol} "
              f"({args.months} months, horizon={args.horizon}, min_history={args.min_history})...")

    metrics = walk_forward_backtest(
        args.symbol,
        months=args.months,
        horizon=args.horizon,
        min_history=args.min_history,
        max_windows=args.max_windows,
        verbose=verbose
    )

    print("\n=== Backtest Results ===")
    for step in [1, 7, 14]:
        vals = metrics.get(step, {})
        if vals:
            print(f"t+{step}: MAE={vals['MAE']:.2f}, MAPE={vals['MAPE']:.2f}%, "
                  f"DirectionAcc={vals['DirectionAcc']:.2f} (N={vals['N']})")
        else:
            print(f"t+{step}: No results")


if __name__ == "__main__":
    # Example:
    #   python src/backtest.py --symbol ACB --months 24 --horizon 14 --min_history 240
    main()
