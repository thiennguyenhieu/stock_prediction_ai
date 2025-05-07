import sys
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from lightgbm import LGBMRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.stock_config import *

# --- Config ---
DATA_PATH = "data/financial_data_final.csv"
VERSION_TAG = "v1_lgbm_eps_bvps"
MODEL_DIR = os.path.join("models", "lgbm_eps_bvps", VERSION_TAG)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS = [COL_EPS, COL_BVPS]
META_COLS = [COL_SYMBOL, COL_YEAR, COL_QUARTER]

# --- Load Data ---
df = pd.read_csv(DATA_PATH)
features = [col for col in df.columns if col not in META_COLS + TARGETS]

X = df[features]
y_eps = df[TARGETS[0]]
y_bvps = df[TARGETS[1]]

# --- Train-Test Split ---
X_train, X_val, y_train_eps, y_val_eps = train_test_split(X, y_eps, test_size=0.2, random_state=42)
_, _, y_train_bvps, y_val_bvps = train_test_split(X, y_bvps, test_size=0.2, random_state=42)

# --- Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_X.pkl"))

# --- Parameter Tuning ---
def tune_model(X, y):
    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05],
        'max_depth': [4, 6, 8],
        'num_leaves': [31, 63],
        'min_child_samples': [10, 20]
    }
    grid = GridSearchCV(
        LGBMRegressor(random_state=42, verbose=-1),
        param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X, y)
    print("Best params:", grid.best_params_)
    return grid.best_estimator_, grid.best_params_

# --- Train Models with Tuning ---
model_eps, best_params_eps = tune_model(X_train_scaled, y_train_eps)
model_bvps, best_params_bvps = tune_model(X_train_scaled, y_train_bvps)

# --- Save Models and Params ---
joblib.dump(model_eps, os.path.join(MODEL_DIR, "model_eps.pkl"))
joblib.dump(model_bvps, os.path.join(MODEL_DIR, "model_bvps.pkl"))

with open(os.path.join(MODEL_DIR, "best_params_eps.json"), "w") as f:
    json.dump(best_params_eps, f, indent=4)
with open(os.path.join(MODEL_DIR, "best_params_bvps.json"), "w") as f:
    json.dump(best_params_bvps, f, indent=4)

# --- Evaluate ---
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} - MAE: {mae:.2f}, R2: {r2:.4f}")
    return mae, r2

y_pred_eps = model_eps.predict(X_val_scaled)
y_pred_bvps = model_bvps.predict(X_val_scaled)

mae_eps, r2_eps = evaluate(y_val_eps, y_pred_eps, "EPS")
mae_bvps, r2_bvps = evaluate(y_val_bvps, y_pred_bvps, "BVPS")

# --- Save Evaluation Metrics ---
metrics = {
    "EPS": {"mae": round(mae_eps, 2), "r2": round(r2_eps, 4)},
    "BVPS": {"mae": round(mae_bvps, 2), "r2": round(r2_bvps, 4)}
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# --- Feature Importance Visualization ---
def plot_feature_importance(model, feature_names, title, save_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # top 20 features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_feature_importance(model_eps, features, "Top 20 Features - EPS", os.path.join(MODEL_DIR, "feature_importance_eps.png"))
plot_feature_importance(model_bvps, features, "Top 20 Features - BVPS", os.path.join(MODEL_DIR, "feature_importance_bvps.png"))
