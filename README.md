# 📊 Stock Price & Financial Forecasting with CNN-LSTM, LightGBM & Streamlit

This project is an AI-powered stock analytics and prediction tool using CNN-LSTM for multi-step Close price forecasting, and LightGBM for predicting fundamental indicators like EPS and BVPS. The tool is deployed through an interactive Streamlit dashboard.

---

## 🐍 Python Version

> Requires **Python 3.11.9**

---

## ⚙️ Installation

Install all dependencies using:

```sh
pip install -r requirements.txt
```

---

## 📦 Data Source

This project fetches stock market data using [`vnstock`](https://github.com/thinh-vu/vnstock), including:
- Historical stock prices
- Financial ratios and reports

---

## 📈 Close Price Prediction (CNN-LSTM)

The deep learning model predicts the future Close prices using a hybrid CNN-LSTM architecture with attention mechanisms and multi-step forecasting.

🔧 Train Model
```sh
python src/encode_symbols.py
python src/fetch_historical_data.py
python src/train_historical_model.py
```

✅ Outputs:
```
models/cnn_lstm_close_regression/<VERSION_TAG>/
```
---

## 💡 EPS & BVPS Forecasting (LightGBM)

The project includes a financial forecasting module to predict EPS and BVPS using LightGBM regressors trained on quarterly financial reports.

🔧 Train Financial Model
```sh
python src/fetch_financial_data.py
python src/train_financial_model.py
```

✅ Outputs:
```
models/lgbm_eps_bvps/<VERSION_TAG>/
```

🎯 Model Targets
- EPS (VND) — Earnings Per Share
- BVPS (VND) — Book Value Per Share

---

## 🚀 Launch the App

Launch the interactive dashboard:

```sh
streamlit run app.py
```

You can:
- Select stock symbols
- Forecast stock Close prices
- Predict EPS and BVPS
- See derived valuations using:
    - Graham Formula
    - P/E-based valuation
    - P/B-based valuation

---

### 🧠 Features

- CNN-LSTM model for Close price forecasting
- LightGBM for EPS & BVPS prediction
- Multi-step Close prediction with attention
- Valuation analysis using forecasted fundamentals
- Streamlit interface with interactive charts
- Local model management, no cloud dependency

---

## 🔗 Contributions & Feedback
- Pull requests and feedback are welcome!
- Please report any bugs or issues for improvement.
