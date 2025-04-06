# 📊 Stock Price Prediction using CNN-LSTM & Streamlit

This project is an AI-powered stock price prediction tool that uses a **CNN-LSTM** architecture for forecasting and **Streamlit** for a clean, interactive UI. It also integrates **PhoBERT**, a pretrained Vietnamese news sentiment analysis model from Hugging Face, to assess market sentiment based on news headlines.

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

This project fetches stock market data using [`vnstock`](https://github.com/thinh-vu/vnstock).

---

## 🤖 Sentiment Analysis with PhoBERT

The app uses `vinai/phobert-base` (or a fine-tuned version of it) for Vietnamese sentiment analysis on financial news. Headlines are labeled as:

- `positive`
- `neutral`
- `negative`

You can auto-label raw headlines using keyword-based heuristics or train your own classifier with the included training script:

```sh
python src/news_model_trainer.py
```
- 📂 Path: `models/phobert-finance/`
- 📥 Download: [Google Drive](https://drive.google.com/your-link-here)

---

## 📈 Stock Prediction with CNN-LSTM

The app uses a hybrid **CNN-LSTM** deep learning model to forecast future stock prices based on historical data. The model is trained using `vnstock`-sourced data.

You can train the CNN-LSTM stock model with the following scripts:

```sh
python src/fetch_stock_data.py
```
- 📂 Path: `models/cnn_lstm_model.h5`
- 📥 Download: [Google Drive](https://drive.google.com/your-link-here)

```sh
python src/stock_model_trainer.py
```
This will train the stock model and save it as `models/cnn_lstm_model.h5`.

---

## 🚀 Run the App

Launch the interactive dashboard with:
```sh
streamlit run app.py
```
This will open the app in your browser where you can explore data, predictions, and sentiment.

---

### 🧠 Features
✅ Loads and visualizes stock market data

✅ Predicts stock prices using CNN-LSTM deep learning

✅ Provides an interactive UI with Streamlit

✅ Visualizes historical & forecasted trends

✅ Integrates PhoBERT for sentiment analysis on financial news

✅ Displays bullish/bearish outlook based on prediction + sentiment

✅ Saves trained models for future use

---
🔗 **Contributions & Feedback**: Feel free to contribute or report any issues!