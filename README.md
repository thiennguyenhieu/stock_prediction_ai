# 📊 Stock Price Prediction using CNN-LSTM & Streamlit

This project is an AI-powered stock price prediction tool that uses a **CNN-LSTM** architecture for forecasting and **Streamlit** for a clean, interactive UI.

---

## 🐍 Python Version

> Requires **Python 3.11.9**

---

## 📦 Data Source

This project fetches stock market data using [`vnstock`](https://github.com/thinh-vu/vnstock).

> Requires **vnstock==3.2.3**

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 2️⃣ Fetch Stock Data
Fetch stock data:
```sh
python src\fetch_stock_data.py
```
This will fetch stock data and save it as `data/trained_stock_data.csv`,`data/test_stock_data.csv`.

### 3️⃣ Train the Model
Before making predictions, train the CNN-LSTM model using historical stock data:
```sh
python src\stock_model_trainer.py
```
This will train the model and save it as `models/cnn_lstm_model.h5`.

### 4️⃣ Run the Streamlit App
To launch the user interface for stock price prediction, run:
```sh
streamlit run app.py
```
This will start the app in your browser, where you can input stock data and get predictions.

## 🧠 Features
✅ Loads and visualizes stock market data

✅ Predicts stock prices using CNN-LSTM deep learning

✅ Provides an interactive UI with Streamlit

✅ Visualizes historical & forecasted trends

✅ Saves trained models for future use

## 📌 Future Improvements
📈 Integrate real-time stock data APIs

🧪 Improve model performance with technical indicators

🔍 Enhance sentiment analysis with live news

---
🔗 **Contributions & Feedback**: Feel free to contribute or report any issues!
