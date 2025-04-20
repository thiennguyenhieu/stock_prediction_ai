# 📊 Stock Price Prediction using CNN-LSTM & Streamlit

This project is an AI-powered stock price prediction tool that uses a **CNN-LSTM** architecture for forecasting and **Streamlit** for a clean, interactive UI. It is designed to help visualize historical stock data and predict future trends using deep learning.

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

## 📈 Stock Prediction with CNN-LSTM

The app uses a hybrid **CNN-LSTM** deep learning model to forecast future stock prices based on historical data.

You can train the model using the following scripts:

```sh
python src/encode_symbols.py
python src/fetch_historical_data.py
python src/train_historical_model.py
```

> 🔒 **Model Location**:  
> The trained model and scalers are saved in the local folder:
```
models/cnn_lstm_close_regression/VERSION_TAG
```
---

## 🚀 Run the App

Launch the interactive dashboard:

```sh
streamlit run app.py
```

The app will open in your default browser where you can:
- Select stock symbols
- Forecast stock prices

---

### 🧠 Features

✅ Loads and visualizes stock market data  
✅ Predicts stock prices using CNN-LSTM  
✅ Interactive interface built with Streamlit  
✅ Historical vs predicted chart visualization  
✅ Local model management (no cloud dependency)  
✅ Clean and lightweight design  

---

## 🔗 Contributions & Feedback

Feel free to submit pull requests or report issues!