# Stock Price Prediction using XGBoost & Streamlit

This project is an AI-powered stock price prediction tool using XGBoost for forecasting and Streamlit for the user interface.

## Python Version
This project requires **Python 3.12.9**.

## Stock Data Source
This project fetches stock market data using [`vnstock`](https://github.com/thinh-vu/vnstock). 
Requires **vnstock 3.2.0**

## 🚀 How to Run

### 1️ Install Dependencies
Install the required packages:
```sh
pip install -r requirements.txt
```
### 2️ Fetch stock data
Fetch stock data:
```sh
python fetch_stock_data.py
```
This will fetch stock data and save it as `data/.._stock_data.csv`.

### 3️ Train the Model
Before making predictions, train the XGBoost model using historical stock data:
```sh
python model_trainer.py
```
This will train the model and save it as `models/xgboost_model.pkl`.

### 4 Run the Streamlit App
To launch the user interface for stock price prediction, run:
```sh
streamlit run app.py
```
This will start the app in your browser, where you can input stock data and get predictions.

## 🛠 Features
✅ Loads stock market data for analysis.  
✅ Uses XGBoost for stock price prediction.  
✅ Provides a user-friendly UI with Streamlit.  
✅ Saves trained models for future predictions.  

## 📌 Future Improvements
- 📈 Integrate real-time stock data API.
- 🔍 Improve model accuracy with more features.
- 📊 Add visualization for historical trends and predictions.

---
🔗 **Contributions & Feedback**: Feel free to contribute or report any issues!
