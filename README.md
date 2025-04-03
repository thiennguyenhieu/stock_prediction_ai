# Stock Price Prediction using CNN-LSTM architecture & Streamlit

This project is an AI-powered stock price prediction tool using CNN-LSTM architecture for forecasting and Streamlit for the user interface.

## Python Version
This project requires **Python 3.12.9**.

## Stock Data Source
This project fetches stock market data using [`vnstock`](https://github.com/thinh-vu/vnstock). 
Requires **vnstock 3.2.3**

## 🚀 How to Run

### 1. Install Dependencies
Install the required packages:
```sh
pip install -r requirements.txt
```
### 2. Fetch stock data
Fetch stock data:
```sh
python src\fetch_stock_data.py
```
This will fetch stock data and save it as `data/trained_stock_data.csv`,`data/test_stock_data.csv`.

### 3. Train the Model
Before making predictions, train the CNN-LSTM model using historical stock data:
```sh
python src\model_trainer.py
```
This will train the model and save it as `models/cnn_lstm_model.h5`.

### 4. Run the Streamlit App
To launch the user interface for stock price prediction, run:
```sh
streamlit run app.py
```
This will start the app in your browser, where you can input stock data and get predictions.

## 🛠 Features
✅ Loads stock market data for analysis.
✅ Uses CNN-LSTM for stock price prediction.
✅ Provides a user-friendly UI with Streamlit.
✅ Saves trained models for future predictions. 
✅ Add visualization for historical trends and predictions.

## 📌 Future Improvements
📈 Integrate real-time stock data API.
🔍 Improve model accuracy with more features.

---
🔗 **Contributions & Feedback**: Feel free to contribute or report any issues!
