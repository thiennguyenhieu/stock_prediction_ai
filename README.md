# Stock Price Prediction using XGBoost & Streamlit

This project is an AI-powered stock price prediction tool using XGBoost for forecasting and Streamlit for the user interface.

## 🚀 How to Run

### 1️⃣ Install Dependencies
Make sure you have Python installed (Python 3.8+ recommended). Install the required packages:
```sh
pip install -r requirements.txt
```

### 2️⃣ Train the Model
Before making predictions, train the XGBoost model using historical stock data:
```sh
python src/model_trainer.py
```
This will train the model and save it as `models/xgboost_model.pkl`.

### 3️⃣ Run the Streamlit App
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
