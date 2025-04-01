import streamlit as st  
import pandas as pd  
import pickle  

st.title("📈 AI-Powered Stock Price Predictor")  

# Load model  
@st.cache_resource  
def load_xgboost_model():  
    with open("models/xgboost_model.pkl", "rb") as f:  
        return pickle.load(f)  

model = load_xgboost_model()  

# User input  
st.sidebar.header("Stock Data Input")  
open_price = st.sidebar.number_input("Open Price", min_value=0.0)  
high_price = st.sidebar.number_input("High Price", min_value=0.0)  
low_price = st.sidebar.number_input("Low Price", min_value=0.0)  
close_price = st.sidebar.number_input("Close Price", min_value=0.0)  

if st.sidebar.button("Predict"):
    input_data = pd.DataFrame([[open_price, high_price, low_price, close_price]], columns=['Open', 'High', 'Low', 'Close'])  
    prediction = model.predict(input_data)[0]  
    st.subheader(f"Predicted Next Closing Price: ${prediction:.2f}")  