import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from src.fetch_stock_data import fetch_historical_data_for_display, fetch_prediction_data_for_display

# ------------------ Constants ------------------

VALID_SYMBOLS = ['ACB', 'VCB', 'BID']
LOOKBACK_DAYS = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 60 Days": 60}
FORECAST_DAYS = {"7 Days": 7, "30 Days": 30, "60 Days": 60}

FAKE_NEWS = [
    ("ACB announces strong Q1 earnings", "positive"),
    ("Market volatility impacts banking stocks", "negative"),
    ("Analysts upgrade ACB outlook to Buy", "positive"),
    ("Central bank policies tighten liquidity", "negative"),
    ("Dividend announcement pleases investors", "positive"),
    ("Banking sector sees cautious optimism", "positive"),
    ("Loan defaults increase slightly", "negative"),
    ("Stock price correction expected next quarter", "negative"),
    ("Digital banking rollout continues smoothly", "positive"),
    ("Foreign investors increase holdings", "positive")
]

# ------------------ Sidebar ------------------

st.title("📈 Stock Predictor App")

with st.sidebar:
    st.header("📊 Stock Settings")

    symbol = st.text_input("Enter Stock Symbol (e.g., ACB)", value="ACB").upper()
    lookback_label = st.selectbox("Lookback Interval", list(LOOKBACK_DAYS.keys()), index=0)
    forecast_label = st.selectbox("Forecast Interval", list(FORECAST_DAYS.keys()), index=0)
    predict_clicked = st.button("Predict")

# ------------------ Main Logic ------------------

if not predict_clicked:
    st.info("👈 Please use the sidebar to select stock symbol and forecast options, then click **Predict**.")

if predict_clicked:
    if symbol not in VALID_SYMBOLS:
        st.error(f"Symbol '{symbol}' not found in model data.")
    else:
        today = date.today()
        lookback_days = LOOKBACK_DAYS.get(lookback_label, 7)
        forecast_days = FORECAST_DAYS.get(forecast_label, 7)

        # Compute start date for prediction (backward-looking)
        predict_start_date = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Fetch data
        df_real = fetch_historical_data_for_display(symbol)
        df_pred = fetch_prediction_data_for_display(symbol, predict_start_date, forecast_days)

        # ------------------ Price Chart ------------------
        st.subheader("📉 Price Forecast")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_real.iloc[:, 0], df_real.iloc[:, 1], label="Actual Price", marker='o')
        ax.plot(df_pred.iloc[:, 0], df_pred.iloc[:, 1], label="Predicted Price", linestyle='--', marker='x')
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ------------------ Evaluation ------------------
        st.subheader("📊 Evaluation on Predicted Price")
        latest_prediction = df_pred.iloc[-1, 1]
        st.markdown(f"**Latest Predicted Price:** {latest_prediction:.2f}")

        # ------------------ News & Sentiment ------------------
        st.subheader("📰 Related News & Sentiment")
        styled_news = []
        for idx, (headline, sentiment) in enumerate(FAKE_NEWS, 1):
            color = {
                "positive": "#2196f3",
                "negative": "#f44336"
            }.get(sentiment, "#ffffff")

            styled_news.append(
                f"<tr><td>{idx}</td><td>{headline}</td><td style='color:{color}'>{sentiment.title()}</td></tr>"
            )

        news_table_html = """
        <table style='width:100%; border-collapse: collapse;'>
            <thead>
                <tr style='text-align: left;'>
                    <th>#</th>
                    <th>Headline</th>
                    <th>Sentiment</th>
                </tr>
            </thead>
            <tbody>
        """ + "\n".join(styled_news) + """
            </tbody>
        </table>
        """
        st.markdown(news_table_html, unsafe_allow_html=True)

        # ------------------ Sentiment Summary ------------------
        sentiment_counts = pd.DataFrame(FAKE_NEWS, columns=["Headline", "Sentiment"])['Sentiment'].value_counts().to_dict()
        positive = sentiment_counts.get('positive', 0)
        negative = sentiment_counts.get('negative', 0)
        neutral = sentiment_counts.get('neutral', 0)
        st.markdown(f"**Sentiment Summary:** {positive} Positive, {negative} Negative, {neutral} Neutral")
