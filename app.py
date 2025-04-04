import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulated stock data for testing
VALID_SYMBOLS = ['ACB', 'VCB', 'BID']

# Simulated news feed
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

st.title("📈 Stock Predictor App")

symbol = st.text_input("Enter Stock Symbol (e.g., ACB)", value="ACB")

if st.button("Predict"):
    if symbol.upper() not in VALID_SYMBOLS:
        st.error(f"Symbol '{symbol}' not found in model data.")
    else:
        # --- Generate linear-style fake data ---
        actual_prices = np.linspace(13.5, 15.0, 10)
        predicted_prices = np.linspace(15.1, 16.5, 5)

        # --- Price Chart ---
        st.subheader("Price Forecast")
        fig, ax = plt.subplots()
        ax.plot(range(len(actual_prices)), actual_prices, label="Actual Price")
        ax.plot(range(len(actual_prices), len(actual_prices) + len(predicted_prices)), predicted_prices, label="Predicted Price", linestyle='--')
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # --- Evaluation ---
        st.subheader("📊 Evaluation on Predicted Price")
        latest_prediction = predicted_prices[-1]
        st.markdown(f"**Latest Predicted Price:** {latest_prediction:.2f}")

        # --- News Feed ---
        st.subheader("📰 Related News & Sentiment")
        styled_news = []
        for idx, (headline, sentiment) in enumerate(FAKE_NEWS, 1):
            if sentiment == "positive":
                color = "#2196f3"  # blue
            elif sentiment == "negative":
                color = "#f44336"  # red
            else:
                color = "#ffffff"  # white
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

        sentiment_counts = pd.DataFrame(FAKE_NEWS, columns=["Headline", "Sentiment"])['Sentiment'].value_counts().to_dict()
        positive = sentiment_counts.get('positive', 0)
        negative = sentiment_counts.get('negative', 0)
        neutral = sentiment_counts.get('neutral', 0)
        st.markdown(f"**Sentiment Summary:** {positive} Positive, {negative} Negative, {neutral} Neutral")
