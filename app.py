import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from src.fetch_stock_data import fetch_historical_data_for_display, fetch_prediction_data_for_display
from src.fetch_news_data import fetch_news_with_sentiment

# ------------------ Constants ------------------

VALID_SYMBOLS = ['ACB', 'VCB', 'BID']
LOOKBACK_DAYS = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 60 Days": 60}
FORECAST_DAYS = {"7 Days": 7, "30 Days": 30, "60 Days": 60}


# ------------------ Helper Functions ------------------

def evaluate_trend(actual: float, predicted: float) -> str:
    if predicted > actual:
        return "Bullish"
    elif predicted < actual:
        return "Bearish"
    else:
        return "Neutral"

def analyze_sentiment(news_df: pd.DataFrame) -> str:
    sentiment_counts = news_df['Sentiment'].value_counts().to_dict()
    pos = sentiment_counts.get('positive', 0)
    neg = sentiment_counts.get('negative', 0)
    neu = sentiment_counts.get('neutral', 0)
    diff = abs(pos - neg)

    if (pos == 0 and neg == 0) or diff <= 2 or neu > max(pos, neg):
        return "Neutral"
    elif pos > neg:
        return "Bullish"
    else:
        return "Bearish"

def summarize_signal(price_trend: str, sentiment_trend: str) -> str:
    if price_trend == "Bullish" and sentiment_trend == "Bullish":
        return "🚀 Strong Bullish"
    elif price_trend == "Bearish" and sentiment_trend == "Bearish":
        return "🔻 Strong Bearish"
    elif price_trend != sentiment_trend:
        return "⚖️ Mixed Signal – Monitor Closely"
    else:
        return "ℹ️ Neutral"

def render_news_table(news_df: pd.DataFrame):
    styled_news = []
    for idx, row in news_df.iterrows():
        color = {
            "positive": "#2196f3",
            "negative": "#f44336",
            "neutral": "#9e9e9e"
        }.get(row['Sentiment'], "#ffffff")

        styled_news.append(
            f"<tr><td>{idx + 1}</td><td>{row['Headline']}</td><td style='color:{color}'>{row['Sentiment'].title()}</td></tr>"
        )

    news_table_html = f"""
    <table style='width:100%; border-collapse: collapse;'>
        <thead>
            <tr style='text-align: left;'>
                <th>#</th>
                <th>Headline</th>
                <th>Sentiment</th>
            </tr>
        </thead>
        <tbody>
            {''.join(styled_news)}
        </tbody>
    </table>
    """
    st.markdown(news_table_html, unsafe_allow_html=True)

# ------------------ UI Sidebar ------------------

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
        # --- Data Fetching ---
        today = date.today()
        lookback_days = LOOKBACK_DAYS[lookback_label]
        forecast_days = FORECAST_DAYS[forecast_label]
        predict_start_date = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        df_real = fetch_historical_data_for_display(symbol)
        df_pred = fetch_prediction_data_for_display(symbol, predict_start_date, forecast_days)
        news_df = fetch_news_with_sentiment(symbol)

        # --- Price Forecast Plot ---
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader("📉 Price Forecast")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_real.iloc[:, 0], y=df_real.iloc[:, 1],
            mode='lines+markers', name='Actual Price',
            marker=dict(symbol='circle', color='blue'),
            line=dict(color='blue'),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Actual Price: %{y:.2f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=df_pred.iloc[:, 0], y=df_pred.iloc[:, 1],
            mode='lines+markers', name='Predicted Price',
            marker=dict(symbol='x', color='orange'),
            line=dict(color='orange', dash='dash'),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Predicted Price: %{y:.2f}<extra></extra>'
        ))
        fig.update_layout(
            xaxis_title='Date', yaxis_title='Close Price',
            legend=dict(x=0, y=1), hovermode='x', template='plotly_white',
            margin=dict(t=10, b=40, l=40, r=10), height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Evaluation ---
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader("📊 Evaluation on Predicted Price")
        latest_prediction = df_pred.iloc[-1, 1]
        st.markdown(f"**Latest Predicted Price:** {latest_prediction:.2f}")

        # --- News & Sentiment ---
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader("📰 Related News & Sentiment")
        render_news_table(news_df)

        # --- Market Summary ---
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader("📋 Market Outlook Summary")

        latest_actual = df_real.iloc[-1, 1]
        trend = evaluate_trend(latest_actual, latest_prediction)
        sentiment_trend = analyze_sentiment(news_df)
        overall_signal = summarize_signal(trend, sentiment_trend)

        st.info(f"📈 **Prediction Trend:** {trend}")
        st.info(f"📰 **News Sentiment Trend:** {sentiment_trend}")

        if "Bullish" in overall_signal:
            st.success(f"✅ **Overall Signal:** {overall_signal}")
        elif "Bearish" in overall_signal:
            st.error(f"🔻 **Overall Signal:** {overall_signal}")
        else:
            st.warning(f"⚖️ **Overall Signal:** {overall_signal}")
