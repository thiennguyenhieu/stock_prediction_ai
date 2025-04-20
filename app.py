import streamlit as st
import plotly.graph_objects as go

from src.fetch_historical_data import fetch_recent_price, fetch_all_symbols
from src.historical_inference import get_prediction
from src.utility import graham_valuation, pe_valuation, pb_valuation

# ------------------ Constants ------------------
valid_symbols_with_info = fetch_all_symbols()  # Return [symbol, exchange, organ name]
valid_symbols = valid_symbols_with_info.iloc[:, 0].tolist()
FORECAST_DAYS = {"7 Days": 7, "30 Days": 30, "60 Days": 60}

# ------------------ Helper Functions ------------------
def get_stock_info_by_symbol(symbol: str, df) -> tuple:
    row = df[df.iloc[:, 0] == symbol]
    if not row.empty:
        exchange = row.iloc[0, 1]
        organ_name = row.iloc[0, 2]
        return exchange, organ_name
    return None, None

# ------------------ UI Sidebar ------------------
st.title("📈 Stock Predictor App")

with st.sidebar:
    st.header("📊 Stock Settings")
    symbol = st.text_input("Enter Stock Symbol (e.g., ACB)", value="ACB").upper()
    forecast_label = st.selectbox("Forecast Interval", list(FORECAST_DAYS.keys()), index=0)
    predict_clicked = st.button("Predict")

# ------------------ Main Logic ------------------
if not predict_clicked:
    st.info("👈 Please use the sidebar to select stock symbol and forecast options, then click **Predict**.")

if predict_clicked:
    if symbol not in valid_symbols:
        st.error(f"Symbol '{symbol}' not found in model data.")
    else:
        forecast_days = FORECAST_DAYS[forecast_label]

        df_real = fetch_recent_price(symbol)
        df_pred = get_prediction(symbol, forecast_days)

        exchange, organ_name = get_stock_info_by_symbol(symbol, valid_symbols_with_info)
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader(f"🏢 {symbol} — {organ_name}")
        st.markdown(f"**Exchange:** {exchange}")

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
        fig.add_trace(go.Scatter(
            x=[df_real.iloc[-1, 0], df_pred.iloc[0, 0]],
            y=[df_real.iloc[-1, 1], df_pred.iloc[0, 1]],
            mode='lines',
            name='',
            line=dict(color='orange', dash='dash'),
            hoverinfo='skip',
            showlegend=False
        ))
        fig.update_layout(
            xaxis_title='Date', yaxis_title='Close Price',
            legend=dict(x=0, y=1), hovermode='x', template='plotly_white',
            margin=dict(t=10, b=40, l=40, r=10), height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Evaluation ---
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader("📊 Valuation Summary")

        eps = 0
        bvps = 0

        graham_value = graham_valuation(eps)
        pe_value = pe_valuation(eps)
        pb_value = pb_valuation(bvps)

        st.markdown(f"**Predicted Intrinsic Value (Graham):** {graham_value:.2f} VND")
        st.markdown(f"**P/E-based Valuation:** {pe_value:.2f} VND")
        st.markdown(f"**P/B-based Valuation:** {pb_value:.2f} VND")
