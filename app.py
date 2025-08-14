import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.fetch_historical_data import fetch_recent_price
from src.fetch_financial_data import fetch_financial_single_symbol
from src.fetch_general_info import fetch_all_symbols, fetch_company_overview, fetch_dividend
from src.historical_inference_v1 import get_close_prediction
from src.stock_config import *

# ------------------ Constants ------------------
valid_symbols_with_info = fetch_all_symbols()  # Return [symbol, exchange, organ name]
valid_symbols = valid_symbols_with_info[valid_symbols_with_info.iloc[:, 0].str.len() == 3].iloc[:, 0].tolist()

# ------------------ Helper Functions ------------------
def get_stock_info_by_symbol(symbol: str, df) -> tuple:
    row = df[df.iloc[:, 0] == symbol]
    if not row.empty:
        exchange = row.iloc[0, 1]
        organ_name = row.iloc[0, 2]
        return exchange, organ_name
    return None, None

def highlight_value(val):
    if val > 0:
        return 'color: green; font-weight: bold;'
    elif val < 0:
        return 'color: red; font-weight: bold;'
    return ''  # no style for exactly zero

def format_thousands(v):
    return "" if pd.isna(v) else f"{v:,.0f}"

# ------------------ UI layout ------------------
st.set_page_config(layout="wide") # set wide layout

st.title("📈 Stock Predictor App")

with st.sidebar:
    st.header("📊 Stock Settings")
    symbol = st.text_input("Enter Stock Symbol (e.g., ACB)", value="ACB").upper()
    apply_clicked = st.button("Apply")

# ------------------ Main Logic ------------------
if not apply_clicked:
    st.info("👈 Please use the sidebar to select stock symbol and forecast options, then click **Apply**.")

if apply_clicked:
    if symbol not in valid_symbols:
        st.error(f"Symbol '{symbol}' not found in model data.")
    else:
        df_real = fetch_recent_price(symbol)
        df_pred = get_close_prediction(symbol, 14)

        exchange, organ_name = get_stock_info_by_symbol(symbol, valid_symbols_with_info)
        shares_outstanding, industry = fetch_company_overview(symbol)
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader(f"🏢 {organ_name} ({exchange}: {symbol})")
        st.markdown(f"**Industry:** {industry}")
        st.markdown(f"**Shares outstanding:** {shares_outstanding:,}")

        # --- Price Forecast Plot ---
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader("📉 Price forecast for the next 14 days")

        fig = go.Figure()

        # --- Actual Close Price ---
        fig.add_trace(go.Scatter(
            x=df_real[COL_TIME], y=df_real[COL_CLOSE],
            mode='lines+markers', name='Actual Price',
            marker=dict(symbol='circle', color='blue'),
            line=dict(color='blue'),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Actual Price: %{y:.2f}<extra></extra>',
            yaxis='y1'
        ))

        # --- Predicted Close Price ---
        fig.add_trace(go.Scatter(
            x=df_pred[COL_TIME], y=df_pred[COL_CLOSE],
            mode='lines+markers', name='Predicted Price',
            marker=dict(symbol='x', color='orange'),
            line=dict(color='orange', dash='dash'),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Predicted Price: %{y:.2f}<extra></extra>',
            yaxis='y1'
        ))

        # --- Connector Line ---
        fig.add_trace(go.Scatter(
            x=[df_real[COL_TIME].iloc[-1], df_pred[COL_TIME].iloc[0]],
            y=[df_real[COL_CLOSE].iloc[-1], df_pred[COL_CLOSE].iloc[0]],
            mode='lines', line=dict(color='orange', dash='dash'),
            hoverinfo='skip', showlegend=False, yaxis='y1'
        ))

        # --- Volume Bars ---
        fig.add_trace(go.Bar(
            x=df_real[COL_TIME], y=df_real[COL_VOLUME],
            name='Volume', marker_color='rgba(100, 100, 255, 0.3)',
            yaxis='y2', opacity=0.5
        ))

        # --- Layout Settings ---
        fig.update_layout(
            xaxis_title='Date',
            yaxis=dict(title='Close Price', side='left'),
            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
            legend=dict(x=0, y=1),
            hovermode='x unified',
            template='plotly_white',
            margin=dict(t=10, b=40, l=40, r=10),
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Dividend ---
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader("💸 Dividend History")

        df_dividend = fetch_dividend(symbol).head(5).reset_index(drop=True)
        html = (
            df_dividend
            .style
            .hide(axis="index")
            .to_html()
        )

        st.markdown(html, unsafe_allow_html=True)

        # --- Financial Report Table --- 
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.subheader("📑 Quarterly Financial Report")
        
        df_finance_display = fetch_financial_single_symbol(symbol).head(8).reset_index(drop=True)

        html = (
            df_finance_display
            .style
            .hide(axis="index")
            .format({COL_ATTRIBUTE: "{:,.0f}", COL_REVENUE: "{:,.0f}"})
            .applymap(highlight_value, subset=[COL_ATTRIBUTE_YOY])
            .to_html()
        )

        st.markdown(html, unsafe_allow_html=True)
