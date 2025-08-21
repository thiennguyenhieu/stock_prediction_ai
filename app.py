import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.fetch_historical_data import fetch_recent_price
from src.fetch_financial_data import fetch_income_statement, fetch_ratio
from src.fetch_general_info import fetch_all_symbols, fetch_company_overview, fetch_dividend
from src.historical_inference_v1 import get_close_prediction
from src.stock_analysis import get_completion
from src.constants import *  # VI_STRINGS, COL_*, PROMPT_TEMPLATE

# ---------- Optional caching ----------
@st.cache_data(show_spinner=False)
def load_recent_price(symbol: str):
    return fetch_recent_price(symbol)

@st.cache_data(show_spinner=False)
def load_prediction(symbol: str, horizon: int = 14):
    return get_close_prediction(symbol, horizon)

@st.cache_data(show_spinner=False)
def load_dividend(symbol: str):
    return fetch_dividend(symbol)

@st.cache_data(show_spinner=False)
def load_financials_income_statement(symbol: str):
    return fetch_income_statement(symbol)

@st.cache_data(show_spinner=False)
def load_financials_ratio(symbol: str):
    return fetch_ratio(symbol)

def run_completion(prompt: str):
    return get_completion(prompt)

# ---------- Helpers ----------
def get_stock_info_by_symbol(symbol: str, df) -> tuple:
    row = df[df.iloc[:, 0] == symbol]
    if not row.empty:
        exchange = row.iloc[0, 1]
        organ_name = row.iloc[0, 2]
        return exchange, organ_name
    return None, None

# ---------- Static data ----------
valid_symbols_with_info = fetch_all_symbols()
valid_symbols = valid_symbols_with_info[valid_symbols_with_info.iloc[:, 0].str.len() == 3].iloc[:, 0].tolist()

# ---------- Layout ----------
st.set_page_config(layout="wide")
st.title(VI_STRINGS["app_title"])

# placeholders for sections
header_ph    = st.empty()
chart_ph     = st.empty()
finance_income_ph = st.empty()
finance_ratio_ph  = st.empty()
dividend_ph  = st.empty()
analysis_ph  = st.empty()
info_ph      = st.empty()

# session state (no default symbol on first load)
if "current_symbol" not in st.session_state:
    st.session_state.current_symbol = None

with st.sidebar:
    st.header(VI_STRINGS["sidebar_header"])
    symbol_input = st.text_input(VI_STRINGS["enter_symbol"], value="").upper()
    ai_enabled = st.checkbox(VI_STRINGS["enable_ai_analysis"], value=False)
    apply_clicked = st.button(VI_STRINGS["apply_button"], type="primary")

def clear_sections():
    header_ph.empty(); chart_ph.empty()
    finance_income_ph.empty(); finance_ratio_ph.empty()
    dividend_ph.empty(); analysis_ph.empty(); info_ph.empty()

def render_dashboard(symbol: str, ai_enabled: bool = False):
    clear_sections()

    # Validate BEFORE any fetch
    if not symbol:
        with info_ph:
            st.info(VI_STRINGS["invalid_symbol_info"])
        return

    if symbol not in valid_symbols:
        with info_ph:
            st.info(VI_STRINGS["invalid_symbol_info"])
            st.error(VI_STRINGS["symbol_not_found"].format(symbol=symbol))
        return

    with st.spinner(VI_STRINGS["loading_spinner"]):
        df_real = load_recent_price(symbol)
        if df_real is None or df_real.empty:
            with info_ph:
                st.error(VI_STRINGS["no_recent_price"])
            return

        df_pred = load_prediction(symbol, 14)
        df_div  = load_dividend(symbol)
        df_income  = load_financials_income_statement(symbol)
        df_ratio = load_financials_ratio(symbol)

        if df_pred is None or df_pred.empty:
            with info_ph:
                st.warning(VI_STRINGS["no_recent_price"])
            return

        # ensure COL_TIME exists on prediction
        if COL_TIME not in df_pred.columns:
            last_dt = pd.to_datetime(df_real[COL_TIME].iloc[-1])
            forecast_dates = pd.date_range(
                start=last_dt + pd.offsets.BDay(1),
                periods=len(df_pred), freq=pd.offsets.BDay()
            )
            df_pred.insert(0, COL_TIME, forecast_dates)

        # ensure datetime
        df_real[COL_TIME] = pd.to_datetime(df_real[COL_TIME])
        df_pred[COL_TIME] = pd.to_datetime(df_pred[COL_TIME])

        exchange, organ_name = get_stock_info_by_symbol(symbol, valid_symbols_with_info)
        shares_outstanding, industry = fetch_company_overview(symbol)

        # Header
        with header_ph.container():
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader(f"🏢 {organ_name} ({exchange}: {symbol})")
            st.markdown(VI_STRINGS["industry"].format(industry=industry))
            st.markdown(VI_STRINGS["shares_outstanding"].format(shares_outstanding=shares_outstanding))

        # Chart
        with chart_ph.container():
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader(VI_STRINGS["price_forecast"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_real[COL_TIME], y=df_real[COL_CLOSE],
                mode='lines+markers', name=VI_STRINGS["actual_price"],
                marker=dict(symbol='circle', color='blue'),
                line=dict(color='blue'),
                hovertemplate=(
                    f'{VI_STRINGS["col_date"]}: %{{x|%Y-%m-%d}}'
                    f'<br>{VI_STRINGS["actual_price"]}: %{{y:.2f}}<extra></extra>'
                ),
                yaxis='y1'
            ))
            fig.add_trace(go.Scatter(
                x=df_pred[COL_TIME], y=df_pred[COL_CLOSE],
                mode='lines+markers', name=VI_STRINGS["predicted_price"],
                marker=dict(symbol='x', color='orange'),
                line=dict(color='orange', dash='dash'),
                hovertemplate=(
                    f'{VI_STRINGS["col_date"]}: %{{x|%Y-%m-%d}}'
                    f'<br>{VI_STRINGS["predicted_price"]}: %{{y:.2f}}<extra></extra>'
                ),
                yaxis='y1'
            ))
            fig.add_trace(go.Scatter(
                x=[df_real[COL_TIME].iloc[-1], df_pred[COL_TIME].iloc[0]],
                y=[df_real[COL_CLOSE].iloc[-1], df_pred[COL_CLOSE].iloc[0]],
                mode='lines', line=dict(color='orange', dash='dash'),
                hoverinfo='skip', showlegend=False, yaxis='y1'
            ))
            if COL_VOLUME in df_real.columns:
                fig.add_trace(go.Bar(
                    x=df_real[COL_TIME], y=df_real[COL_VOLUME],
                    name=VI_STRINGS["col_volume"],
                    marker_color='rgba(100, 100, 255, 0.3)',
                    yaxis='y2', opacity=0.5
                ))
            fig.update_layout(
                xaxis_title=VI_STRINGS["col_date"],
                yaxis=dict(title=VI_STRINGS["col_close_price"], side='left'),
                yaxis2=dict(title=VI_STRINGS["col_volume"], overlaying='y', side='right', showgrid=False),
                legend=dict(x=0, y=1),
                hovermode='x unified',
                template='plotly_white',
                margin=dict(t=10, b=40, l=40, r=10),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

        # Financials income
        with finance_income_ph.container():
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader(VI_STRINGS["financial_income"])
            st.markdown(df_income.style.hide(axis="index").to_html(), unsafe_allow_html=True)

        # Financial Ratios
        with finance_ratio_ph.container():
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader(VI_STRINGS["financial_ratio"])         
            st.markdown(df_ratio.style.hide(axis="index").to_html(), unsafe_allow_html=True)

        # Dividend
        with dividend_ph.container():
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader(VI_STRINGS["dividend_history"])
            st.markdown(df_div.style.hide(axis="index").to_html(), unsafe_allow_html=True)

        # AI Analysis (conditional)
        if ai_enabled:
            with analysis_ph.container():
                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.subheader(VI_STRINGS["ai_analysis"])
                final_prompt = PROMPT_TEMPLATE.format(
                    company_name=organ_name,
                    ticker=symbol,
                    industry=industry,
                    issue_share=shares_outstanding,
                    current_price=df_real[COL_CLOSE].iloc[-1],
                    industry_news={},
                    company_news={},
                    pe_industry_avg=10,
                    json_financial=df_div.to_json(orient="records", indent=2),
                    json_dividend=df_fin.to_json(orient="records", indent=2)
                )
                response = run_completion(final_prompt)
                st.write(response)

# -------- Single render path --------
# Only render after clicking Apply (and remember the choice)
if apply_clicked:
    st.session_state.current_symbol = symbol_input
    render_dashboard(symbol_input, ai_enabled=ai_enabled)
elif st.session_state.current_symbol:
    # If a symbol was applied previously in this session, show it
    render_dashboard(st.session_state.current_symbol, ai_enabled=ai_enabled)
else:
    # First load: no symbol yet — show guidance and no data fetches
    clear_sections()
    with info_ph:
        st.info(VI_STRINGS["invalid_symbol_info"])
