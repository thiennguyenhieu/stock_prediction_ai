import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json

# Best practice: page config at top
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

from src.fetch_historical_data import fetch_recent_price
from src.fetch_financial_data import fetch_income_statement, fetch_ratio, fetch_balance_sheet
from src.fetch_general_info import (
    fetch_all_symbols,
    fetch_company_overview,
    fetch_dividend,
    filter_by_valuation,
    filter_by_growth
)
from src.historical_inference_v1 import get_close_prediction
from src.stock_analysis import get_completion
from src.constants import *  # VI_STRINGS, COL_*, PROMPT_ANALYSIS, PROMPT_FILTER

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

@st.cache_data(show_spinner=False)
def load_financials_balance_sheet(symbol: str):
    return fetch_balance_sheet(symbol)

@st.cache_data(show_spinner=False)
def load_all_symbols():
    return fetch_all_symbols()

def run_completion(prompt: str):
    return get_completion(prompt)

# ---------- Helpers ----------
def _get_stock_info_by_symbol(symbol: str, df: pd.DataFrame) -> tuple[str | None, str | None]:
    row = df[df.iloc[:, 0] == symbol]
    if not row.empty:
        exchange = row.iloc[0, 1]
        organ_name = row.iloc[0, 2]
        return exchange, organ_name
    return None, None

def _to_html_no_index(df: pd.DataFrame) -> str:
    """Render DataFrame without index; fall back if .style.hide isn't available."""
    try:
        return df.style.hide(axis="index").to_html()
    except Exception:
        return df.to_html(index=False)

# ---- init logical AI flag once ----
if "ai_enabled" not in st.session_state:
    st.session_state.ai_enabled = False

# ---- AI state sync callbacks ----
def _sync_ai_from_filter():
    st.session_state.ai_enabled = st.session_state.get("ai_enabled_filter", False)

def _sync_ai_from_analysis():
    st.session_state.ai_enabled = st.session_state.get("ai_enabled_analysis", False)

# ---------- Static data ----------
valid_symbols_with_info = load_all_symbols()  # cached
valid_symbols = valid_symbols_with_info[
    valid_symbols_with_info.iloc[:, 0].str.len() == 3
].iloc[:, 0].tolist()

# ---------- Layout ----------
st.title(VI_STRINGS["app_title"])

# Global placeholders for analysis area
header_ph    = st.empty()
chart_ph     = st.empty()
finance_income_ph = st.empty()
finance_ratio_ph  = st.empty()
finance_balance_sheet_ph  = st.empty()
dividend_ph  = st.empty()
analysis_ph  = st.empty()
info_ph      = st.empty()

def _clear_analysis_sections():
    header_ph.empty()
    chart_ph.empty()
    finance_income_ph.empty()
    finance_ratio_ph.empty()
    finance_balance_sheet_ph.empty()
    dividend_ph.empty()
    analysis_ph.empty()
    info_ph.empty()

def _render_dashboard(symbol: str, ai_enabled: bool = False):
    _clear_analysis_sections()

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
        df_balance_sheet = load_financials_balance_sheet(symbol)

        if df_pred is None or df_pred.empty:
            with info_ph:
                st.warning(VI_STRINGS["no_prediction"])
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

        exchange, organ_name = _get_stock_info_by_symbol(symbol, valid_symbols_with_info)
        industry = fetch_company_overview(symbol)

        # Header
        with header_ph.container():
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader(f"🏢 {organ_name} ({exchange}: {symbol})")
            st.markdown(VI_STRINGS["industry"].format(industry=industry))

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
            # bridge line, no hover
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
                legend=dict(orientation="h", x=0, y=1.1),
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
            st.markdown(_to_html_no_index(df_income), unsafe_allow_html=True)

        # Financial Ratios
        with finance_ratio_ph.container():
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader(VI_STRINGS["financial_ratio"])
            st.markdown(_to_html_no_index(df_ratio), unsafe_allow_html=True)

        # Financial Balance sheet
        with finance_balance_sheet_ph.container():
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader(VI_STRINGS["financial_balance_sheet"])
            st.markdown(_to_html_no_index(df_balance_sheet), unsafe_allow_html=True)

        # Dividend
        with dividend_ph.container():
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader(VI_STRINGS["dividend_history"])
            st.markdown(_to_html_no_index(df_div), unsafe_allow_html=True)

        # AI Analysis (conditional)
        if ai_enabled:
            with analysis_ph.container():
                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.subheader(VI_STRINGS["ai_analysis"])
                final_prompt = PROMPT_ANALYSIS.format(
                    company_name=organ_name,
                    ticker=symbol,
                    industry=industry,
                    current_price=df_real[COL_CLOSE].iloc[-1] * 1000,
                    json_financial_income=df_income.to_json(orient="records", indent=2, force_ascii=False),
                    json_financial_ratio=df_ratio.to_json(orient="records", indent=2, force_ascii=False),
                    json_financial_balance_sheet=df_balance_sheet.to_json(orient="records", indent=2, force_ascii=False),
                    json_dividend=df_div.to_json(orient="records", indent=2, force_ascii=False)
                )
                response = run_completion(final_prompt)
                st.write(response)

def _render_results_once(
    df: pd.DataFrame,
    results_ph,
    ai_enabled_filter: bool = False,
    max_tickers_for_ai: int = 50
) -> None:
    """Render filter results into `results_ph`, replacing previous content."""
    results_ph.empty()

    if df is None or df.empty:
        results_ph.info(VI_STRINGS["no_result"])
        return

    # Table (hide index if supported)
    try:
        with results_ph.container():
            st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        html = df.to_html(index=False)
        with results_ph.container():
            st.markdown(html, unsafe_allow_html=True)

    # Optional AI summary
    if ai_enabled_filter:
        # Robust ticker extraction
        if "Mã cổ phiếu" in df.columns:
            tickers = df["Mã cổ phiếu"].dropna().astype(str).unique().tolist()
        elif "ticker" in df.columns:
            tickers = df["ticker"].dropna().astype(str).unique().tolist()
        else:
            tickers = df.iloc[:, 0].dropna().astype(str).unique().tolist()

        tickers = tickers[:max_tickers_for_ai]
        if not tickers:
            return

        try:
            with results_ph.container(), st.spinner(VI_STRINGS.get("ai_processing", "Đang phân tích bằng AI...")):
                json_tickers = json.dumps(tickers, ensure_ascii=False)
                prompt_filter = PROMPT_FILTER.format(tickers=json_tickers)
                response = run_completion(prompt_filter)
                st.write(response)
        except Exception as e:
            with results_ph.container():
                st.warning(VI_STRINGS.get("ai_error_generic", f"Lỗi khi chạy AI: {e}"))

# Tabs: Analysis + Filter (Analysis first => default)
tab_analysis, tab_filter = st.tabs([VI_STRINGS["tab_analysis"], VI_STRINGS["tab_filter"]])

# --------------------- FILTER TAB ---------------------
with tab_filter:
    st.subheader(VI_STRINGS["fitler_header"])  # keep your existing key

    # Controls always on top
    controls_box = st.container()
    with controls_box:
        c1, c2, c3 = st.columns([1, 1, 1])
        run_value  = c1.button(VI_STRINGS["filter_value"],  use_container_width=True)
        run_growth = c2.button(VI_STRINGS["filter_growth"], use_container_width=True)

        ai_enabled_filter = c3.checkbox(
            VI_STRINGS["enable_ai_analysis"],
            value=st.session_state.get("ai_enabled", False),
            key="ai_enabled_filter",
            on_change=_sync_ai_from_filter
        )

    # Results area BELOW controls
    results_ph = st.empty()

    if run_value:
        with st.spinner(VI_STRINGS["filtering_value_spinner"]):
            try:
                df_val = filter_by_valuation()
                _render_results_once(df_val, results_ph, ai_enabled_filter=st.session_state.get("ai_enabled", False))
            except Exception as e:
                results_ph.error(VI_STRINGS["filter_error_value"].format(e=e))

    if run_growth:
        with st.spinner(VI_STRINGS["filtering_growth_spinner"]):
            try:
                df_gro = filter_by_growth()
                _render_results_once(df_gro, results_ph, ai_enabled_filter=st.session_state.get("ai_enabled", False))  # fixed
            except Exception as e:
                results_ph.error(VI_STRINGS["filter_error_growth"].format(e=e))

# --------------------- ANALYSIS TAB ---------------------
with tab_analysis:
    st.subheader(VI_STRINGS["sidebar_header"])

    # Top controls container (stays in place)
    controls_box = st.container()
    with controls_box:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            symbol_input = st.text_input(
                VI_STRINGS["enter_symbol"],
                value=(st.session_state.get("current_symbol") or "")
            ).upper()
        with c2:
            ai_enabled = st.checkbox(
                VI_STRINGS["enable_ai_analysis"],
                value=st.session_state.get("ai_enabled", False),
                key="ai_enabled_analysis",
                on_change=_sync_ai_from_analysis
            )
        with c3:
            apply_clicked = st.button(
                VI_STRINGS["apply_button"],
                type="primary",
                use_container_width=True
            )

    # Keep session state in sync
    if apply_clicked:
        st.session_state.current_symbol = symbol_input
        st.query_params["symbol"] = symbol_input
        # ai_enabled already synced via callback

    # Render analysis (default tab)
    if st.session_state.get("current_symbol"):
        _render_dashboard(
            st.session_state["current_symbol"],
            ai_enabled=st.session_state.get("ai_enabled", False)
        )
    else:
        _clear_analysis_sections()
        st.info(VI_STRINGS["invalid_symbol_info"])
