SOURCE_DATA = 'VCI'

COL_TICKER = 'ticker'
COL_EXCHANGE = 'exchange'
COL_ORGAN_NAME = 'organ_name'
COL_YEAR_REPORT = 'yearReport'
COL_QUARTER_REPORT = 'lengthReport'

# --- Price prediction related columns ---
COL_TIME = 'time'
COL_SYMBOL = 'symbol'
COL_OPEN = 'open'
COL_HIGH = 'high'
COL_LOW = 'low'
COL_CLOSE = 'close'
COL_VOLUME = 'volume'
COL_INDEX_OPEN = 'index_open'
COL_INDEX_HIGH = 'index_high'
COL_INDEX_LOW = 'index_low'
COL_INDEX_CLOSE = 'index_close'
COL_INDEX_VOLUME = 'index_volume'
COL_INDEX_PCT_CHANGE = 'index_pct_change'

# --- Intrinsic value related columns ---
COL_NET_PROFIT_MARGIN = 'Net Profit Margin (%)'
COL_ROE = 'ROE (%)'
COL_ROIC = 'ROIC (%)'
COL_ROA = 'ROA (%)'
COL_DEBT_EQUITY = 'Debt/Equity'
COL_LEVERAGE = 'Financial Leverage'
COL_DIVIDEND_YIELD = 'Dividend yield (%)'
COL_PE = 'P/E'
COL_PB = 'P/B'
COL_PS = 'P/S'
COL_PCASHFLOW = 'P/Cash Flow'
COL_EPS = 'EPS (VND)'
COL_BVPS = 'BVPS (VND)'
COL_FIXED_ASSET_TO_EQUITY = 'Fixed Asset-To-Equity'
COL_MARKET_CAP = 'Market Capital (Bn. VND)'
COL_EQUITY_TO_CHARTER = "Owners' Equity/Charter Capital"

INPUT_FEATURES = [
    COL_SYMBOL, COL_OPEN, COL_HIGH, COL_LOW, 
    COL_CLOSE, COL_VOLUME, COL_INDEX_OPEN,
    COL_INDEX_HIGH, COL_INDEX_LOW, COL_INDEX_CLOSE,
    COL_INDEX_VOLUME, COL_INDEX_PCT_CHANGE,
    COL_NET_PROFIT_MARGIN, COL_ROE, COL_ROIC, COL_ROA,
    COL_DEBT_EQUITY, COL_LEVERAGE, COL_DIVIDEND_YIELD,
    COL_PE, COL_PB, COL_PS, COL_PCASHFLOW,
    COL_EPS, COL_BVPS, COL_FIXED_ASSET_TO_EQUITY,
    COL_MARKET_CAP, COL_EQUITY_TO_CHARTER
]

TARGET_COLS = [COL_CLOSE, COL_EPS, COL_BVPS]
