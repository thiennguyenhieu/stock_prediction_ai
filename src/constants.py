
SOURCE_DATA = 'VCI'

COL_EXCHANGE = 'exchange'
COL_ORGAN_NAME = 'organ_name'
COL_ISSUE_SHARE = 'issue_share'
COL_ICB_NAME = 'icb_name2'
COL_TIME = 'time'
COL_QUARTER = 'quarter'
COL_YEAR = 'year'
COL_TIME_ORDINAL = 'time_ordinal'
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
COL_NET_PROFIT_MARGIN = 'Net Profit Margin (%)'
COL_ROE = 'ROE (%)'
COL_ROA = 'ROA (%)'
COL_DEBT_EQUITY = 'Debt/Equity'
COL_LEVERAGE = 'Financial Leverage'
COL_DIVIDEND_YIELD = 'Dividend yield (%)'
COL_PE = 'P/E'
COL_PB = 'P/B'
COL_PS = 'P/S'
COL_PCASHFLOW = 'P/Cash Flow'
COL_EPS = 'EPS (VND)'
COL_EPS_TTM = 'EPS TTM (VND)'
COL_BVPS = 'BVPS (VND)'
COL_FIXED_ASSET_TO_EQUITY = 'Fixed Asset-To-Equity'
COL_MARKET_CAP = 'Market Capital (Bn. VND)'
COL_EQUITY_TO_CHARTER = "Owners' Equity/Charter Capital"
COL_MA_10 = 'MA_10'
COL_MA_30 = 'MA_30'
COL_MOMENTUM_10 = 'Momentum_10'
COL_VOLATILITY_20 = 'Volatility_20d'
COL_RETURN_1D = 'Return_1d'
COL_RETURN_5D = 'Return_5d'
COL_RSI_14 = 'RSI_14'
COL_EMA_10 = 'ema_10'
COL_EMA_30 = 'ema_30'
COL_BB_UPPER = 'bb_upper'
COL_BB_LOWER = 'bb_lower'
COL_MACD = 'macd'
COL_MACD_SIGNAL = 'macd_signal'
COL_REVENUE = 'Revenue (Bn. VND)'
COL_REVENUE_YOY = 'Revenue YoY (%)'
COL_ATTRIBUTE = 'Attribute to parent company (Bn. VND)'
COL_ATTRIBUTE_YOY = 'Attribute to parent company YoY (%)'

ENCODER_PATH = "data/symbol_encoder.pkl"

PROMPT_ANALYSIS = """
Bạn là chuyên gia phân tích chứng khoán. 
Dựa trên dữ liệu đầu vào (Tên, Mã, Ngành, Giá hiện tại, Income, Ratios, Balance sheet, Cổ tức), hãy phân tích ngắn gọn cho nhà đầu tư theo cấu trúc sau:

I. Giới thiệu công ty  
- Tên, mã, ngành, hoạt động chính.

II. Bối cảnh ngành & công ty  
- Tin tức/diễn biến ngành và công ty ảnh hưởng tới doanh thu, lợi nhuận.

III. KQKD gần nhất  
- Doanh thu, LNST, EPS, ROE/ROA.  
- Phân tích vốn chủ, nợ, đòn bẩy tài chính.

IV. Dự phóng (1 quý & 2 quý tới)  
- LNST & EPS dự kiến.  
- Lý do dự phóng (xu hướng ngành, biên LN, chi phí, nợ...).

V. Định giá  
- Giá hợp lý = EPS dự phóng × P/E TB ngành (nếu thiếu lấy 10).  
- So sánh với giá hiện tại.  

VI. Khuyến nghị  
- MUA / GIỮ / BÁN, kèm 1 câu lý do ngắn gọn.

⚠️ Yêu cầu: Trình bày ngắn gọn, rõ ràng, dễ đọc (bullet points, không viết lan man).

ĐẦU VÀO
- Tên: {company_name}  
- Mã: {ticker}  
- Ngành: {industry}  
- Giá hiện tại: {current_price}   
- Income: {json_financial_income}  
- Ratios: {json_financial_ratio}  
- Balance sheet: {json_financial_balance_sheet}  
- Cổ tức: {json_dividend}
"""

PROMPT_FILTER = """
Bạn là chuyên gia phân tích chứng khoán.
Đầu vào: danh sách các mã cổ phiếu {tickers}.

Nhiệm vụ: với mỗi mã, viết nhận xét ngắn gọn (1–2 câu).
+ ⚠️ Cảnh báo nếu lợi nhuận chủ yếu đến từ khoản một lần (thanh lý tài sản, hoàn nhập dự phòng, lãi tài chính bất thường…) → “không bền vững”.
+ ✅ Đánh giá cao nếu lợi nhuận đến từ hoạt động cốt lõi, ổn định và bền vững.

Yêu cầu: Xuất kết quả dưới dạng bảng Markdown với 2 cột:
+ Mã cổ phiếu
+ Nhận xét (có icon ✅ hoặc ⚠️ ở đầu để phân loại)
+ Danh sách mã tích cực: chỉ liệt kê các ticker được đánh giá tích cực/khả quan.
"""

VI_STRINGS = {
    "app_title": "📈 Ứng dụng Phân Tích Cổ phiếu",
    "sidebar_header": "🗂️ Thiết lập Cổ phiếu",
    "enter_symbol": "Nhập mã cổ phiếu (ví dụ: ACB)",
    "apply_button": "Áp dụng",
    "invalid_symbol_info": "Vui lòng chọn một mã hợp lệ gồm 3 ký tự và nhấn **Áp dụng**.",
    "symbol_not_found": "Mã '{symbol}' không tồn tại trong dữ liệu mô hình.",
    "loading_spinner": "⏳ Đang tải và phân tích dữ liệu cổ phiếu...",
    "industry": "**Ngành nghề:** {industry}",
    "shares_outstanding": "**Số cổ phiếu đang lưu hành:** {shares_outstanding:,}",
    "price_forecast": "🔮 Dự báo giá cho 14 ngày tới (chỉ mang tính chất tham khảo)",
    "dividend_history": "💸 Lịch sử Cổ tức",
    "financial_income": "💹 Báo cáo lãi lỗ",
    "financial_ratio": "📊 Chỉ số tài chính",
    "financial_balance_sheet": "📒 Bảng cân đối kế toán",
    "ai_analysis": "🤖 Phân tích Cổ phiếu bằng AI",
    "no_recent_price": "Không có dữ liệu giá gần đây.",
    "col_date": "Ngày",
    "col_close_price": "Giá đóng cửa",
    "col_volume": "Khối lượng",
    "actual_price": "Giá thực tế",
    "predicted_price": "Giá dự báo",
    "exercise_date": "Ngày thực hiện",
    "cash_year": "Năm chi trả",
    "cash_dividend_percentage": "Tỷ lệ cổ tức tiền mặt (%)",
    "issue_method": "Phương thức phát hành",
    "enable_ai_analysis": "Bật phân tích AI",
    "cash": "Tiền mặt",
    "share": "Cổ phiếu",
    "no_prediction": "Không có dữ liệu dự báo.",
    "tab_analysis": "📊 Phân tích",
    "tab_filter": "🧰 Bộ lọc",
    "fitler_header": "Bộ lọc cổ phiếu",
    "filter_value": "Lọc theo giá trị",
    "filter_growth": "Lọc theo tăng trưởng",
    "filtering_value_spinner": "⏳ Đang lọc theo giá trị...",
    "filtering_growth_spinner": "⏳ Đang lọc theo tăng trưởng...",
    "filter_error_value": "Lỗi khi lọc theo giá trị: {e}",
    "filter_error_growth": "Lỗi khi lọc theo tăng trưởng: {e}",
    "no_result": "Không có mã nào thỏa điều kiện.",
    "ai_processing": "Đang phân tích bằng AI...",
    "ai_error_generic": "Lỗi khi chạy AI: {e}",
}