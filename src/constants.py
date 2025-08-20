
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

PROMPT_TEMPLATE = """
Bạn là chuyên gia phân tích tài chính. Hãy thực hiện và tuân thủ nghiêm các ràng buộc:

Mục tiêu: Phân tích ngắn gọn, có luận điểm. Không lặp số liệu gốc từ JSON.

1) Mô hình kinh doanh & ngành:
- Trình bày súc tích (gạch đầu dòng hoặc xuống dòng), nêu nguồn doanh thu chính, lợi thế cạnh tranh, rủi ro đặc thù ngành.

2) Tin tức & chính sách ngành:
- Tóm lược xu hướng gần đây nếu có trong đầu vào; nếu thiếu dữ liệu mới, nêu 2–3 rủi ro và 2–3 cơ hội ngành (mang tính nguyên lý, không suy diễn vô dữ liệu).

3) KQKD 4 quý gần nhất (từ JSON):
- Phân tích xu hướng doanh thu, LNST, biên LN gộp/thuần, chi phí vận hành, đòn bẩy tài chính, dòng tiền nếu có.
- Không liệt kê lại số; chỉ kết luận theo hướng: tăng/giảm/ổn định, cải thiện/suy giảm biên, chất lượng lợi nhuận.

4) Dự báo 2 quý tới:
- Nêu giả định then chốt (tăng trưởng doanh thu, biên gộp/thuần, mùa vụ/one-off).
- Cho 3 kịch bản: Base, Best (+delta doanh thu/biên), Worst (−delta).
- Xuất kết quả dạng: Doanh thu & LNST (mô tả xu hướng + mức thay đổi tương đối, không ghi số tuyệt đối).

5) Định giá:
- EPS TTM = LNST 4 quý gần nhất / Số CP lưu hành.
- EPS TTM dự đoán = (LNST 4 quý gần nhất – LNST quý sớm nhất + LNST 2 quý dự báo) / Số CP lưu hành.
- Giá hợp lý = P/E trung bình ngành × EPS TTM dự đoán.

6) Khuyến nghị:
- MUA / GIỮ / BÁN dựa trên chênh lệch giữa Giá hợp lý và Giá hiện tại.
- Nêu giá mục tiêu (Base case) và vùng stop-loss (mặc định 8–12% dưới giá mua; điều chỉnh theo rủi ro ngành).

Ràng buộc trình bày:
- Văn bản thuần, không HTML.
- Mỗi mục ≤ 6 dòng.
- Không trích dẫn hoặc lặp số liệu gốc; chỉ nêu xu hướng/luận điểm.
- Kết thúc bằng "Tổng kết" (≤ 5 dòng) nêu 3 ý: xu hướng cốt lõi, định giá tương đối, hành động khuyến nghị.

Thông tin công ty:
- Tên công ty: {company_name}
- Mã chứng khoán: {ticker}
- Ngành nghề: {industry}
- Số lượng cổ phiếu lưu hành: {issue_share}
- Giá thị trường hiện tại: {current_price}

Đầu vào:
- Báo cáo tài chính theo quý (JSON): {json_financial}
- Lịch sử cổ tức theo năm (JSON): {json_dividend}
"""

VI_STRINGS = {
    "app_title": "📈 Ứng dụng Dự báo Cổ phiếu",
    "sidebar_header": "📊 Thiết lập Cổ phiếu",
    "enter_symbol": "Nhập mã cổ phiếu (ví dụ: ACB)",
    "apply_button": "Áp dụng",
    "invalid_symbol_info": "👈 Vui lòng chọn một mã hợp lệ gồm 3 ký tự và nhấn **Áp dụng**.",
    "symbol_not_found": "Mã '{symbol}' không tồn tại trong dữ liệu mô hình.",
    "loading_spinner": "⏳ Đang tải và phân tích dữ liệu cổ phiếu...",
    "industry": "**Ngành nghề:** {industry}",
    "shares_outstanding": "**Số cổ phiếu đang lưu hành:** {shares_outstanding:,}",
    "price_forecast": "📉 Dự báo giá cho 14 ngày tới",
    "dividend_history": "💸 Lịch sử Cổ tức",
    "financial_report": "📑 Báo cáo Tài chính Hàng quý",
    "ai_analysis": "📊 Phân tích Cổ phiếu bằng AI",
    "no_recent_price": "Không có dữ liệu giá gần đây.",
    "col_date": "Ngày",
    "col_close_price": "Giá đóng cửa",
    "col_volume": "Khối lượng",
    "actual_price": "Giá thực tế",
    "predicted_price": "Giá dự báo",
}