
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
Bạn là CHUYÊN GIA PHÂN TÍCH TÀI CHÍNH.  
Nhiệm vụ: Viết báo cáo ngắn gọn, có luận điểm rõ ràng, dựa trên dữ liệu JSON.  

QUY TẮC BẮT BUỘC:  
- Văn bản thuần, không HTML.  
- Mỗi mục ≤ 6 dòng, súc tích.  
- Không liệt kê số liệu từng quý; chỉ đưa số **tổng hợp** (4 quý, 2 quý + dự phóng).  
- TÍNH TOÁN chính xác từ JSON, KHÔNG tự bịa số liệu.  
- Các chỉ số quan trọng viết IN HOA hoặc **đậm**: DOANH THU, LNST, EPS TTM, EPS DỰ PHÓNG, GIÁ HỢP LÝ, KHUYẾN NGHỊ.  

THÔNG TIN CÔNG TY:  
- Tên: {company_name}  
- Mã: {ticker}  
- Ngành: {industry}  
- Số CP lưu hành: {issue_share}  
- Giá hiện tại: {current_price}  
- P/E ngành trung bình: {pe_industry_avg}  

ĐẦU VÀO:  
- Báo cáo lãi lỗ (JSON): {json_financial_income}  
- Lịch sử cổ tức (JSON): {json_dividend}  
- Tin tức ngành/công ty: {industry_news} {company_news}  

CẤU TRÚC BÁO CÁO:  

1) Mô hình kinh doanh & ngành  
- Nêu nguồn doanh thu chính, lợi thế cạnh tranh, rủi ro ngành.  

2) Tin tức & chính sách ngành  
- Tóm lược 2–3 tin tức/chính sách mới nhất.  
- Nếu thiếu tin tức: nêu cơ hội và rủi ro ngành nguyên lý.  

3) Kết quả kinh doanh 4 quý gần nhất  
- Đưa số tổng hợp: **DOANH THU**, **LNST**.  
- Nhận xét xu hướng: tăng/giảm/ổn định.  
- Nêu yếu tố chính: biên lợi nhuận, nợ vay, dòng tiền.  

4) Dự báo 2 quý tới  
- Giả định then chốt.  
- Trình bày 3 kịch bản (Base, Best, Worst).  
- Mỗi kịch bản: **DOANH THU dự phóng**, **LNST dự phóng** (2 số, làm tròn) + xu hướng.  

5) Định giá  
- Tính từ JSON:  
  + **LNST 4 quý gần nhất**  
  + **LNST 2 quý gần nhất + 2 quý dự báo**  
  + **EPS TTM**  
  + **EPS DỰ PHÓNG**  
  + **GIÁ HỢP LÝ (Base case)** = EPS DỰ PHÓNG × P/E ngành {pe_industry_avg}  
- So sánh với {current_price}.  

6) Khuyến nghị  
- Đưa ra **KHUYẾN NGHỊ: MUA / GIỮ / BÁN** dựa trên chênh lệch GIÁ HỢP LÝ vs {current_price}.  
- Nêu rõ giá mục tiêu (Base case).  
- Xác định vùng stop-loss = 8–12% dưới giá mua (giá hiện tại).  

7) Tổng kết  
- Tóm tắt 3 ý: xu hướng kinh doanh, định giá, hành động khuyến nghị.  
"""

VI_STRINGS = {
    "app_title": "📈 Ứng dụng Phân Tích Cổ phiếu",
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
    "financial_income": "📑 Báo cáo lãi lỗ",
    "financial_ratio": "📊 Chỉ số tài chính",
    "ai_analysis": "📊 Phân tích Cổ phiếu bằng AI",
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
}