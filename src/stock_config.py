
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
Bạn là chuyên gia phân tích tài chính. Hãy thực hiện:

1. Phân tích mô hình kinh doanh & ngành hoạt động của công ty (trình bày ngắn gọn, không dùng thẻ HTML, chỉ dùng gạch đầu dòng hoặc xuống dòng).
2. Tìm tin tức & chính sách mới nhất ảnh hưởng đến công ty và ngành nghề kinh doanh.
3. Đánh giá kết quả kinh doanh 4 quý gần nhất từ dữ liệu báo cáo tài chính (doanh thu, LNST, YoY, ROE, EPS, P/E). Nêu rõ số liệu lấy từ quý nào.
4. Dự báo doanh thu & LNST 1–2 quý tới dựa trên dữ liệu tài chính và lịch sử cổ tức. Trình bày thêm kịch bản tích cực (best case) và tiêu cực (worst case).
5. Định giá cổ phiếu theo công thức: Giá hợp lý = (P/E trung bình ngành × EPS TTM).
6. Đưa ra khuyến nghị MUA / GIỮ / BÁN dựa trên so sánh giá hợp lý với giá thị trường hiện tại. Nếu có, đưa thêm mức giá mục tiêu và vùng stop-loss.

Thông tin công ty:
- Tên công ty: {company_name}
- Mã chứng khoán: {ticker}
- Ngành nghề: {industry}
- Số lượng cổ phiếu lưu hành: {issue_share}
- Giá thị trường hiện tại: {current_price}

Đầu vào:
- Báo cáo tài chính theo quý (HTML): {html_financial}
- Lịch sử cổ tức theo năm (HTML): {html_dividend}

Yêu cầu định dạng:
- Trình bày báo cáo dưới dạng văn bản thuần, không dùng thẻ HTML.
- Có bảng/tóm tắt số liệu tài chính (4 quý gần nhất).
- Kết thúc bằng Executive Summary ngắn gọn (tối đa 5 dòng).
"""