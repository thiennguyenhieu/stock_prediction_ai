import pandas as pd

def fetch_news_with_sentiment(symbol: str) -> pd.DataFrame:
    """
    Fetch latest news headlines for a given stock symbol and predict sentiment.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Headline', 'Sentiment']
    """
    # For now, return fake data
    fake_news = [
        ("ACB announces strong Q1 earnings", "positive"),
        ("Market volatility impacts banking stocks", "negative"),
        ("Analysts upgrade ACB outlook to Buy", "positive"),
        ("Central bank policies tighten liquidity", "negative"),
        ("Dividend announcement pleases investors", "positive"),
        ("Banking sector sees cautious optimism", "positive"),
        ("Loan defaults increase slightly", "negative"),
        ("Stock price correction expected next quarter", "negative"),
        ("Digital banking rollout continues smoothly", "positive"),
        ("Foreign investors increase holdings", "positive")
    ]

    return pd.DataFrame(fake_news, columns=["Headline", "Sentiment"])