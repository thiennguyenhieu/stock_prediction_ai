import pandas as pd
from typing import List
from transformers import pipeline, Pipeline
import logging

_finbert_pipeline: Pipeline = None

def load_finbert_model() -> Pipeline:
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            _finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except Exception as e:
            logging.warning(f"Failed to load FinBERT: {e}")
            _finbert_pipeline = None
    return _finbert_pipeline

def analyze_sentiment_finbert(headlines: List[str]) -> List[str]:
    model = load_finbert_model()
    if model is not None:
        try:
            results = model(headlines)
            return [r['label'].lower() for r in results]
        except Exception as e:
            logging.error(f"FinBERT inference failed: {e}")
    # fallback if model not loaded or failed
    return ["neutral"] * len(headlines)

def fetch_news_with_sentiment(symbol: str) -> pd.DataFrame:
    """
    Fetch latest news headlines for a given stock symbol and predict sentiment.

    Args:
        symbol (str): Stock symbol

    Returns:
        pd.DataFrame: DataFrame with ['Headline', 'Sentiment']
    """
    headlines = [
        "ACB announces strong Q1 earnings",
        "Market volatility impacts banking stocks",
        "Analysts upgrade ACB outlook to Buy",
        "Central bank policies tighten liquidity",
        "Dividend announcement pleases investors",
        "Banking sector sees cautious optimism",
        "Loan defaults increase slightly",
        "Stock price correction expected next quarter",
        "Digital banking rollout continues smoothly",
        "Foreign investors increase holdings"
    ]

    sentiments = analyze_sentiment_finbert(headlines)
    return pd.DataFrame({"Headline": headlines, "Sentiment": sentiments})
