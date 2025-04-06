import pandas as pd
import logging
from typing import List
from vnstock.explorer.vci import Company
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from underthesea import word_tokenize
from src.utility import convert_if_millis

# === GLOBAL PHOBERT MODEL ===
_phobert_tokenizer = None
_phobert_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load PhoBERT Model ===
def load_phobert_model():
    global _phobert_model, _phobert_tokenizer
    try:
        if _phobert_model is None or _phobert_tokenizer is None:
            local_model_path = "./models/phobert-finance"
            _phobert_tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)
            _phobert_model = AutoModelForSequenceClassification.from_pretrained(local_model_path).to(_device)
            _phobert_model.eval()
    except Exception as e:
        logging.warning(f"Failed to load PhoBERT locally: {e}")
        _phobert_model = None
        _phobert_tokenizer = None


def preprocess_vietnamese(text: str) -> str:
    return word_tokenize(text, format="text")


def analyze_sentiment_phobert(texts: List[str]) -> List[str]:
    """
    Analyze sentiment using PhoBERT model.
    """
    load_phobert_model()
    if _phobert_model is None or _phobert_tokenizer is None:
        return ["neutral"] * len(texts)

    sentiments = []
    labels_map = {0: "negative", 1: "neutral", 2: "positive"}

    for text in texts:
        try:
            preprocessed = preprocess_vietnamese(text)
            inputs = _phobert_tokenizer(preprocessed, return_tensors="pt", padding=True, truncation=True).to(_device)
            with torch.no_grad():
                logits = _phobert_model(**inputs).logits
                pred = torch.argmax(logits, dim=1).item()
                sentiments.append(labels_map.get(pred, "neutral"))
        except Exception as e:
            logging.error(f"PhoBERT inference failed on text: {text[:50]}... | Error: {e}")
            sentiments.append("neutral")

    return sentiments


# === Main Function ===
def fetch_news_with_sentiment(symbol: str) -> pd.DataFrame:
    """
    Fetch latest report, news, and event headlines for a stock symbol, and analyze sentiment using PhoBERT.

    Args:
        symbol (str): Stock symbol

    Returns:
        pd.DataFrame: DataFrame with headlines and predicted sentiment
    """
    company = Company(symbol)

    # Fetch top 5 items from each category
    news_df = company.news().sort_values(by='public_date', ascending=False).head(5)
    events_df = company.events().sort_values(by='public_date', ascending=False).head(5)
    reports_df = company.reports().sort_values(by='date', ascending=False).head(5)
    
    # Normalize to 'Date' + 'Headline'
    news_df = news_df[['public_date', 'news_short_content']].dropna().rename(columns={
        'public_date': 'Date',
        'news_short_content': 'Headline'
    })
    events_df = events_df[['public_date', 'event_title']].dropna().rename(columns={
        'public_date': 'Date',
        'event_title': 'Headline'
    })
    reports_df = reports_df[['date', 'name']].dropna().rename(columns={
        'date': 'Date',
        'name': 'Headline'
    })

    # Combine all sources
    combined_df = pd.concat([news_df, events_df, reports_df], ignore_index=True)

    # Convert Date column to string first
    combined_df['Date'] = combined_df['Date'].astype(str)
    combined_df['Date'] = combined_df['Date'].apply(convert_if_millis)

    # Drop rows with invalid dates
    combined_df = combined_df.dropna(subset=['Date'])

    # Sort and format
    combined_df = combined_df.sort_values(by='Date', ascending=False).reset_index(drop=True)
    combined_df['Date'] = combined_df['Date'].dt.strftime('%b %d, %Y')

    # Clean headlines
    combined_df = combined_df.dropna(subset=['Headline'])
    combined_df['Headline'] = combined_df['Headline'].astype(str)

    # Sentiment analysis
    try:
        combined_df['Sentiment'] = analyze_sentiment_phobert(combined_df['Headline'].tolist())
    except Exception as e:
        print(f"[PhoBERT ERROR] {e}")
        combined_df['Sentiment'] = ['neutral'] * len(combined_df)

    return combined_df[['Date', 'Headline', 'Sentiment']]


