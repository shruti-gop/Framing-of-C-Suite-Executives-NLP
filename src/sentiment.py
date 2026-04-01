import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

MODEL_NAME = "ProsusAI/finbert"
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
SCORE_MAP = {"positive": 1, "neutral": 0, "negative": -1}

class FinBERTSentiment:
    def __init__(self):
        print("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        print(f"  Using device: {self.device}")

    def predict(self, text: str) -> dict:
        """Return sentiment label, score, and confidence for one text."""
        inputs = self.tokenizer(
            text[:512],
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()[0]

        probs = softmax(logits)
        label_id = int(np.argmax(probs))
        label = LABEL_MAP[label_id]

        return {
            "sentiment_label": label,
            "sentiment_score": SCORE_MAP[label],
            "confidence": float(probs[label_id]),
            "prob_positive": float(probs[0]),
            "prob_negative": float(probs[1]),
            "prob_neutral": float(probs[2]),
        }

    def predict_batch(self, df: pd.DataFrame, text_col="clean_text") -> pd.DataFrame:
        """Run sentiment on full dataframe with progress bar."""
        results = []
        for text in tqdm(df[text_col], desc="Running FinBERT"):
            results.append(self.predict(text))
        return df.join(pd.DataFrame(results))

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level scores to daily per-executive signals.
    Aligned to paper: rolling sentiment averages, volatility, attention volume.
    """
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby(["ticker", "date"])
        .agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            article_count=("sentiment_score", "count"),
            prob_positive_mean=("prob_positive", "mean"),
            prob_negative_mean=("prob_negative", "mean"),
        )
        .reset_index()
    )

    # Rolling 7-day and 14-day averages per ticker
    # Aligned to paper: aggregated sentiment features
    daily = daily.sort_values(["ticker", "date"])
    for window in [7, 14]:
        daily[f"sentiment_rolling_{window}d"] = (
            daily.groupby("ticker")["sentiment_mean"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    # Sentiment volatility rolling window
    daily["sentiment_volatility"] = (
        daily.groupby("ticker")["sentiment_std"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    return daily

if __name__ == "__main__":
    df = pd.read_csv("data/processed/articles_clean.csv")
    model = FinBERTSentiment()
    df_scored = model.predict_batch(df)
    df_scored.to_csv("data/processed/articles_sentiment.csv", index=False)

    daily = aggregate_daily_sentiment(df_scored)
    daily.to_csv("data/processed/daily_sentiment.csv", index=False)
    print(f"Done. Scored {len(df_scored)} articles.")
    print(f"Daily sentiment: {len(daily)} rows saved.")