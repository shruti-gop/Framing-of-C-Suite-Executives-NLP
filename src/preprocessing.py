import re
import pandas as pd
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Clean a single text string for NLP input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def combine_text(row) -> str:
    """Merge title + description + content into one document."""
    parts = [row.get("title", ""), row.get("description", ""), row.get("content", "")]
    return " ".join(p for p in parts if isinstance(p, str) and p)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to the combined news dataframe."""
    df = df.copy()
    df["raw_text"] = df.apply(combine_text, axis=1)
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 50]
    df = df.drop_duplicates(subset="clean_text")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df["date"] = df["published_at"].dt.date
    return df.reset_index(drop=True)

def merge_all_sources() -> pd.DataFrame:
    """
    Merge all three data sources into one unified dataframe.
    Aligned to paper: combines news articles and social media text.
    """
    import os
    dfs = []
    
    files = {
        "data/raw/google_news_articles.csv": "google_news",
        "data/raw/gdelt_articles.csv": "gdelt",
        "data/raw/newsapi_articles.csv": "newsapi",
    }
    
    for filepath, source in files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} rows from {source}")
            dfs.append(df)
        else:
            print(f"Skipping {source} - file not found")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} total articles")
    return combined

if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)
    
    raw = merge_all_sources()
    clean = preprocess_dataframe(raw)
    clean.to_csv("data/processed/articles_clean.csv", index=False)
    print(f"After cleaning: {len(clean)} articles saved to data/processed/articles_clean.csv")