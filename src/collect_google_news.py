import feedparser
import pandas as pd
import time
import os
from datetime import datetime
from src.financial import EXECUTIVES

def collect_google_news():
    """
    Pull articles from Google News RSS feeds for each executive.
    Free, no API key needed. Covers broader media and blogs.
    Aligned to paper: broader media coverage source.
    """
    os.makedirs("data/raw", exist_ok=True)
    all_articles = []

    for ticker, info in EXECUTIVES.items():
        exec_name = info["name"]
        company = info["company"]
        print(f"Fetching Google News for {exec_name}...")

        # Build Google News RSS URL
        query = f"{exec_name} {company}".replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        try:
            feed = feedparser.parse(url)
            count = 0

            for entry in feed.entries:
                all_articles.append({
                    "ticker": ticker,
                    "executive": exec_name,
                    "company": company,
                    "published_at": entry.get("published", ""),
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "content": entry.get("summary", ""),
                    "source": entry.get("source", {}).get("title", "Google News"),
                    "url": entry.get("link", ""),
                    "data_source": "google_news"
                })
                count += 1

            print(f"  Found {count} articles")
            time.sleep(1)

        except Exception as e:
            print(f"  Error for {exec_name}: {e}")

    df = pd.DataFrame(all_articles)
    df.to_csv("data/raw/google_news_articles.csv", index=False)
    print(f"\nGoogle News: Saved {len(df)} articles total.")
    return df

if __name__ == "__main__":
    collect_google_news()