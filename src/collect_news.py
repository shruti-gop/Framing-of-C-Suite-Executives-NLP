import os
import time
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv
from src.financial import EXECUTIVES

load_dotenv()

api = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

def collect_newsapi(start="2026-03-01", end="2026-03-30"):
    """
    Pull news articles mentioning each executive from NewsAPI.
    Free tier only goes back 30 days.
    """
    os.makedirs("data/raw", exist_ok=True)
    all_articles = []

    for ticker, info in EXECUTIVES.items():
        exec_name = info["name"]
        company = info["company"]
        print(f"Fetching NewsAPI articles for {exec_name}...")

        try:
            response = api.get_everything(
                q=f'"{exec_name}" AND "{company}"',
                language="en",
                from_param=start,
                to=end,
                sort_by="publishedAt",
                page_size=100
            )

            for article in response.get("articles", []):
                all_articles.append({
                    "ticker": ticker,
                    "executive": exec_name,
                    "company": company,
                    "published_at": article["publishedAt"],
                    "title": article["title"],
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "source": article["source"]["name"],
                    "url": article["url"],
                    "data_source": "newsapi"
                })

            print(f"  Found {len(response.get('articles', []))} articles")
            time.sleep(1)

        except Exception as e:
            print(f"  Error for {exec_name}: {e}")

    df = pd.DataFrame(all_articles)
    df.to_csv("data/raw/newsapi_articles.csv", index=False)
    print(f"\nNewsAPI: Saved {len(df)} articles total.")
    return df

if __name__ == "__main__":
    collect_newsapi()