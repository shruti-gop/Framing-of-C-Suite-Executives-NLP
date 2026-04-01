from gdeltdoc import GdeltDoc, Filters
import pandas as pd
import os
import time
from src.financial import EXECUTIVES

def collect_gdelt():
    """
    Pull news articles from GDELT for each executive.
    Free, no API key needed. Full historical coverage back to 2018.
    Aligned to paper: 2018-2023 date range.
    """
    os.makedirs("data/raw", exist_ok=True)
    all_articles = []
    gd = GdeltDoc()

    for ticker, info in EXECUTIVES.items():
        exec_name = info["name"]
        company = info["company"]
        print(f"Fetching GDELT articles for {exec_name}...")

        try:
            f = Filters(
                keyword=f"{exec_name} {company}",
                start_date="2018-01-01",
                end_date="2023-12-31"
            )

            articles = gd.article_search(f)

            if articles is not None and len(articles) > 0:
                for _, row in articles.iterrows():
                    all_articles.append({
                        "ticker": ticker,
                        "executive": exec_name,
                        "company": company,
                        "published_at": row.get("seendate", ""),
                        "title": row.get("title", ""),
                        "description": row.get("title", ""),
                        "content": row.get("title", ""),
                        "source": row.get("domain", ""),
                        "url": row.get("url", ""),
                        "data_source": "gdelt"
                    })
                print(f"  Found {len(articles)} articles")
            else:
                print(f"  No articles found")

            time.sleep(2)

        except Exception as e:
            print(f"  Error for {exec_name}: {e}")

    df = pd.DataFrame(all_articles)
    df.to_csv("data/raw/gdelt_articles.csv", index=False)
    print(f"\nGDELT: Saved {len(df)} articles total.")
    return df

if __name__ == "__main__":
    collect_gdelt()