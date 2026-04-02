from gdeltdoc import GdeltDoc, Filters
import pandas as pd
import os
import time
from src.financial import EXECUTIVES

YEARS = [
    ("2018-01-01", "2018-12-31"),
    ("2019-01-01", "2019-12-31"),
    ("2020-01-01", "2020-12-31"),
    ("2021-01-01", "2021-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
]

def collect_gdelt_historical():
    """
    Pull GDELT articles year by year for each executive.
    Aligned to paper: full 2018-2023 coverage.
    """
    os.makedirs("data/raw", exist_ok=True)
    all_articles = []
    gd = GdeltDoc()

    for ticker, info in EXECUTIVES.items():
        exec_name = info["name"]
        company = info["company"]
        print(f"\nFetching {exec_name}...")

        for start, end in YEARS:
            year = start[:4]
            print(f"  Year {year}...")

            for attempt in range(3):
                try:
                    f = Filters(
                        keyword=f"{exec_name} {company}",
                        start_date=start,
                        end_date=end
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
                        print(f"    Found {len(articles)} articles")
                    else:
                        print(f"    No articles found")

                    time.sleep(6)
                    break

                except Exception as e:
                    wait = 20 * (attempt + 1)
                    print(f"    Attempt {attempt+1} failed, waiting {wait}s...")
                    time.sleep(wait)

        # Save after each executive so we don't lose progress
        df = pd.DataFrame(all_articles)
        df.to_csv("data/raw/gdelt_articles_2018_2023.csv", index=False)
        print(f"  Progress saved: {len(df)} total articles so far")

    df = pd.DataFrame(all_articles)
    df.to_csv("data/raw/gdelt_articles_2018_2023.csv", index=False)
    print(f"\nDone. Total: {len(df)} articles")
    return df

if __name__ == "__main__":
    collect_gdelt_historical()