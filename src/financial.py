import yfinance as yf
import pandas as pd
import os

EXECUTIVES = {
    "AAPL": {"name": "Tim Cook", "company": "Apple"},
    "MSFT": {"name": "Satya Nadella", "company": "Microsoft"},
    "NVDA": {"name": "Jensen Huang", "company": "Nvidia"},
    "TSLA": {"name": "Elon Musk", "company": "Tesla"},
    "META": {"name": "Mark Zuckerberg", "company": "Meta"},
    "AMZN": {"name": "Andy Jassy", "company": "Amazon"},
    "GOOGL": {"name": "Sundar Pichai", "company": "Alphabet"},
    "ADBE": {"name": "Shantanu Narayen", "company": "Adobe"},
    "CRM":  {"name": "Marc Benioff", "company": "Salesforce"},
    "INTC": {"name": "Pat Gelsinger", "company": "Intel"},
}

def download_stock_data(start="2018-01-01", end="2023-12-31", output_dir="data/stocks_2018_2023"):
    """Download daily stock data for all tickers and compute 5-day forward return."""
    os.makedirs(output_dir, exist_ok=True)

    for ticker, info in EXECUTIVES.items():
        print(f"Downloading {ticker} ({info['company']})...")
        
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Reset index so Date becomes a regular column
        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})
        
        # Keep only what we need
        df = df[["date", "Close", "Volume"]]
        df.columns = ["date", "close", "volume"]
        
        # Compute 5-day forward return (prediction target from paper)
        df["future_return"] = (df["close"].shift(-5) - df["close"]) / df["close"]
        df["label"] = (df["future_return"] > 0).astype(int)
        
        df["ticker"] = ticker
        df["executive"] = info["name"]
        df["company"] = info["company"]
        
        df.to_csv(f"{output_dir}/{ticker}.csv", index=False)
        print(f"  Saved {len(df)} rows for {ticker}")

if __name__ == "__main__":
    download_stock_data()