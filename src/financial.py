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

def download_stock_data(start="2018-01-01", end="2023-12-31"):
    """Download daily OHLCV data for all tickers and compute 5-day forward return."""
    os.makedirs("data/stocks", exist_ok=True)
    
    for ticker, info in EXECUTIVES.items():
        print(f"Downloading {ticker} ({info['company']})...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        
        # Compute 5-day forward return (the prediction target from the paper)
        df["future_return"] = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
        df["label"] = (df["future_return"] > 0).astype(int)
        
        df["ticker"] = ticker
        df["executive"] = info["name"]
        df["company"] = info["company"]
        
        df.to_csv(f"data/stocks/{ticker}.csv")
        print(f"  Saved {len(df)} rows for {ticker}")

if __name__ == "__main__":
    download_stock_data()