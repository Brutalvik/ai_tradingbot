import requests
import pandas as pd
import time
import os
import sys

def fetch_ohlcv(symbol, interval="1m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    
    os.makedirs("data", exist_ok=True)
    file_path = f"data/train_data_{symbol.lower()}.csv"
    df.to_csv(file_path)
    print(f"✅ Saved to {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide a trading pair symbol like BTCUSDT")
        sys.exit(1)

    symbol = sys.argv[1]
    fetch_ohlcv(symbol, interval="1m", limit=1000)
