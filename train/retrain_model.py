from utils.fetch_data import fetch_binance_ohlcv
import time

SYMBOL = "DOGE/USDT"  # Change to your symbol
INTERVAL = "1m"       # 1-minute candles
LIMIT = 2000          # Number of data points (candles) to fetch

seven_days_ago = int((time.time() - 7*24*60*60) * 1000)


df = fetch_binance_ohlcv(SYMBOL, INTERVAL, since=seven_days_ago)

print(f"Fetched {len(df)} rows of data for {SYMBOL}")
print(df.head())