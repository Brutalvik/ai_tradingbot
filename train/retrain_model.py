import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from utils.fetch_data import fetch_binance_ohlcv
from utils.processing import create_features_and_labels

SYMBOL = "DOGE/USDT"  # Change to your symbol
INTERVAL = "1m"       # 1-minute candles
LIMIT = 2000          # Number of data points (candles) to fetch

seven_days_ago = int((time.time() - 7*24*60*60) * 1000)


df = fetch_binance_ohlcv(SYMBOL, INTERVAL, since=seven_days_ago)

X, y = create_features_and_labels(df)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print(f"Fetched {len(df)} rows of data for {SYMBOL}")
print(df.head())

joblib.dump(model, 'models/latest_model.pkl')
print("Model saved to models/latest_model.pkl")