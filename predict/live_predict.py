# predict/live_predict.py
import time
import joblib
from utils.fetch_data import fetch_binance_ohlcv
from utils.indicators import add_technical_indicators
from utils.label_data import clean_data_for_model

MODEL_PATH = "models/model.joblib"
SYMBOL = "BTC/USDT"
INTERVAL = "1m"

def live_predict():
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded.")

    while True:
        print("📈 Fetching new data...")
        df = fetch_binance_ohlcv(SYMBOL, INTERVAL, limit=100)
        df = add_technical_indicators(df)
        df = clean_data_for_model(df)

        latest = df.tail(1)
        prediction = model.predict(latest)[0]

        action = "🟢 BUY" if prediction == 1 else "🔴 SELL"
        print(f"\n🔔 Prediction: {action} at {latest.index[0]}\n")

        time.sleep(60)  # Wait 1 minute for the next prediction

if __name__ == "__main__":
    live_predict()
