# predict/live_predict.py
import time
import joblib
from utils.fetch_data import fetch_binance_ohlcv
from utils.indicators import add_technical_indicators
from utils.label_data import clean_data_for_model

MODEL_PATH = "models/model.joblib"
SYMBOL = "BTC/USDT"
INTERVAL = "1m"
CONFIDENCE_THRESHOLD = 0.6  # Optional: ignore weak signals

def live_predict():
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded.")

    while True:
        print("📈 Fetching new data...")
        df = fetch_binance_ohlcv(SYMBOL, INTERVAL, limit=100)
        if df.empty:
            print("⚠️ No data fetched. Retrying in 60 seconds...")
            time.sleep(60)
            continue

        df = add_technical_indicators(df)
        print("🔧 Technical indicators added.")

        df = clean_data_for_model(df)
        print("🔧 Data cleaned and prepared for prediction.")

        latest = df.tail(1)
        if latest.empty:
            print("⚠️ No data available for prediction. Retrying...")
            time.sleep(60)
            continue

        latest = clean_data_for_model(latest)
        print("🔍 Latest data prepared for prediction.")

        latest_numeric = latest.select_dtypes(include=['number']).drop(columns=['open', 'high', 'low', 'close'], errors='ignore')
        if latest_numeric.empty:
            print("⚠️ No numeric data available for prediction. Retrying...")
            time.sleep(60)
            continue

        print("🔍 Numeric data extracted for prediction.")

        # Make prediction
        prediction = model.predict(latest_numeric)[0]

        # Get confidence score if available
        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(latest_numeric)[0])
        else:
            confidence = 1.0  # fallback

        # Get current price from original df
        current_price = df.iloc[-1]['close']
        timestamp = df.index[-1] if df.index.name else time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"🔮 Prediction made: {prediction} (Confidence: {confidence:.2f})")

        # Determine and print the trade action
        if confidence >= CONFIDENCE_THRESHOLD:
            action = "🟢 BUY" if prediction == 1 else "🔴 SELL"
            print(f"\n🔔 Trade Signal: {action} at ${current_price:.2f} (Confidence: {confidence:.2f}) — {timestamp}\n")

            # Optional: insert real trading logic here (e.g., broker API call)
            # place_trade(action, current_price)

        else:
            print(f"❔ Signal too weak. Confidence: {confidence:.2f} < Threshold: {CONFIDENCE_THRESHOLD}. No trade.\n")

        time.sleep(60)

if __name__ == "__main__":
    live_predict()
