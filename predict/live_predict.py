import json
import joblib
import websocket
import argparse
from utils.ohlcv_buffer import OHLCVBuffer
from utils.indicators import add_technical_indicators
from utils.label_data import clean_data_for_model
from utils.processing import create_features_and_labels
import requests
from datetime import datetime, timedelta

MODEL_PATH = "models/latest_model.pkl"
INTERVAL = 60  # 1-minute bars
CONFIDENCE_THRESHOLD = 0.6


def is_valid_symbol(symbol):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        response.raise_for_status()
        symbols = [s['symbol'].lower() for s in response.json()['symbols']]
        return symbol.lower() in symbols
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return False

from datetime import datetime, timedelta

def run_prediction(df, model):
    df = add_technical_indicators(df)
    df = clean_data_for_model(df)
    X, _ = create_features_and_labels(df)

    latest = X.tail(1)
    if latest.empty:
        print("⚠️ No valid feature row to predict.")
        return

    prediction = model.predict(latest)[0]
    confidence = model.predict_proba(latest)[0].max() if hasattr(model, "predict_proba") else 1.0

    current_price = df.iloc[-1]['close']
    raw_timestamp = df.iloc[-1]['timestamp']
    current_time = datetime.utcfromtimestamp(raw_timestamp / 1000)
    action_time = current_time + timedelta(minutes=1)

    if confidence >= CONFIDENCE_THRESHOLD:
        action = "🟢 BUY" if prediction == 1 else "🔴 SELL"

        print(f"""
🔔 LIMIT ORDER ALERT
----------------------------
🕒 Prediction Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
📅 Recommended Action Time: {action_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
📈 Suggested Action: {action}
💰 Suggested Limit Price: ${current_price:.2f}
🎯 Confidence Level: {confidence:.2%}
""")
    else:
        print(f"❔ Weak Signal ({confidence:.2%} < {CONFIDENCE_THRESHOLD:.0%}) — No action @ {current_time.strftime('%H:%M:%S')} UTC")


def main(symbol):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded.")

    if not is_valid_symbol(symbol):
        print(f"❌ Symbol '{symbol}' is not valid on Binance. Please try another.")
        return

    ohlcv_buffer = OHLCVBuffer(interval_seconds=INTERVAL)

    def on_message(ws, message):
        # Parse the incoming message
        try:
            message = message.decode('utf-8')
        except AttributeError:
            pass
        except json.JSONDecodeError:
            print("❗ Error decoding JSON message.")
            return
        except Exception as e:
            print(f"❗ Unexpected error: {e}")
            return
        if not message:
            print("❗ Empty message received.")
            return
        data = json.loads(message)
        price = float(data['p'])
        volume = float(data['q'])
        timestamp = int(data['T'])

        previous_length = len(ohlcv_buffer.bars)
        ohlcv_buffer.update(price, volume, timestamp)
        new_length = len(ohlcv_buffer.bars)

        if new_length > previous_length:
            print(f"🕒 New OHLCV bar formed. Total bars: {new_length}")
            df = ohlcv_buffer.get_dataframe()
            run_prediction(df, model)
        else:
            print(f"📩 Tick received: price={price} time={timestamp} - no new bar yet.")

    def on_open(ws):
        stream_url = f"{symbol}@trade"
        ws.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": [stream_url],
            "id": 1
        }))
        print(f"📡 Subscribed to {stream_url}")

    url = "wss://stream.binance.com:9443/ws"
    ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message)
    print("🚀 Starting live prediction stream...")
    ws.run_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live crypto prediction streamer.")
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol, e.g. btcusdt')
    args = parser.parse_args()
    main(args.symbol.lower())
# This script listens to live trade data from Binance, updates an OHLCV buffer,