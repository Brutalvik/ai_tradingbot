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
from models.custom_model import CombinedModel

MODEL_PATH = "models/latest_model.pkl"
REGRESSION_MODEL_PATH = "models/next_close_regressor.pkl"
BINANCE_API_URL = "https://api.binance.com/api/v3/exchangeInfo"

INTERVAL = 60  # 1-minute bars
CONFIDENCE_THRESHOLD = 0.6

action_model = joblib.load(MODEL_PATH)
next_close_model = joblib.load(REGRESSION_MODEL_PATH)

def is_valid_symbol(symbol):
    symbol = symbol.upper()

    if '/' in symbol:
        symbol = symbol.split('/')[0] + 'USDT'  # Convert to USDT pair if needed
    elif not symbol.endswith('USDT'):
        symbol += 'USDT'  # Default to USDT pair
    
    url = f"{BINANCE_API_URL}?symbol={symbol.upper()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        symbols = [s['symbol'].lower() for s in response.json()['symbols']]
        return symbol.lower() in symbols
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return False

def run_prediction(df, model):
    df = add_technical_indicators(df)
    df = clean_data_for_model(df)
    X, _ = create_features_and_labels(df)

    latest = X.tail(1)
    if latest.empty:
        print("⚠️ No valid feature row to predict.")
        return

    # Predict action (buy/sell) — classification
    action_prediction = action_model.predict(latest)[0]
    confidence = action_model.predict_proba(latest)[0].max() if hasattr(action_model, "predict_proba") else 1.0

    # Predict next candle close — regression
    try:
        next_close_price = next_close_model.predict(latest)[0]
    except Exception as e:
        print(f"⚠️ Regression model prediction failed. Using last close. Error: {e}")
        next_close_price = df.iloc[-1]['close']

    current_price = df.iloc[-1]['close']
    timestamp = df.iloc[-1]['timestamp']

    if confidence >= CONFIDENCE_THRESHOLD:
        action = "🟢 BUY" if action_prediction == 1 else "🔴 SELL"
        print(f"""
🔔 Trade Signal: {action}
📈 Current Price: ${current_price:.2f}
📅 Time: {timestamp}
📊 Confidence: {confidence*100:.2f}%
🔮 Predicted Next Close: ${next_close_price:.2f}
📌 Suggested Limit {'Buy' if action_prediction == 1 else 'Sell'} at ${next_close_price:.2f}
""")
    else:
        print(f"❔ Weak Signal: Confidence {confidence*100:.2f}% < {CONFIDENCE_THRESHOLD}. No action — {timestamp}")



def main(symbol):
    model = CombinedModel()
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