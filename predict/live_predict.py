import json
import joblib
import websocket
import argparse
import signal
import threading
import requests
import itertools
import time
import sys
from utils.ohlcv_buffer import OHLCVBuffer
from utils.indicators import add_technical_indicators
from utils.label_data import clean_data_for_model
from utils.processing import create_features_and_labels

MODEL_PATH = "models/BTCUSDT_1m_latest_model.pkl"
REGRESSION_MODEL_PATH = "models/next_close_regressor.pkl"
BINANCE_API_URL = "https://api.binance.com/api/v3/exchangeInfo"

INTERVAL = 60  # 1-minute bars
CONFIDENCE_THRESHOLD = 0.6

# Load models
action_model = joblib.load(MODEL_PATH)
next_close_model = joblib.load(REGRESSION_MODEL_PATH)

# Graceful shutdown handling
shutdown_event = threading.Event()

def start_spinner_with_timer(stop_event):
    spinner_cycle = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r⏳ Waiting for first OHLCV bar... {elapsed:.1f}s {next(spinner_cycle)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r✅ First OHLCV bar formed.                \n")

def is_valid_symbol(symbol):
    symbol = symbol.upper().replace("/", "")
    url = f"{BINANCE_API_URL}?symbol={symbol}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return True
    except Exception:
        return False

def run_prediction(df):
    df = add_technical_indicators(df)
    df = clean_data_for_model(df)
    X, _ = create_features_and_labels(df)

    if X.empty:
        print("⚠️ No valid features to predict.")
        return

    latest = X.tail(1)
    action = action_model.predict(latest)[0]
    confidence = action_model.predict_proba(latest)[0].max()

    try:
        next_close = next_close_model.predict(latest)[0]
    except Exception as e:
        print(f"⚠️ Regression model failed: {e}")
        next_close = df.iloc[-1]['close']

    current_price = df.iloc[-1]['close']
    timestamp = df.iloc[-1]['timestamp']

    if confidence >= CONFIDENCE_THRESHOLD:
        signal = "🟢 BUY" if action == 1 else "🔴 SELL"
        print(f"""
🔔 Trade Signal: {signal}
📈 Current Price: ${current_price:.2f}
📅 Time: {timestamp}
📊 Confidence: {confidence * 100:.2f}%
🔮 Predicted Next Close: ${next_close:.2f}
📌 Suggested Limit {'Buy' if action == 1 else 'Sell'} at ${next_close:.2f}
""")
    else:
        print(f"❔ Weak Signal: {confidence * 100:.2f}% < {CONFIDENCE_THRESHOLD} — {timestamp}")

def main(symbol):
    if not is_valid_symbol(symbol):
        print(f"❌ Invalid symbol '{symbol.upper()}'.")
        return

    ohlcv_buffer = OHLCVBuffer(interval_seconds=INTERVAL)
    first_bar_formed = threading.Event()
    spinner_stop_event = threading.Event()

    spinner_thread = threading.Thread(target=start_spinner_with_timer, args=(spinner_stop_event,))
    spinner_thread.start()

    def on_message(ws, message):
        try:
            data = json.loads(message)

            if data.get("e") != "trade":
                return

            price_str = data.get("p") or data.get("price")
            volume_str = data.get("q")
            timestamp = data.get("T")

            if not price_str or not volume_str or not timestamp:
                print(f"⚠️ Skipping message with missing fields: {data}")
                return

            price = float(price_str)
            volume = float(volume_str)
            timestamp = int(timestamp)

            prev_len = len(ohlcv_buffer.bars)
            ohlcv_buffer.update(price, volume, timestamp)
            new_len = len(ohlcv_buffer.bars)

            if new_len > prev_len:
                if not first_bar_formed.is_set():
                    spinner_stop_event.set()
                    first_bar_formed.set()
                print(f"🕒 New OHLCV bar formed. Total bars: {new_len}")
                df = ohlcv_buffer.get_dataframe()
                run_prediction(df)

        except Exception as e:
            print(f"❗ Error in message handling: {e}")

    def on_open(ws):
        ws.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@trade"],
            "id": 1
        }))
        print(f"\n📡 Subscribed to live trade stream for {symbol.upper()}")

    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")

    def on_close(ws, code, reason):
        print(f"\n🔌 WebSocket closed. Code: {code}, Reason: {reason}")

    def signal_handler(sig, frame):
        print("\n👋 Caught Ctrl+C. Shutting down gracefully...")
        shutdown_event.set()
        spinner_stop_event.set()
        ws.close()

    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    print("🚀 Starting live stream...")
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{symbol}@trade",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.start()

    while not shutdown_event.is_set():
        time.sleep(0.1)

    ws.close()
    ws_thread.join()
    print("🛑 Exited cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Crypto Predictor")
    parser.add_argument("--symbol", required=True, help="Symbol like BTCUSDT")
    args = parser.parse_args()
    main(args.symbol.lower())
