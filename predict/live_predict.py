import json
import joblib
import websocket
import argparse
from datetime import datetime
from utils.ohlcv_buffer import OHLCVBuffer
from utils.indicators import add_technical_indicators
from utils.label_data import clean_data_for_model
from utils.processing import create_features_and_labels
import requests
import itertools
import threading
import time
import sys
import signal

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import print

console = Console()

MODEL_PATH = "models/BTCUSDT_1m_latest_model.pkl"
REGRESSION_MODEL_PATH = "models/next_close_regressor.pkl"
BINANCE_API_URL = "https://api.binance.com/api/v3/exchangeInfo"

INTERVAL = 60  # 1-minute bars
CONFIDENCE_THRESHOLD = 0.6
stop_event = threading.Event()


def graceful_exit(signum, frame):
    stop_event.set()
    console.print("\n👋 Terminating Program. Goodbye!", style="bold red")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)


def is_valid_symbol(symbol):
    console.print(f"\n🔍 Validating symbol: [cyan]{symbol.upper()}[/cyan]")
    symbol = symbol.upper().replace("/", "")
    url = f"{BINANCE_API_URL}?symbol={symbol}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return True
    except Exception:
        return False


def run_prediction(df):
    if len(df) < 10:
        console.print(f"⏳ Waiting for more data... only {len(df)} bars available (need 10).", style="yellow")
        return
    
    df = add_technical_indicators(df)
    df = clean_data_for_model(df)
    X, _ = create_features_and_labels(df)

    if X.empty or len(X.columns) < 1:
        console.print("⚠️ No valid features to predict.\n")
        return

    latest = X.tail(1)
    try:
        action = action_model.predict(latest)[0]
        confidence = action_model.predict_proba(latest)[0].max()
    except Exception as e:
        console.print(f"❌ Model prediction failed: {e}")
        return

    try:
        next_close = next_close_model.predict(latest)[0]
    except Exception as e:
        console.print(f"⚠️ Regression model failed: {e}")
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


def spinner_task(event, label):
    spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    start = time.time()
    while not event.is_set():
        elapsed = time.time() - start
        sys.stdout.write(f"\r⏳ {label} {elapsed:.1f}s {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.2)
    sys.stdout.write("\r✅ First OHLCV bar formed.                \n")


def main(symbol):
    global action_model, next_close_model

    # Header
    console.print(Panel("🧠 Loading models...", expand=False, border_style="green"))
    console.print("\n⚙️ Initializing prediction engine...")

    with Progress(
        SpinnerColumn(),
        "[bold blue]🔧 Warming up engine...",
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("warming_up", total=100)
        for _ in range(100):
            time.sleep(0.02)  # simulate work
            progress.update(task, advance=1)

    console.print("✅ [green]Engine ready.")

    if not is_valid_symbol(symbol):
        console.print(f"❌ Invalid symbol '{symbol.upper()}'", style="red")
        return

    ohlcv_buffer = OHLCVBuffer(interval_seconds=INTERVAL)
    bar_event = threading.Event()

    spinner_thread = threading.Thread(target=spinner_task, args=(bar_event, "Waiting for first OHLCV bar..."))
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
                console.print(f"⚠️ Skipping incomplete data: {data}")
                return

            price = float(price_str)
            volume = float(volume_str)
            timestamp = int(timestamp)

            prev_len = len(ohlcv_buffer.bars)
            ohlcv_buffer.update(price, volume, timestamp)
            new_len = len(ohlcv_buffer.bars)

            if new_len > prev_len:
                console.print(f"🕒 New OHLCV bar formed. Total bars: {new_len}")

            if new_len < 10:
                status_message = f"⏳ Waiting for more data... {new_len}/10 bars collected."
                console.status(status_message, spinner="dots", spinner_style="cyan")
                time.sleep(0.5)  # short pause to show spinner
                console.status("").stop()
            else:
                if not bar_event.is_set():
                    bar_event.set()
                df = ohlcv_buffer.get_dataframe()
                run_prediction(df)

        except Exception as e:
            console.print(f"❗ Error in message handling: {e}")

    def on_open(ws):
        ws.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@trade"],
            "id": 1
        }))
        console.print(f"📡 Subscribed to live trade stream for {symbol.upper()}")

    def on_error(ws, error):
        console.print(f"❌ WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        console.print(f"🔌 WebSocket closed. Code: {close_status_code}, Message: {close_msg}")

    console.print("🚀 Starting live stream...")
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{symbol}@trade",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    try:
        ws.run_forever()
    except KeyboardInterrupt:
        graceful_exit(None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Crypto Predictor")
    parser.add_argument("--symbol", required=True, help="Symbol like BTCUSDT")
    args = parser.parse_args()
    main(args.symbol.lower())
