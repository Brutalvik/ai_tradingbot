import json
import joblib
import websocket
import argparse
import requests
import threading
import time
from datetime import datetime

from utils.ohlcv_buffer import OHLCVBuffer
from utils.indicators import add_technical_indicators
from utils.label_data import clean_data_for_model
from utils.processing import create_features_and_labels

from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich import box

# Constants
MODEL_PATH = "models/BTCUSDT_1m_latest_model.pkl"
REGRESSION_MODEL_PATH = "models/next_close_regressor.pkl"
BINANCE_API_URL = "https://api.binance.com/api/v3/exchangeInfo"
INTERVAL = 60  # 1-minute bars
CONFIDENCE_THRESHOLD = 0.6
MIN_BARS_REQUIRED = 30

# Rich console
console = Console()

# Load models
console.print(Panel("🧠 Loading models...", expand=False, box=box.ROUNDED))
action_model = joblib.load(MODEL_PATH)
next_close_model = joblib.load(REGRESSION_MODEL_PATH)
time.sleep(0.5)
console.print("[green]✅ Models loaded successfully.[/green]")

def is_valid_symbol(symbol: str) -> bool:
    console.print(f"\n🔍 Validating symbol: [cyan]{symbol.upper()}[/cyan]")
    url = f"{BINANCE_API_URL}?symbol={symbol.upper()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return True
    except Exception as e:
        console.print(f"[red]❌ Invalid symbol: {symbol.upper()} ({e})[/red]")
        return False

def run_prediction(df):
    try:
        df = add_technical_indicators(df)
        df = clean_data_for_model(df)
        X, _ = create_features_and_labels(df)

        if X.empty or len(X) < 1:
            console.print("[yellow]⚠️ No valid features for prediction.[/yellow]")
            return

        latest = X.tail(1)
        action = action_model.predict(latest)[0]
        confidence = action_model.predict_proba(latest)[0].max()

        try:
            next_close = next_close_model.predict(latest)[0]
        except Exception as e:
            console.print(f"[yellow]⚠️ Regression model failed: {e}[/yellow]")
            next_close = df.iloc[-1]['close']

        current_price = df.iloc[-1]['close']
        timestamp = df.iloc[-1]['timestamp']

        if confidence >= CONFIDENCE_THRESHOLD:
            signal = "🟢 BUY" if action == 1 else "🔴 SELL"
            console.print(f"""
[bold]{signal} SIGNAL[/bold]
📈 Price: [cyan]${current_price:.2f}[/cyan]
📅 Time: [blue]{timestamp}[/blue]
📊 Confidence: [magenta]{confidence * 100:.2f}%[/magenta]
🔮 Predicted Close: [green]${next_close:.2f}[/green]
📌 Suggested Limit {'Buy' if action == 1 else 'Sell'} at: [green]${next_close:.2f}[/green]
""")
        else:
            console.print(f"❔ Weak signal: {confidence * 100:.2f}% < {CONFIDENCE_THRESHOLD} — {timestamp}")
    except Exception as e:
        console.print(f"[red]❗ Prediction error: {e}[/red]")

def start_spinner_with_timer(stop_event):
    spinner = Spinner("dots", text="⏳ Waiting for first OHLCV bar...")
    with console.status(spinner, spinner_style="cyan"):
        start = time.time()
        while not stop_event.is_set():
            time.sleep(0.1)
        elapsed = time.time() - start
    console.print(f"✅ First OHLCV bar formed in [green]{elapsed:.1f}[/green] seconds.")

def main(symbol: str):
    if not is_valid_symbol(symbol):
        return

    ohlcv_buffer = OHLCVBuffer(interval_seconds=INTERVAL)
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=start_spinner_with_timer, args=(stop_spinner,))
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
                console.print(f"[yellow]⚠️ Skipping message with missing fields: {data}[/yellow]")
                return

            price = float(price_str)
            volume = float(volume_str)
            timestamp = int(timestamp)

            previous_count = len(ohlcv_buffer.bars)
            ohlcv_buffer.update(price, volume, timestamp)
            new_count = len(ohlcv_buffer.bars)

            if new_count > previous_count:
                console.print(f"🕒 New OHLCV bar formed. Total bars: {new_count}")
                if new_count >= MIN_BARS_REQUIRED:
                    stop_spinner.set()
                    df = ohlcv_buffer.get_dataframe()
                    run_prediction(df)
                else:
                    console.print(f"[yellow]⏳ Not enough bars yet: {new_count}/{MIN_BARS_REQUIRED}[/yellow]")

        except Exception as e:
            console.print(f"[red]❗ Error in message handling: {e}[/red]")

    def on_open(ws):
        ws.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@trade"],
            "id": 1
        }))
        console.print(f"[green]📡 Subscribed to live trade stream for {symbol.upper()}[/green]")

    def on_error(ws, error):
        console.print(f"[red]❌ WebSocket error: {error}[/red]")

    def on_close(ws, code, msg):
        console.print(f"[red]🔌 WebSocket closed. Code: {code}, Message: {msg}[/red]")

    console.print("🚀 Starting live stream...")
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{symbol}@trade",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Crypto Predictor")
    parser.add_argument("--symbol", required=True, help="Symbol like BTCUSDT")
    args = parser.parse_args()
    main(args.symbol.lower())
