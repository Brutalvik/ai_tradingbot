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
import itertools, threading, time, sys
from rich.console import Console
from rich.spinner import Spinner
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.progress import track

MODEL_PATH = "models/BTCUSDT_1m_latest_model.pkl"
REGRESSION_MODEL_PATH = "models/next_close_regressor.pkl"
BINANCE_API_URL = "https://api.binance.com/api/v3/exchangeInfo"

INTERVAL = 60  # 1-minute bars
CONFIDENCE_THRESHOLD = 0.6
REQUIRED_BARS = 10

console = Console()

# Load models
console.print(Panel("🧠 [bold blue]Loading models...", expand=False))
action_model = joblib.load(MODEL_PATH)
next_close_model = joblib.load(REGRESSION_MODEL_PATH)

# Simulated loading steps
def simulate_progress_steps(steps):
    for step in track(steps, description="[cyan]📦 Preparing models and environment..."):
        time.sleep(0.3)  # Simulate delay per step

def start_spinner_with_timer(stop_event):
    start_time = time.time()

    with Live(refresh_per_second=10) as live:
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            spinner = Spinner("dots", text=f"⏳ Waiting for first OHLCV bar... {elapsed:.1f}s")
            live.update(spinner)
            time.sleep(0.1)

    console.print("\n✅ [green]First OHLCV bar formed.[/green]")

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
    if len(df) < REQUIRED_BARS:
        console.print(f"🕒 [yellow]Not enough bars yet ({len(df)}/{REQUIRED_BARS}). Waiting...[/yellow]")
        return

    df = add_technical_indicators(df)
    df = clean_data_for_model(df)
    X, _ = create_features_and_labels(df)

    if X.empty:
        console.print("[bold yellow]⚠️ No valid features to predict.[/bold yellow]")
        return

    latest = X.tail(1)
    action = action_model.predict(latest)[0]
    confidence = action_model.predict_proba(latest)[0].max()

    try:
        next_close = next_close_model.predict(latest)[0]
    except Exception as e:
        console.print(f"[red]⚠️ Regression model failed: {e}[/red]")
        next_close = df.iloc[-1]['close']

    current_price = df.iloc[-1]['close']
    timestamp = df.iloc[-1]['timestamp']

    if confidence >= CONFIDENCE_THRESHOLD:
        signal_text = Text("🟢 BUY" if action == 1 else "🔴 SELL", style="bold green" if action == 1 else "bold red")
        console.print(Panel.fit(
            f"[bold]🔔 Trade Signal:[/bold] {signal_text}\n"
            f"[cyan]📈 Current Price:[/cyan] ${current_price:.2f}\n"
            f"[magenta]📅 Time:[/magenta] {timestamp}\n"
            f"[bold yellow]📊 Confidence:[/bold yellow] {confidence * 100:.2f}%\n"
            f"[blue]🔮 Predicted Next Close:[/blue] ${next_close:.2f}\n"
            f"[white]📌 Suggested Limit {'Buy' if action == 1 else 'Sell'} at ${next_close:.2f}[/white]",
            border_style="green" if action == 1 else "red"
        ))
    else:
        console.print(f"[grey]❔ Weak signal ({confidence * 100:.2f}% < {CONFIDENCE_THRESHOLD}) — {timestamp}[/grey]")

def main(symbol):
    console.print(Panel("🧠 Loading models...", style="bold magenta"))
    symbol = symbol.lower()
    simulate_progress_steps([
    "🔧 Initializing buffers",
    "📂 Loading indicators",
    "🧪 Preparing ML pipeline",
    "🔒 Setting up WebSocket stream",
    ])

    console.print(f"🔍 [bold]Validating symbol:[/bold] {symbol.upper()}")

    if not is_valid_symbol(symbol):
        console.print(f"[red]❌ Invalid symbol '{symbol.upper()}'.[/red]")
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

            previous_bar_count = len(ohlcv_buffer.bars)
            ohlcv_buffer.update(price, volume, timestamp)
            new_bar_count = len(ohlcv_buffer.bars)

            if new_bar_count > previous_bar_count:
                if not stop_spinner.is_set():
                    stop_spinner.set()

                console.print(f"[blue]🕒 New OHLCV bar formed. Total bars: {new_bar_count}[/blue]")
                df = ohlcv_buffer.get_dataframe()
                run_prediction(df)

        except Exception as e:
            console.print(f"[bold red]❗ Error in message handling: {e}[/bold red]")

    def on_open(ws):
        ws.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@trade"],
            "id": 1
        }))
        console.print(f"📡 [green]Subscribed to live trade stream for {symbol.upper()}[/green]")

    def on_error(ws, error):
        console.print(f"[red]❌ WebSocket error: {error}[/red]")

    def on_close(ws, close_status_code, close_msg):
        console.print(f"[red]🔌 WebSocket closed. Code: {close_status_code}, Message: {close_msg}[/red]")

    console.print("[bold green]🚀 Starting live stream...[/bold green]")
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
    main(args.symbol)
