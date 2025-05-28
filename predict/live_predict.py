import json
import joblib
import websocket
import argparse
import requests
import threading
import time
import signal
import pandas as pd
import os
from datetime import datetime

from utils.ohlcv_buffer import OHLCVBuffer
from utils.indicators import add_technical_indicators
from utils.label_data import clean_data_for_model

from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, BarColumn, TimeElapsedColumn, SpinnerColumn, TextColumn
from rich import box

# Constants
MODEL_PATH = "models/BTCUSDT_1m_latest_model.pkl"
REGRESSION_MODEL_PATH = "models/next_close_regressor.pkl"
BINANCE_API_URL = "https://api.binance.com/api/v3/exchangeInfo"
INTERVAL = 60
CONFIDENCE_THRESHOLD = 0.6
MIN_BARS_REQUIRED = 30

console = Console()
start_time = time.time()
shutdown_requested = False
last_signal = None  # Track the last signal to suppress duplicates

class StatusView:
    def __init__(self):
        self.spinner = Spinner("dots")
        self.bar_count = 0
        self.confidence_text = "⏳ Waiting for strong signal..."

    def set_count(self, count):
        self.bar_count = count

    def set_confidence(self, confidence, timestamp):
        pct = f"{confidence * 100:.2f}%"
        self.confidence_text = f"⚠️ Weak signal: {pct} < {CONFIDENCE_THRESHOLD*100:.0f}% at {timestamp}"

    def render(self):
        elapsed = int(time.time() - start_time)
        h, m = divmod(elapsed // 60, 60)
        s = elapsed % 60
        spinner_frame = self.spinner.frames[int(time.time() * self.spinner.speed) % len(self.spinner.frames)]
        spinner_line = f"{spinner_frame} 🕑 Bars Formed: {self.bar_count}/{MIN_BARS_REQUIRED}  [Elapsed Time: {h:02}:{m:02}:{s:02}]"
        progress_bar = "▰" * self.bar_count + "▱" * (MIN_BARS_REQUIRED - self.bar_count)
        lines = [spinner_line, progress_bar]
        if self.confidence_text:
            lines.append(self.confidence_text)
        return Panel("\n".join(lines), box=box.ROUNDED, padding=(1, 2), title="Live Prediction Status")

status_view = StatusView()

# Model loading with progress bar
with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]🔄 Loading Models...[/bold blue]"),
    BarColumn(),
    TimeElapsedColumn(),
    console=console,
    transient=True
) as progress:
    task = progress.add_task("Loading", total=100)
    for _ in range(50):
        time.sleep(0.01)
        progress.advance(task)
    action_model = joblib.load(MODEL_PATH)
    for _ in range(50):
        time.sleep(0.01)
        progress.advance(task)
    next_close_model = joblib.load(REGRESSION_MODEL_PATH)

console.print("✅ Models loaded successfully.")

# Start live dashboard
live = Live(status_view.render(), refresh_per_second=4, console=console)
live.start()

def play_alert():
    try:
        if os.name == 'posix':
            if hasattr(os, "uname") and os.uname().sysname == 'Darwin':
                os.system("afplay alert.mp3")
            else:
                os.system("aplay alert.wav")
        elif os.name == 'nt':
            import winsound
            winsound.PlaySound("alert.wav", winsound.SND_FILENAME)
    except Exception as e:
        console.log(f"[red]Sound alert failed:[/red] {e}")

def run_prediction(df):
    global last_signal
    df['returns'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['return'] = df['close'].pct_change()
    df['sma'] = df['close'].rolling(window=5).mean()
    df.dropna(inplace=True)

    clf_features = ['returns', 'ma5', 'ma10']
    reg_features = ['open', 'high', 'low', 'close', 'volume', 'return', 'sma']

    latest_clf = df[clf_features].tail(1)
    latest_reg = df[reg_features].tail(1)

    action = action_model.predict(latest_clf)[0]
    confidence = action_model.predict_proba(latest_clf)[0].max()
    next_close = next_close_model.predict(latest_reg)[0]

    timestamp = df.iloc[-1]['timestamp']
    current_price = df.iloc[-1]['close']

    if last_signal == (timestamp, action):
        return
    last_signal = (timestamp, action)

    if confidence >= CONFIDENCE_THRESHOLD:
        signal = "🟢 BUY" if action == 1 else "🔴 SELL"
        console.print(Panel(
            f"{signal} SIGNAL\n"
            f"📈 Price: ${current_price:.2f}\n"
            f"🔮 Predicted Close: ${next_close:.2f}\n"
            f"📊 Confidence: {confidence * 100:.2f}%\n"
            f"📅 Time: {timestamp}",
            title="Action Signal", box=box.DOUBLE, style="green"
        ))
        play_alert()
    else:
        status_view.set_confidence(confidence, timestamp)

def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
    live.update(Panel("🛑 Terminating program...", style="bold red"))
    time.sleep(0.5)
    live.update(Panel("🧹 Cleaning up resources...", style="yellow"))
    time.sleep(0.5)
    live.update(Panel("👋 Exiting. Goodbye!", style="green"))
    time.sleep(0.5)
    live.stop()
    console.print("✔ Back to shell.")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main(symbol):
    url = f"{BINANCE_API_URL}?symbol={symbol.upper()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        console.print(f"✅ Symbol {symbol.upper()} validated.")
    except Exception as e:
        console.print(f"❌ Invalid symbol: {e}")
        return

    ohlcv_buffer = OHLCVBuffer(interval_seconds=INTERVAL)

    def spinner_loop():
        while not shutdown_requested:
            live.update(status_view.render())
            time.sleep(0.1)

    threading.Thread(target=spinner_loop, daemon=True).start()

    def on_message(ws, msg):
        data = json.loads(msg)
        if data.get("e") != "trade":
            return

        price = float(data['p'])
        volume = float(data['q'])
        timestamp = int(data['T'])

        ohlcv_buffer.update(price, volume, timestamp)
        count = len(ohlcv_buffer.bars)
        status_view.set_count(count)

        if count >= MIN_BARS_REQUIRED:
            df = ohlcv_buffer.get_dataframe()
            run_prediction(df)

    console.print("🚀 Starting live stream...")
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{symbol}@trade",
        on_open=lambda ws: console.print(f"📡 Subscribed to {symbol.upper()}"),
        on_message=on_message,
        on_error=lambda ws, err: console.print(f"❌ WebSocket error: {err}"),
        on_close=lambda ws, code, msg: console.print(f"🔌 Connection closed: {msg}")
    )
    ws.run_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    main(args.symbol.lower())
