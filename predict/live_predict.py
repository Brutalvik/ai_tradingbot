import json
import joblib
import websocket
import argparse
import requests
import threading
import time
import signal
import pandas as pd
from datetime import datetime

from utils.ohlcv_buffer import OHLCVBuffer
from utils.indicators import add_technical_indicators
from utils.label_data import clean_data_for_model

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TimeElapsedColumn, SpinnerColumn, TextColumn, MofNCompleteColumn
from rich import box

# Constants
MODEL_PATH = "models/BTCUSDT_1m_latest_model.pkl"
REGRESSION_MODEL_PATH = "models/next_close_regressor.pkl"
BINANCE_API_URL = "https://api.binance.com/api/v3/exchangeInfo"
INTERVAL = 60
CONFIDENCE_THRESHOLD = 0.6
MIN_BARS_REQUIRED = 30

console = Console()

# Enhanced graphical model loading
with Progress(
    SpinnerColumn("dots"),
    TextColumn("{task.description}"),
    BarColumn(),
    TimeElapsedColumn(),
    MofNCompleteColumn()
) as progress:
    task_action_model = progress.add_task("🔄 Loading action model", total=100)
    action_model = joblib.load(MODEL_PATH)
    for _ in range(100):
        time.sleep(0.003)
        progress.advance(task_action_model)

    task_regression_model = progress.add_task("🔄 Loading regression model", total=100)
    next_close_model = joblib.load(REGRESSION_MODEL_PATH)
    for _ in range(100):
        time.sleep(0.003)
        progress.advance(task_regression_model)

console.print("[bold green]✅ Models loaded successfully.[/bold green]")

def is_valid_symbol(symbol: str) -> bool:
    console.print(f"🔍 Validating symbol: [cyan]{symbol.upper()}[/cyan]")
    url = f"{BINANCE_API_URL}?symbol={symbol.upper()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        console.print("[bold green]✅ Symbol validation successful.[/bold green]")
        return True
    except Exception as e:
        console.print(f"[bold red]❌ Invalid symbol: {symbol.upper()} ({e})[/bold red]")
        return False

def run_prediction(df):
    with console.status("🔍 Running prediction..."):
        # Classifier features
        df['returns'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()

        # Regressor features
        df['return'] = df['close'].pct_change()
        df['sma'] = df['close'].rolling(window=5).mean()

        df.dropna(inplace=True)

        # Classifier prediction
        clf_features = ['returns', 'ma5', 'ma10']
        latest_clf = df[clf_features].tail(1)
        action = action_model.predict(latest_clf)[0]
        confidence = action_model.predict_proba(latest_clf)[0].max()

        # Regressor prediction
        reg_features = ['open', 'high', 'low', 'close', 'volume', 'return', 'sma']
        latest_reg = df[reg_features].tail(1)
        next_close = next_close_model.predict(latest_reg)[0]

        current_price = df.iloc[-1]['close']
        timestamp = df.iloc[-1]['timestamp']

        if confidence >= CONFIDENCE_THRESHOLD:
            signal = "🟢 BUY" if action == 1 else "🔴 SELL"
            console.print(Panel(f"""
{signal} SIGNAL
📈 Price: ${current_price:.2f}
📅 Time: {timestamp}
📊 Confidence: {confidence * 100:.2f}%
🔮 Predicted Close: ${next_close:.2f}
📌 Suggested Limit {'Buy' if action == 1 else 'Sell'} at: ${next_close:.2f}
""", box=box.DOUBLE_EDGE, expand=False))
        else:
            console.print(f"❔ Weak signal: {confidence * 100:.2f}% < {CONFIDENCE_THRESHOLD} — {timestamp}")

def start_spinner_with_timer(stop_event, ohlcv_buffer):
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("🕑 Waiting for OHLCV bars...", total=MIN_BARS_REQUIRED)
        last_count = 0
        while not stop_event.is_set():
            current_count = len(ohlcv_buffer.bars)
            if current_count != last_count:
                progress.update(task, completed=current_count)
                last_count = current_count
            time.sleep(0.1)

# Signal handler for graceful termination
def signal_handler(sig, frame):
    console.print("\n[bold red]🛑 Program terminated by user. Goodbye![/bold red]")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main(symbol: str):
    if not is_valid_symbol(symbol):
        return

    ohlcv_buffer = OHLCVBuffer(interval_seconds=INTERVAL)
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=start_spinner_with_timer, args=(stop_spinner, ohlcv_buffer))
    spinner_thread.start()

    def on_message(ws, message):
        data = json.loads(message)
        if data.get("e") != "trade":
            return

        price = float(data['p'])
        volume = float(data['q'])
        timestamp = int(data['T'])

        previous_count = len(ohlcv_buffer.bars)
        ohlcv_buffer.update(price, volume, timestamp)
        new_count = len(ohlcv_buffer.bars)

        if new_count > previous_count:
            console.print(f"✅ OHLCV bar formed. Total bars: [green]{new_count}/{MIN_BARS_REQUIRED}[/green]")

        if new_count >= MIN_BARS_REQUIRED:
            stop_spinner.set()
            df = ohlcv_buffer.get_dataframe()
            run_prediction(df)

    console.print("🚀 Starting live stream...")
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{symbol}@trade",
        on_open=lambda ws: console.print(f"📡 Subscribed to {symbol.upper()}"),
        on_message=on_message,
        on_error=lambda ws, err: console.print(f"❌ WebSocket error: {err}"),
        on_close=lambda ws, code, msg: console.print(f"🔌 WebSocket closed. Code: {code}, Message: {msg}"),
    )
    ws.run_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Crypto Predictor")
    parser.add_argument("--symbol", required=True, help="Symbol like BTCUSDT")
    args = parser.parse_args()
    main(args.symbol.lower())
