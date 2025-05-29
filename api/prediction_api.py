# prediction_api.py — FastAPI server that sends predictions + OHLCV via WebSocket

import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from utils.ohlcv_buffer import OHLCVBuffer
from datetime import datetime
from threading import Thread
import joblib
import time
import pandas as pd
import websocket

app = FastAPI()

# Allow frontend (React app) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients = set()
ohlcv_buffer = OHLCVBuffer(interval_seconds=60)
MODEL_PATH = "models/BTCUSDT_1m_latest_model.pkl"
REGRESSION_MODEL_PATH = "models/next_close_regressor.pkl"
CONFIDENCE_THRESHOLD = 0.6
MIN_BARS_REQUIRED = 30

action_model = joblib.load(MODEL_PATH)
reg_model = joblib.load(REGRESSION_MODEL_PATH)
last_signal = None

@app.websocket("/ws/predictions")
async def prediction_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

def push_update(data):
    json_data = json.dumps(data)
    for ws in connected_clients.copy():
        try:
            asyncio.run(ws.send_text(json_data))
        except Exception:
            connected_clients.discard(ws)

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
    next_close = reg_model.predict(latest_reg)[0]
    timestamp = df.iloc[-1]['timestamp']
    price = df.iloc[-1]['close']

    if last_signal == (timestamp, action):
        return
    last_signal = (timestamp, action)

    result = {
        "timestamp": timestamp,
        "price": price,
        "predicted_close": next_close,
        "confidence": round(confidence * 100, 2),
        "action": "BUY" if action == 1 else "SELL",
        "bars_formed": len(ohlcv_buffer.bars)
    }
    push_update(result)

def on_message(ws, msg):
    data = json.loads(msg)
    if data.get("e") != "trade":
        return
    price = float(data['p'])
    volume = float(data['q'])
    timestamp = int(data['T'])
    ohlcv_buffer.update(price, volume, timestamp)

    if len(ohlcv_buffer.bars) >= MIN_BARS_REQUIRED:
        df = ohlcv_buffer.get_dataframe()
        run_prediction(df)

def start_stream(symbol):
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{symbol}@trade",
        on_message=on_message,
        on_error=lambda ws, err: print("WebSocket error:", err),
        on_close=lambda ws, code, msg: print("WebSocket closed:", msg),
        on_open=lambda ws: print(f"📡 Subscribed to {symbol.upper()}")
    )
    ws.run_forever()

def run_bot():
    symbol = "btcusdt"
    t = Thread(target=start_stream, args=(symbol,), daemon=True)
    t.start()
    print("✅ Background trading bot running...")

run_bot()