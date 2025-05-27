# live_streamer.py

import websocket
import json
import threading
import datetime
import sys

def start_stream(symbol: str):
    def on_message(ws, message):
        data = json.loads(message)
        trade_price = float(data['p'])
        quantity = float(data['q'])
        timestamp = datetime.datetime.fromtimestamp(data['T'] / 1000.0)

        print(f"[{timestamp}] {symbol.upper()} Price: ${trade_price:.6f} | Qty: {quantity}")

    def on_error(ws, error):
        print("❌ WebSocket error:", error)

    def on_close(ws, close_status_code, close_msg):
        print("❗ WebSocket closed:", close_msg)

    def on_open(ws):
        print(f"✅ Subscribing to live trades for {symbol.upper()}...")
        payload = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@trade"],
            "id": 1
        }
        ws.send(json.dumps(payload))

    url = "wss://stream.binance.com:9443/ws"
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Usage: python live_streamer.py <symbol>")
        print("🔎 Example: python live_streamer.py dogeusdt")
        sys.exit(1)

    symbol = sys.argv[1].lower()
    thread = threading.Thread(target=start_stream, args=(symbol,))
    thread.start()
