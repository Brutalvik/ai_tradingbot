import sys
import time
import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm
from utils.fetch_data import fetch_binance_ohlcv
from utils.processing import create_features_and_labels

def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").upper()

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def retrain(symbol: str, interval: str = "1m", csv_path: str = None):
    print(f"[INFO] Starting retraining for symbol: {symbol}")

    # Load data
    if csv_path:
        print(f"[INFO] Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[INFO] Fetching OHLCV data from Binance for {symbol}...")
        since_ms = int((time.time() - 7*24*60*60) * 1000)  # Last 7 days
        df = fetch_binance_ohlcv(symbol, interval, since=since_ms)

    if df.empty:
        print(f"[ERROR] No data found. Exiting.")
        return

    X, y = create_features_and_labels(df)

    print(f"[INFO] Fetched {len(df)} rows of data for {symbol}")
    print(df.head())

    # Training model
    n_estimators = 100
    model = RandomForestClassifier(
        n_estimators=1,
        warm_start=True,
        random_state=42,
        n_jobs=-1
    )

    print("[INFO] Training model...")

    try:
        with tqdm(total=n_estimators, desc="Training Progress", unit="tree", colour="green") as pbar:
            for i in range(1, n_estimators + 1):
                model.set_params(n_estimators=i)
                model.fit(X, y)
                pbar.update(1)

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        with tqdm(total=n_estimators, desc="Training Failed", unit="tree", colour="red") as pbar:
            for _ in range(n_estimators):
                pbar.update(1)
                time.sleep(0.01)
        return

    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    safe_symbol = sanitize_symbol(symbol)
    time_tag = timestamp()

    model_filename = f"{safe_symbol}_{interval}_{time_tag}_model.pkl"
    latest_model_filename = f"{safe_symbol}_{interval}_latest_model.pkl"

    model_path = os.path.join(model_dir, model_filename)
    latest_model_path = os.path.join(model_dir, latest_model_filename)

    joblib.dump(model, model_path)
    joblib.dump(model, latest_model_path)

    print(f"[SUCCESS] Model saved: {model_path}")
    print(f"[INFO] Latest model also saved as: {latest_model_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retrain_model.py SYMBOL [INTERVAL] [CSV_FILE]")
        print("Example: python retrain_model.py BTC/USDT 5m")
        print("         python retrain_model.py ETH/USDT 1m data/eth_data.csv")
        sys.exit(1)

    input_symbol = sys.argv[1]
    input_interval = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith(".csv") else "1m"
    input_csv = sys.argv[3] if len(sys.argv) > 3 else (sys.argv[2] if sys.argv[2].endswith(".csv") else None)

    retrain(input_symbol, input_interval, input_csv)
    print
