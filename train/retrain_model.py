import sys
import time
import os
import joblib
import pandas as pd
import threading
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from utils.fetch_data import fetch_binance_ohlcv
from utils.processing import create_features_and_labels


def load_model_with_progress(model_path: str):
    result = {}

    def load():
        result['model'] = joblib.load(model_path)

    # Start loading the model in a background thread
    loader_thread = threading.Thread(target=load)
    loader_thread.start()

    spinner = ['|', '/', '-', '\\']
    idx = 0

    # Show a rotating progress bar while loading
    with tqdm(total=0, desc="Loading model --> Please wait", bar_format="{desc} {postfix}", colour="cyan") as pbar:
        pbar.set_postfix_str("⏳")
        while loader_thread.is_alive():
            time.sleep(0.1)
            current = pbar.postfix if hasattr(pbar, "postfix") else "⏳"
            pbar.set_postfix_str(current + ".")
            if len(current) > 6:
                pbar.set_postfix_str("⏳")
            
            pbar.refresh()


    loader_thread.join()
    return result['model']


def sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()

def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    print(f"[INFO] Normalizing symbol: {symbol}")
    if "/" not in symbol and len(symbol) >= 6:
        return f"{symbol[:-4]}/{symbol[-4:]}"
    return symbol

def retrain(symbol: str, interval: str = "1m", csv_path: str = None):
    print(f"[INFO] Starting retraining for symbol: {symbol}")

    # === Load data ===
    if csv_path:
        print(f"[INFO] Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[INFO] Fetching OHLCV data from Binance for {symbol}...")
        since_ms = int((time.time() - 7*24*60*60) * 1000)
        df = fetch_binance_ohlcv(symbol, interval, since=since_ms)

    if df.empty:
        print(f"[ERROR] No data found. Exiting.")
        return

    X, y = create_features_and_labels(df)

    print(f"[INFO] Fetched {len(df)} rows of data for {symbol}")
    print(df.head())

    # === Load or initialize model ===
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    safe_symbol = sanitize_symbol(symbol)
    print(f"safe_symbol {safe_symbol}")
    base_filename = f"{safe_symbol}_{interval}"
    original_model_path = os.path.join(model_dir, f"{base_filename}_model.pkl")
    latest_model_path = os.path.join(model_dir, f"{base_filename}_latest_model.pkl")

    additional_estimators = 100
    initial_model_created = False

    if os.path.exists(latest_model_path):
        print(f"[INFO] Existing model found: {latest_model_path}")
        model = load_model_with_progress(latest_model_path)
        if not getattr(model, "warm_start", False):
            print("[WARNING] Model doesn't support warm_start. Reinitializing new model.")
            model = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=42, n_jobs=-1)
            current_estimators = 0
        else:
            current_estimators = model.n_estimators
    else:
        print(f"[INFO] No existing model found. Creating new model.")
        model = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=42, n_jobs=-1)
        current_estimators = 0
        initial_model_created = True

    # === Train additional estimators ===
    print(f"[INFO] Training model from {current_estimators} to {current_estimators + additional_estimators} estimators...")
    sys.stdout.flush()

    try:
        # Enhanced progress with multiple phases
        with tqdm(total=additional_estimators + 3, desc="Initializing", unit="step", colour="green") as pbar:
            pbar.set_description("Preparing model...")
            time.sleep(0.2)
            pbar.update(1)

            pbar.set_description("Preparing features...")
            time.sleep(0.2)
            pbar.update(1)

            pbar.set_description("Finalizing setup...")
            time.sleep(0.2)
            pbar.update(1)

            pbar.set_description("Training model...")
            
            for i in range(current_estimators + 1, current_estimators + additional_estimators + 1):
                model.set_params(n_estimators=i)
                model.fit(X, y)
                pbar.update(1)

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        with tqdm(total=additional_estimators, desc="Training Failed", unit="tree", colour="red") as pbar:
            for _ in range(additional_estimators):
                pbar.update(1)
                time.sleep(0.01)
        return

    # === Save updated model ===
    if initial_model_created:
        joblib.dump(model, original_model_path)
        print(f"[SUCCESS] Initial model saved: {original_model_path}")

    joblib.dump(model, latest_model_path)
    print(f"[SUCCESS] Updated latest model saved: {latest_model_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retrain_model.py SYMBOL [INTERVAL] [CSV_FILE]")
        print("Example: python retrain_model.py BTC/USDT 5m")
        print("         python retrain_model.py ETH/USDT 1m data/eth_data.csv")
        sys.exit(1)

    raw_symbol = sys.argv[1]
    input_symbol = normalize_symbol(raw_symbol)

    input_interval = "1m"
    input_csv = None

    if len(sys.argv) >= 3:
        if sys.argv[2].endswith(".csv"):
            input_csv = sys.argv[2]
        else:
            input_interval = sys.argv[2]

    if len(sys.argv) >= 4:
        input_csv = sys.argv[3]
    
    retrain(input_symbol, input_interval, input_csv)
