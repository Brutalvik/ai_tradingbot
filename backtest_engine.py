import pandas as pd
import joblib
from utils.indicators import add_technical_indicators
from utils.label_data import clean_data_for_model
from datetime import datetime
from rich.progress import track
from rich.console import Console

console = Console()

# Paths to models
CLASSIFIER_PATH = "models/BTCUSDT_1m_latest_model.pkl"
REGRESSOR_PATH = "models/next_close_regressor.pkl"

def load_models():
    console.print("[bold blue]🔄 Loading models...[/bold blue]")
    clf_model = joblib.load(CLASSIFIER_PATH)
    reg_model = joblib.load(REGRESSOR_PATH)
    console.print("[green]✅ Models loaded successfully.[/green]")
    return clf_model, reg_model

def prepare_data(df):
    console.print("[bold blue]🔄 Preparing and cleaning data...[/bold blue]")
    df = add_technical_indicators(df)
    df['returns'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['return'] = df['close'].pct_change()
    df['sma'] = df['close'].rolling(window=5).mean()
    df.dropna(inplace=True)
    console.print(f"[green]✅ Data prepared. {len(df)} rows after cleaning.[/green]")
    return df

def backtest(df, clf_model, reg_model, threshold=0.6):
    df = prepare_data(df)
    sample_size = int(len(df) * 0.1)  # Use only 10% of data
    df = df.head(sample_size)

    clf_features = ['returns', 'ma5', 'ma10']
    reg_features = ['open', 'high', 'low', 'close', 'volume', 'return', 'sma']

    signals = []

    console.print("[bold blue]📊 Running backtest...[/bold blue]")
    for _, row in track(df.iterrows(), total=len(df), description="Processing bars"):
        clf_input = row[clf_features].to_frame().T
        reg_input = row[reg_features].to_frame().T

        action = clf_model.predict(clf_input)[0]
        confidence = clf_model.predict_proba(clf_input)[0].max()
        predicted_close = reg_model.predict(reg_input)[0]

        if confidence >= threshold:
            signal_type = "BUY" if action == 1 else "SELL"
            signals.append({
                "timestamp": row['timestamp'],
                "price": row['close'],
                "signal": signal_type,
                "predicted_close": predicted_close,
                "confidence": confidence
            })

    console.print(f"[green]✅ Backtest complete. {len(signals)} signals generated.[/green]")
    return pd.DataFrame(signals)

if __name__ == "__main__":
    console.print("[bold blue]📥 Loading historical data...[/bold blue]")
    df = pd.read_csv("data/BTC-USDT.csv")  # Adjust path as needed
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    console.print(f"[green]✅ Loaded {len(df)} rows of historical data.[/green]")

    clf_model, reg_model = load_models()
    result = backtest(df, clf_model, reg_model)

    console.print("\n[bold blue]📄 Backtest Results:[/bold blue]")
    console.print(result.head())
    result.to_csv("backtest_results.csv", index=False)
    console.print("[green]📁 Results saved to backtest_results.csv[/green]")
