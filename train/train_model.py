import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from joblib import dump
import os

from utils.fetch_data import fetch_binance_ohlcv
from utils.indicators import add_technical_indicators
from utils.label_data import add_labels

def main():
    df = fetch_binance_ohlcv()
    df = add_technical_indicators(df)
    df = add_labels(df)
    df.dropna(inplace=True)

    features = df.drop(['timestamp', 'label', 'future_return', 'open', 'high', 'low', 'close'], axis=1)
    labels = df['label']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=False, test_size=0.2)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', base_score=0.5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    dump(model, 'models/model.joblib')
    print("Model saved to models/model.joblib")

if __name__ == "__main__":
    main()
