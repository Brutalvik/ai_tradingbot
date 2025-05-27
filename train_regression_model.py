import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# 1. Load your OHLCV data
df = pd.read_csv("data/train_data.csv")  # adjust path

# 2. Create features — same as your classifier (or more advanced)
df['return'] = df['close'].pct_change()
df['sma'] = df['close'].rolling(window=5).mean()
df = df.dropna()

# 3. Define X and y
X = df[['open', 'high', 'low', 'close', 'volume', 'return', 'sma']]
y = df['close'].shift(-1)  # next candle close price
X = X[:-1]
y = y[:-1]

# 4. Train regressor
regressor = GradientBoostingRegressor()
regressor.fit(X, y)

# 5. Save model
joblib.dump(regressor, "models/next_close_regressor.pkl")
print("✅ Next candle close regressor trained and saved.")
