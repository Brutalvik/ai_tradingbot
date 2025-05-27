def create_features_and_labels(df):
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['future_close'] = df['close'].shift(-1)

    # Label: 1 = Buy, -1 = Sell, 0 = Hold
    df['signal'] = 0
    df.loc[df['future_close'] > df['close'], 'signal'] = 1
    df.loc[df['future_close'] < df['close'], 'signal'] = -1

    df.dropna(inplace=True)

    X = df[['returns', 'ma5', 'ma10']]
    y = df['signal']

    return X, y
