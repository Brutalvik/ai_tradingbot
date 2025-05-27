def add_labels(df, future_period=5, threshold=0.02):
    df['future_return'] = df['close'].shift(-future_period) / df['close'] - 1
    df['label'] = (df['future_return'] > threshold).astype(int)
    return df
