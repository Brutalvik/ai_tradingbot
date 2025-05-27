from ta import add_all_ta_features

def add_technical_indicators(df):
    df = add_all_ta_features(
        df, open='open', high='high', low='low',
        close='close', volume='volume', fillna=True
    )
    return df
