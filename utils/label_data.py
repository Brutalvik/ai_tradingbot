def clean_data_for_model(df):
    # Drop target, future_return, and any datetime columns
    drop_cols = ['target', 'future_return']

    # Drop 'timestamp' or index if it's datetime
    if 'timestamp' in df.columns:
        drop_cols.append('timestamp')

    df = df.drop(columns=drop_cols, errors='ignore')

    # Drop non-numeric columns just in case
    df = df.select_dtypes(include=['number'])

    return df
