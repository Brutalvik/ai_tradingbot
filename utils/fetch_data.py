import time
import pandas as pd
import ccxt  # Make sure ccxt is installed: pip install ccxt

exchange = ccxt.binance({
    'enableRateLimit': True,
})

def fetch_binance_ohlcv(symbol, interval, since=None, limit=1000, max_chunks=100):
    print(f"Fetching OHLCV data for {symbol} at {interval} interval, starting from {since} with limit {limit}...")
    """
    Fetch all OHLCV data in chunks using pagination. 
    Params:
        symbol (str): Trading pair like 'DOGE/USDT'
        interval (str): Timeframe like '1m', '5m', '1h'
        since (int or None): Unix ms timestamp to start fetching from (None = most recent)
        limit (int): max candles per API call (usually 1000 max)
        max_chunks (int): max number of chunks to fetch to avoid infinite loops
    Returns:
        pd.DataFrame: concatenated OHLCV data
    """
    all_data = []
    current_since = since
    chunk_count = 0

    while chunk_count < max_chunks:
        print(f"Fetching chunk {chunk_count + 1}...")
        # Call ccxt's fetch_ohlcv, NOT this function recursively
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, since=current_since, limit=limit)
        if not ohlcv:
            print("No data fetched in this chunk, stopping.")
            break

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        all_data.append(df)

        last_timestamp = df['timestamp'].iloc[-1]
        print(f"Chunk {chunk_count + 1} fetched {len(df)} rows. Last timestamp: {last_timestamp}")

        # Move since cursor forward to avoid overlap (add 1 ms)
        current_since = int(df['timestamp'].iloc[-1].timestamp() * 1000) + 1

        chunk_count += 1

        # If fewer rows than limit, likely no more data
        if len(df) < limit:
            print("Fewer rows than limit, assuming no more data.")
            break

        time.sleep(exchange.rateLimit / 1000)  # respect rate limit

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.drop_duplicates(subset='timestamp', inplace=True)
        full_df.sort_values('timestamp', inplace=True)
        full_df.reset_index(drop=True, inplace=True)
        print(f"Total fetched rows after concat: {len(full_df)}")
        return full_df
    else:
        print("No data fetched, returning empty DataFrame.")
        return pd.DataFrame()
