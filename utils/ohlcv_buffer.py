# utils/ohlcv_buffer.py

from collections import deque
import pandas as pd
from datetime import datetime, timedelta

class OHLCVBuffer:
    def __init__(self, interval_seconds=60, max_bars=200):
        self.interval_seconds = interval_seconds
        self.max_bars = max_bars
        self.bars = deque(maxlen=max_bars)
        self.current_bar = None
        self.current_bar_start = None

    def update(self, price, volume, timestamp):
        ts_dt = datetime.fromtimestamp(timestamp / 1000)
        ts_start = ts_dt.replace(second=0, microsecond=0)

        if not self.current_bar or ts_start != self.current_bar_start:
            if self.current_bar:
                self.bars.append(self.current_bar)
            self.current_bar_start = ts_start
            self.current_bar = {
                'timestamp': ts_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            self.current_bar['high'] = max(self.current_bar['high'], price)
            self.current_bar['low'] = min(self.current_bar['low'], price)
            self.current_bar['close'] = price
            self.current_bar['volume'] += volume

    def get_dataframe(self):
        return pd.DataFrame(list(self.bars))
