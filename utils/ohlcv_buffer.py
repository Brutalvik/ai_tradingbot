# utils/ohlcv_buffer.py

from collections import deque
import pandas as pd
from datetime import datetime, timedelta

class OHLCVBuffer:
    def __init__(self, interval_seconds=60, max_bars=200):
        self.interval = timedelta(seconds=interval_seconds)
        self.current_bar = None
        self.last_timestamp = None
        self.bars = deque(maxlen=max_bars)

    def update(self, tick_price, tick_volume, tick_time):
        tick_dt = datetime.fromtimestamp(tick_time / 1000)

        # Start a new bar
        if self.current_bar is None or tick_dt >= self.last_timestamp + self.interval:
            if self.current_bar:
                self.bars.append(self.current_bar)

            self.last_timestamp = tick_dt.replace(second=0, microsecond=0)
            self.current_bar = {
                "timestamp": self.last_timestamp,
                "open": tick_price,
                "high": tick_price,
                "low": tick_price,
                "close": tick_price,
                "volume": tick_volume,
            }
        else:
            # Update current bar
            self.current_bar["high"] = max(self.current_bar["high"], tick_price)
            self.current_bar["low"] = min(self.current_bar["low"], tick_price)
            self.current_bar["close"] = tick_price
            self.current_bar["volume"] += tick_volume

    def get_dataframe(self):
        return pd.DataFrame(self.bars)
