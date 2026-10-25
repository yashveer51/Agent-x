import os
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, Any

class DataFetcher:
    """
    Data layer for retrieving, validating, and caching historical stock market data.
    Uses free sources (yfinance) without paid API keys.
    """

    def __init__(self, cache_dir: str = ".data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def fetch_symbol(
        self,
        symbol: str,
        start_date: str = "2018-01-01",
        end_date: str = "2023-12-31",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV data for a given ticker symbol.
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}.csv")

        if use_cache and os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return self.validate_data(df, symbol)

        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = self.validate_data(df, symbol)

        if use_cache and not df.empty:
            df.to_csv(cache_file)

        return df

    def validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate presence of OHLCV columns, proper DatetimeIndex sorting, and non-empty rows.
        """
        if df.empty:
            raise ValueError(f"No data retrieved for symbol '{symbol}'. Check ticker or date range.")

        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Data for '{symbol}' missing required columns: {missing}")

        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Ensure index is sorted ascending and duplicates removed
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Drop rows where all prices are zero or NaN
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        return df
