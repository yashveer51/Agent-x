import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Any, Tuple

class DataFetcher:
    """
    Data layer for retrieving, validating, and caching historical stock market data.
    Uses free sources (yfinance) without paid API keys.
    """

    def __init__(self, cache_dir: str = ".data_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.last_fetch_info: Dict[str, Any] = {}

    def fetch_symbol(
        self,
        symbol: str,
        start_date: str = "2018-01-01",
        end_date: str = "2024-12-31",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV data for a given ticker symbol.
        Tracks provenance (cache vs fresh download) and actual date boundaries.
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}.csv")

        if use_cache and os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            df = self.validate_data(df, symbol)
            self.last_fetch_info[symbol] = {
                "data_source": "cache",
                "cache_file": cache_file,
                "min_date": str(df.index.min().date()) if not df.empty else None,
                "max_date": str(df.index.max().date()) if not df.empty else None,
                "n_observations": len(df)
            }
            return df

        try:
            df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
        except Exception as e:
            raise RuntimeError(f"Failed download for '{symbol}': {e}") from e

        if df.empty:
            raise ValueError(f"No data retrieved for symbol '{symbol}'. Check ticker or date range.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = self.validate_data(df, symbol)

        if use_cache and not df.empty:
            df.to_csv(cache_file)

        self.last_fetch_info[symbol] = {
            "data_source": "fresh_download",
            "cache_file": cache_file if use_cache else None,
            "min_date": str(df.index.min().date()) if not df.empty else None,
            "max_date": str(df.index.max().date()) if not df.empty else None,
            "n_observations": len(df)
        }

        return df

    def validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate presence of OHLCV columns, proper DatetimeIndex sorting, non-empty rows,
        and price sanity checks (High >= Low, non-negative prices).
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

        # Sort ascending and remove exact duplicate timestamps
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Drop rows where essential prices are NaN or zero/negative
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        invalid_prices = (df['Open'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0) | (df['Close'] <= 0)
        if invalid_prices.any():
            df = df[~invalid_prices]

        # Sanity check: High >= Low
        invalid_hl = df['High'] < df['Low']
        if invalid_hl.any():
            df = df[~invalid_hl]

        if df.empty:
            raise ValueError(f"Data for symbol '{symbol}' contained no valid price observations after filtering.")

        return df
