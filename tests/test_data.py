import pytest
import pandas as pd
import numpy as np
from agent_x.data import DataFetcher

def test_validate_data_empty():
    fetcher = DataFetcher()
    with pytest.raises(ValueError, match="No data retrieved"):
        fetcher.validate_data(pd.DataFrame(), "TEST")

def test_validate_data_missing_cols():
    fetcher = DataFetcher()
    df = pd.DataFrame({"Open": [10, 11], "Close": [10.5, 11.5]})
    with pytest.raises(ValueError, match="missing required columns"):
        fetcher.validate_data(df, "TEST")

def test_validate_data_clean():
    fetcher = DataFetcher()
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({
        "Open": [10.0, 11.0, 12.0],
        "High": [10.5, 11.5, 12.5],
        "Low": [9.5, 10.5, 11.5],
        "Close": [10.2, 11.2, 12.2],
        "Volume": [1000, 1100, 1200]
    }, index=dates)

    clean_df = fetcher.validate_data(df, "TEST")
    assert len(clean_df) == 3
    assert isinstance(clean_df.index, pd.DatetimeIndex)
