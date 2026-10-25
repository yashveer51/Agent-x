import pytest
import pandas as pd
import numpy as np
from agent_x.validation import WalkForwardValidator
from agent_x.metrics import PerformanceMetrics

def test_walk_forward_split_boundaries():
    dates = pd.date_range("2020-01-01", periods=1000)
    df = pd.DataFrame({"Value": np.arange(1000)}, index=dates)

    validator = WalkForwardValidator(min_train_size=500, test_size=100, step_size=100, purge_gap=1)
    splits = list(validator.split(df))

    assert len(splits) >= 4

    for train_df, test_df, fold in splits:
        # Check training date is strictly prior to test date
        assert train_df.index.max() < test_df.index.min()
        # Check test size
        assert len(test_df) == 100

def test_financial_metrics_zero_volatility():
    zero_returns = pd.Series([0.0] * 100)
    metrics = PerformanceMetrics.calculate_financial_metrics(zero_returns)

    assert metrics['total_return'] == 0.0
    assert metrics['annualized_return'] == 0.0
    assert metrics['sharpe_ratio'] == 0.0
    assert metrics['max_drawdown'] == 0.0
