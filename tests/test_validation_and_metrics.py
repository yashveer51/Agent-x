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

def test_sharpe_and_sortino_formulas():
    # Hand-crafted example:
    # returns = [0.01, -0.01, 0.02, -0.005]
    # mean = (0.01 - 0.01 + 0.02 - 0.005) / 4 = 0.015 / 4 = 0.00375
    # std (ddof=1) of [0.01, -0.01, 0.02, -0.005]
    # downside diff = [0, -0.01, 0, -0.005] -> sq = [0, 0.0001, 0, 0.000025] -> mean = 0.000125 / 4 = 0.00003125
    # downside dev = sqrt(0.00003125) = 0.00559016994
    returns = pd.Series([0.01, -0.01, 0.02, -0.005])
    metrics = PerformanceMetrics.calculate_financial_metrics(returns)

    mean_ret = returns.mean() # 0.00375
    std_ret = returns.std(ddof=1)
    ann_vol = std_ret * np.sqrt(252)
    expected_sharpe = (mean_ret * 252) / ann_vol

    downside_sq_mean = np.mean([0.0, (-0.01)**2, 0.0, (-0.005)**2])
    expected_downside_vol = np.sqrt(downside_sq_mean) * np.sqrt(252)
    expected_sortino = (mean_ret * 252) / expected_downside_vol

    assert abs(metrics['sharpe_ratio'] - expected_sharpe) < 1e-5
    assert abs(metrics['sortino_ratio'] - expected_sortino) < 1e-5
