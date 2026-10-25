import pytest
import pandas as pd
import numpy as np
from agent_x.targets import TargetBuilder
from agent_x.backtest import BacktestEngine
from agent_x.metrics import PerformanceMetrics
from agent_x.validation import WalkForwardValidator

def test_overnight_gap_not_credited_to_open_entry():
    """
    Test Fix A: Large overnight gap followed by flat intraday price.
    Strategy entering at Open[t+1] MUST NOT receive the overnight gap return.
    """
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({
        "Open":  [100.0, 150.0, 150.0],  # Day 1 -> Day 2 has +50% overnight gap
        "High":  [100.0, 150.0, 150.0],
        "Low":   [100.0, 150.0, 150.0],
        "Close": [100.0, 150.0, 150.0]   # Flat intraday price on Day 2 (Close[1] / Open[1] = 1.0)
    }, index=dates)

    # Signal 1 at t=0 means long position entered at Open[1] (150.0) and exited at Close[1] (150.0)
    df_target = TargetBuilder.add_directional_target(df, execution_convention="open_to_close", use_log_return=False)

    # Executable return on Day 0 for trade executed on Day 1 is 150.0 / 150.0 - 1 = 0.0
    assert abs(df_target.iloc[0]['Next_Executable_Return']) < 1e-7

    signals = pd.Series([1, 0, 0], index=dates)
    engine = BacktestEngine(transaction_cost_pct=0.0, use_log_return=False)
    res = engine.run_backtest(df_target, signals)

    # Strategy Gross Return on Day 0 must be 0.0, NOT 0.50!
    assert abs(res.iloc[0]['Gross_Return']) < 1e-7
    assert abs(res.iloc[0]['Wealth'] - 1.0) < 1e-7

def test_missing_future_label_excluded_and_not_zero():
    """
    Test Fix C: The last row (missing future label) must have Target = NaN, not 0.
    """
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({
        "Open": [10.0, 11.0, 12.0],
        "Close": [10.5, 11.5, 12.5]
    }, index=dates)

    df_target = TargetBuilder.add_directional_target(df)
    assert pd.isna(df_target.iloc[-1]['Target'])
    assert df_target.iloc[-1]['Target'] != 0

def test_wealth_accounting_and_costs():
    """
    Test Fix B: Detailed wealth accounting through Cash -> Entry -> Hold -> Exit -> Cash.
    """
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        "Next_Executable_Return": [0.05, 0.02, -0.01, 0.04, 0.00]
    }, index=dates)

    # Signals: Day 0: Cash (0), Day 1: Enter Long (1), Day 2: Hold Long (1), Day 3: Exit to Cash (0), Day 4: Cash (0)
    signals = pd.Series([0, 1, 1, 0, 0], index=dates)

    engine = BacktestEngine(transaction_cost_pct=0.0010, initial_wealth=100.0, use_log_return=False)
    res = engine.run_backtest(df, signals)

    # Day 0: Cash (0) -> No cost, wealth = 100.0
    assert res.iloc[0]['Cost'] == 0.0
    assert abs(res.iloc[0]['Wealth'] - 100.0) < 1e-7

    # Day 1: Entry (0 -> 1) -> Cost = 0.0010 * 100 = 0.10. Net wealth start = 99.90. Return = +2%. End Wealth = 99.90 * 1.02 = 101.898
    assert abs(res.iloc[1]['Position_Change'] - 1.0) < 1e-7
    assert abs(res.iloc[1]['Cost'] - 0.0010) < 1e-7
    assert abs(res.iloc[1]['Wealth'] - 101.898) < 1e-5

    # Day 2: Hold (1 -> 1) -> No cost. Return = -1%. End Wealth = 101.898 * 0.99 = 100.87902
    assert abs(res.iloc[2]['Position_Change']) < 1e-7
    assert abs(res.iloc[2]['Cost']) < 1e-7
    assert abs(res.iloc[2]['Wealth'] - (101.898 * 0.99)) < 1e-5

    # Day 3: Exit (1 -> 0) -> Cost = 0.0010 * 100.87902. Target Exposure = 0 -> Gross return = 0%.
    assert abs(res.iloc[3]['Position_Change'] - 1.0) < 1e-7
    assert abs(res.iloc[3]['Cost'] - 0.0010) < 1e-7

def test_max_drawdown_includes_initial_wealth():
    """
    Test Fix E: Max drawdown calculation includes initial wealth W_0 = 1.0.
    Immediate loss from 1.0 down to 0.8 must show -20% drawdown.
    """
    returns = pd.Series([-0.20, 0.05, 0.05])
    metrics = PerformanceMetrics.calculate_financial_metrics(returns, initial_wealth=1.0)
    assert abs(metrics['max_drawdown'] - (-0.20)) < 1e-5

def test_bootstrap_sharpe_ci_reproducible():
    """
    Test Fix E: Reproducible block bootstrap Sharpe CI using random_seed.
    """
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.01, 100))
    ci1 = PerformanceMetrics.block_bootstrap_sharpe_ci(returns, random_seed=42)
    ci2 = PerformanceMetrics.block_bootstrap_sharpe_ci(returns, random_seed=42)
    assert ci1 == ci2
    assert not np.isnan(ci1[0])

def test_investable_benchmark_costs_deducted():
    """
    Test Phase C: Verify that an investable benchmark deducts transaction costs on entry.
    """
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame({
        "Next_Executable_Return": [0.05, 0.02, 0.01]
    }, index=dates)

    signals = pd.Series([1.0, 1.0, 1.0], index=dates)

    engine_investable = BacktestEngine(transaction_cost_pct=0.0010, initial_wealth=1.0)
    engine_frictionless = BacktestEngine(transaction_cost_pct=0.0000, initial_wealth=1.0)

    res_inv = engine_investable.run_backtest(df, signals)
    res_fric = engine_frictionless.run_backtest(df, signals)

    # Investable benchmark must incur cost on day 0
    assert abs(res_inv.iloc[0]['Cost'] - 0.0010) < 1e-7
    assert res_fric.iloc[0]['Cost'] == 0.0
    assert res_inv.iloc[-1]['Wealth'] < res_fric.iloc[-1]['Wealth']

def test_holdout_set_retrieval_isolation():
    """
    Test Phase D: Verify holdout set is strictly isolated from walk-forward folds.
    """
    dates = pd.date_range("2020-01-01", periods=1000)
    df = pd.DataFrame({"Value": np.arange(1000)}, index=dates)

    validator = WalkForwardValidator(min_train_size=500, test_size=100, step_size=100, purge_gap=1, holdout_size=150)
    holdout = validator.get_holdout(df)

    assert len(holdout) == 150
    assert holdout.index.min() == df.index[850]

    for train_df, test_df, fold in validator.split(df):
        assert test_df.index.max() < holdout.index.min()
