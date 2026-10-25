import pytest
import pandas as pd
import numpy as np
from agent_x.backtest import BacktestEngine

def test_backtest_costs_applied_on_turnover_only():
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        "Next_Log_Return": [0.01, 0.02, -0.01, 0.01, 0.02]
    }, index=dates)

    # Signal: Hold position for 3 days (1, 1, 1), then Cash (0, 0)
    signals = pd.Series([1, 1, 1, 0, 0], index=dates)

    engine = BacktestEngine(transaction_cost_pct=0.0010)
    res = engine.run_backtest(df, signals)

    # Day 1: Entry -> Turnover = 1.0 -> Cost = 0.0010
    assert res.iloc[0]['Position_Change'] == 1.0
    assert res.iloc[0]['Cost'] == 0.0010

    # Day 2: Continuous hold -> Turnover = 0.0 -> Cost = 0.0000
    assert res.iloc[1]['Position_Change'] == 0.0
    assert res.iloc[1]['Cost'] == 0.0000

    # Day 3: Continuous hold -> Turnover = 0.0 -> Cost = 0.0000
    assert res.iloc[2]['Position_Change'] == 0.0
    assert res.iloc[2]['Cost'] == 0.0000

    # Day 4: Exit to Cash -> Turnover = 1.0 -> Cost = 0.0010
    assert res.iloc[3]['Position_Change'] == 1.0
    assert res.iloc[3]['Cost'] == 0.0010

    # Day 5: Continuous cash -> Turnover = 0.0 -> Cost = 0.0000
    assert res.iloc[4]['Position_Change'] == 0.0
    assert res.iloc[4]['Cost'] == 0.0000

def test_missing_execution_return_resets_exposure_for_costs():
    """
    Test Phase A: When execution return is missing (valid_mask is False),
    prev_exposure must reset to 0 so next position entry incurs full turnover cost.
    """
    dates = pd.date_range("2023-01-01", periods=4)
    df = pd.DataFrame({
        "Next_Executable_Return": [0.01, np.nan, 0.02, 0.01]
    }, index=dates)

    # Day 0: Signal 1 (Enter Long) -> Turnover 1.0, Cost 0.0010
    # Day 1: Signal 1 (Missing Return) -> Skipped, position reset to 0.0
    # Day 2: Signal 1 (Enter Long after missing return row) -> Turnover 1.0, Cost 0.0010
    # Day 3: Signal 1 (Hold Long) -> Turnover 0.0, Cost 0.0000
    signals = pd.Series([1, 1, 1, 1], index=dates)

    engine = BacktestEngine(transaction_cost_pct=0.0010)
    res = engine.run_backtest(df, signals)

    assert abs(res.iloc[0]['Cost'] - 0.0010) < 1e-7
    assert res.iloc[1]['Cost'] == 0.0
    assert abs(res.iloc[2]['Position_Change'] - 1.0) < 1e-7
    assert abs(res.iloc[2]['Cost'] - 0.0010) < 1e-7
    assert res.iloc[3]['Position_Change'] == 0.0
    assert res.iloc[3]['Cost'] == 0.0
