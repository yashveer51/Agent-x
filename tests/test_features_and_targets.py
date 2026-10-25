import pytest
import pandas as pd
import numpy as np
from agent_x.features import FeatureEngineer
from agent_x.targets import TargetBuilder

def test_feature_engineering_no_future_leakage():
    # Synthetic dataset
    dates = pd.date_range("2023-01-01", periods=100)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100))
    df = pd.DataFrame({
        "Open": prices,
        "High": prices + 1,
        "Low": prices - 1,
        "Close": prices,
        "Volume": 10000
    }, index=dates)

    df_feat, feature_cols = FeatureEngineer.compute_features(df)

    # Modify prices at index 50 onwards
    df_mod = df.copy()
    df_mod.iloc[50:, df_mod.columns.get_loc("Close")] += 1000.0
    df_feat_mod, _ = FeatureEngineer.compute_features(df_mod)

    # Features at t < 50 MUST be IDENTICAL regardless of future prices at t >= 50
    pdt_orig = df_feat.iloc[:49][feature_cols].dropna()
    pdt_mod = df_feat_mod.iloc[:49][feature_cols].dropna()

    pd.testing.assert_frame_equal(pdt_orig, pdt_mod)

def test_target_alignment():
    dates = pd.date_range("2023-01-01", periods=5)
    prices = [100.0, 105.0, 102.0, 108.0, 110.0]
    df = pd.DataFrame({"Close": prices}, index=dates)

    df_target = TargetBuilder.add_directional_target(df)

    # Day 0: Close = 100.0, Day 1: Close = 105.0 -> Log return = ln(105/100) > 0 -> Target = 1
    assert df_target.iloc[0]['Target'] == 1
    # Day 1: Close = 105.0, Day 2: Close = 102.0 -> Log return = ln(102/105) < 0 -> Target = 0
    assert df_target.iloc[1]['Target'] == 0
