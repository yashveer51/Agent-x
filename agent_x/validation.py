import pandas as pd
import numpy as np
from typing import Generator, Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

class WalkForwardValidator:
    """
    Chronological Expanding-Window or Rolling-Window Walk-Forward Cross Validation.
    Prevents temporal data leakage and applies target-horizon-aware purging/embargo gaps.
    """

    def __init__(
        self,
        min_train_size: int = 500,
        test_size: int = 125,
        step_size: int = 125,
        purge_gap: int = 1,
        holdout_size: int = 0,
        scale_features: bool = False
    ):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.purge_gap = purge_gap
        self.holdout_size = holdout_size
        self.scale_features = scale_features

    def split(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        """
        Yields (train_df, test_df, fold_number).
        Ensures training observations strictly precede test observations with purge_gap isolation.
        Reserves final holdout_size observations if specified.
        """
        n_samples = len(df) - self.holdout_size
        if n_samples <= 0:
            raise ValueError("Dataset length is insufficient for requested holdout_size.")

        fold = 1
        start_idx = 0

        while True:
            train_end_idx = self.min_train_size + (fold - 1) * self.step_size
            test_start_idx = train_end_idx + self.purge_gap
            test_end_idx = test_start_idx + self.test_size

            if test_end_idx > n_samples:
                break

            train_df = df.iloc[start_idx:train_end_idx].copy()
            test_df = df.iloc[test_start_idx:test_end_idx].copy()

            # Verify no index overlap
            assert train_df.index.max() < test_df.index.min(), "Leakage detected: train timestamp overlaps test timestamp!"

            yield train_df, test_df, fold
            fold += 1

    def fit_transform_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Fits StandardScaler strictly on training observations and transforms train and test features.
        Prevents test set distribution leakage into feature scaling.
        """
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols])
        X_test = scaler.transform(test_df[feature_cols])
        return X_train, X_test, scaler

    def get_holdout(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Retrieves the isolated final holdout dataset if holdout_size > 0.
        """
        if self.holdout_size <= 0 or len(df) <= self.holdout_size:
            return None
        return df.iloc[-self.holdout_size:].copy()
