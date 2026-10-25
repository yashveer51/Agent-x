import pandas as pd
import numpy as np
from typing import Generator, Tuple, List, Dict, Any

class WalkForwardValidator:
    """
    Chronological Expanding-Window or Rolling-Window Walk-Forward Cross Validation.
    Prevents temporal data leakage and applies time gaps where necessary.
    """

    def __init__(
        self,
        min_train_size: int = 500,
        test_size: int = 125,
        step_size: int = 125,
        purge_gap: int = 1
    ):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.purge_gap = purge_gap

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        n_samples = len(df)
        fold = 1

        start_idx = 0
        while True:
            train_end_idx = self.min_train_size + (fold - 1) * self.step_size
            test_start_idx = train_end_idx + self.purge_gap
            test_end_idx = test_start_idx + self.test_size

            if test_end_idx > n_samples:
                break

            train_df = df.iloc[start_idx:train_end_idx]
            test_df = df.iloc[test_start_idx:test_end_idx]

            yield train_df, test_df, fold
            fold += 1
