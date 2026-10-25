import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    """
    Computes backward-looking technical indicators.
    Strictly avoids future data leakage by observing information available at or prior to Close[t].
    """

    @staticmethod
    def compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        df = df.copy()

        # Log return on day t
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Lagged log returns
        df['Lag1'] = df['Log_Return'].shift(1)
        df['Lag2'] = df['Log_Return'].shift(2)
        df['Lag3'] = df['Log_Return'].shift(3)
        df['Lag5'] = df['Log_Return'].shift(5)

        # Moving Average Ratios (Close[t] relative to rolling mean)
        df['SMA_10_Ratio'] = (df['Close'] / df['Close'].rolling(window=10).mean()) - 1.0
        df['SMA_50_Ratio'] = (df['Close'] / df['Close'].rolling(window=50).mean()) - 1.0

        # Rolling Volatility
        df['Vol_10'] = df['Log_Return'].rolling(window=10).std()
        df['Vol_20'] = df['Log_Return'].rolling(window=20).std()

        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # High-Low relative spread
        df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']

        feature_cols = [
            'Lag1', 'Lag2', 'Lag3', 'Lag5',
            'SMA_10_Ratio', 'SMA_50_Ratio',
            'Vol_10', 'Vol_20',
            'RSI_14', 'HL_Spread'
        ]

        return df, feature_cols
