import pandas as pd
import numpy as np
from typing import Dict, Any

class PerformanceMetrics:
    """
    Computes financial metrics (Return, Sharpe, Sortino, Drawdown, Turnover)
    and classification metrics (Accuracy, Precision, Recall).
    """

    @staticmethod
    def calculate_financial_metrics(log_returns: pd.Series, costs: pd.Series = None) -> Dict[str, Any]:
        if len(log_returns) == 0:
            return {}

        total_return = float(np.exp(log_returns.sum()) - 1.0)
        ann_return = float(np.exp(log_returns.mean() * 252) - 1.0)
        ann_vol = float(log_returns.std() * np.sqrt(252))

        # Sharpe ratio
        sharpe = float((log_returns.mean() * 252) / (ann_vol + 1e-10))

        # Downside volatility for Sortino
        negative_returns = log_returns[log_returns < 0]
        downside_vol = float(negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 else 1e-10
        sortino = float((log_returns.mean() * 252) / (downside_vol + 1e-10))

        # Max Drawdown
        cum_ret = np.exp(log_returns.cumsum())
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_drawdown = float(drawdown.min())

        total_cost = float(costs.sum()) if costs is not None else 0.0

        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'total_costs': total_cost
        }

    @staticmethod
    def block_bootstrap_sharpe_ci(log_returns: pd.Series, n_bootstraps: int = 500, block_size: int = 20) -> tuple[float, float]:
        """
        Block bootstrap confidence interval for Sharpe ratio to respect time dependence.
        """
        if len(log_returns) < block_size * 2:
            return (0.0, 0.0)

        n = len(log_returns)
        sharpes = []

        for _ in range(n_bootstraps):
            indices = []
            while len(indices) < n:
                start = np.random.randint(0, n - block_size + 1)
                indices.extend(range(start, start + block_size))
            sample = log_returns.iloc[indices[:n]]
            vol = sample.std() * np.sqrt(252)
            sharpe = (sample.mean() * 252) / (vol + 1e-10)
            sharpes.append(sharpe)

        lower_ci = float(np.percentile(sharpes, 2.5))
        upper_ci = float(np.percentile(sharpes, 97.5))
        return (lower_ci, upper_ci)
