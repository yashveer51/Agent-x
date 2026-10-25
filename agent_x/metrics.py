import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

class PerformanceMetrics:
    """
    Computes financial metrics (Return, Sharpe, Sortino, Drawdown, Turnover)
    and statistical uncertainty intervals.
    """

    @staticmethod
    def calculate_financial_metrics(
        returns: pd.Series,
        costs: Optional[pd.Series] = None,
        wealth: Optional[pd.Series] = None,
        is_log_return: bool = False,
        initial_wealth: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculates comprehensive financial performance metrics.
        Includes initial portfolio wealth W_0 in peak calculation for Maximum Drawdown.

        Missing observations (NaN) and infinite values (inf/-inf) are explicitly dropped
        from returns, wealth, and costs series so they are not silently treated as zero-return periods.
        """
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns_clean) == 0:
            return {}

        if wealth is not None and len(wealth) > 0:
            wealth_clean = wealth.replace([np.inf, -np.inf], np.nan).dropna()
            if len(wealth_clean) == 0:
                total_return = 0.0
                max_drawdown = 0.0
            else:
                total_return = float((wealth_clean.iloc[-1] / initial_wealth) - 1.0)
                wealth_array = np.insert(wealth_clean.values, 0, initial_wealth)
                peaks = np.maximum.accumulate(wealth_array)
                drawdowns = (wealth_array - peaks) / peaks
                max_drawdown = float(np.min(drawdowns))
        else:
            if is_log_return:
                total_return = float(np.exp(returns_clean.sum()) - 1.0)
                cum_wealth = np.exp(returns_clean.cumsum().values)
            else:
                total_return = float(np.prod(1.0 + returns_clean) - 1.0)
                cum_wealth = np.cumprod(1.0 + returns_clean.values)

            wealth_array = np.insert(cum_wealth, 0, 1.0)
            peaks = np.maximum.accumulate(wealth_array)
            drawdowns = (wealth_array - peaks) / peaks
            max_drawdown = float(np.min(drawdowns))

        n_periods = len(returns_clean)
        if n_periods > 0 and total_return > -1.0:
            ann_return = float((1.0 + total_return) ** (252.0 / n_periods) - 1.0)
        else:
            ann_return = -1.0

        mean_ret = float(returns_clean.mean()) if n_periods > 0 else 0.0
        std_dev = float(returns_clean.std(ddof=1)) if n_periods > 1 else 0.0
        ann_vol = float(std_dev * np.sqrt(252))

        sharpe = float((mean_ret * 252) / (ann_vol + 1e-10)) if ann_vol > 1e-8 else 0.0

        downside_diff = np.minimum(returns_clean.values, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside_diff ** 2))) if n_periods > 0 else 0.0
        downside_vol = float(downside_dev * np.sqrt(252))
        sortino = float((mean_ret * 252) / (downside_vol + 1e-10)) if downside_vol > 1e-8 else 0.0

        if costs is not None:
            costs_clean = costs.replace([np.inf, -np.inf], np.nan).dropna()
            total_cost = float(costs_clean.sum())
        else:
            total_cost = 0.0

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
    def block_bootstrap_sharpe_ci(
        returns: pd.Series,
        n_bootstraps: int = 500,
        block_size: int = 20,
        random_seed: Optional[int] = 42
    ) -> Tuple[float, float]:
        """
        Reproducible block bootstrap confidence interval for Sharpe ratio.
        Returns (np.nan, np.nan) if sample size is insufficient.

        Missing observations (NaN) and infinite values (inf/-inf) are explicitly dropped
        prior to bootstrapping.
        """
        clean_rets_series = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(clean_rets_series) < block_size * 2:
            return (np.nan, np.nan)

        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
        else:
            rng = np.random.RandomState()

        clean_rets = clean_rets_series.values
        n = len(clean_rets)
        sharpes = []

        for _ in range(n_bootstraps):
            indices = []
            while len(indices) < n:
                start = rng.randint(0, n - block_size + 1)
                indices.extend(range(start, start + block_size))
            sample = clean_rets[indices[:n]]
            std_dev = sample.std()
            vol = std_dev * np.sqrt(252)
            if vol < 1e-8:
                sharpes.append(0.0)
            else:
                sharpes.append((sample.mean() * 252) / vol)

        lower_ci = float(np.percentile(sharpes, 2.5))
        upper_ci = float(np.percentile(sharpes, 97.5))
        return (lower_ci, upper_ci)
