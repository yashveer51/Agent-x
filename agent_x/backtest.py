import pandas as pd
import numpy as np

class BacktestEngine:
    """
    Deterministic backtest simulation engine.
    Calculates portfolio returns, turnover, and applies transaction costs on position changes.
    """

    def __init__(self, transaction_cost_pct: float = 0.0010):
        self.transaction_cost_pct = transaction_cost_pct

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        return_col: str = "Next_Log_Return"
    ) -> pd.DataFrame:
        """
        Run backtest on test dataframe with signals.
        - Position: 1 for Long, 0 for Cash
        - Turnover: |Position[t] - Position[t-1]|
        - Cost: Turnover * transaction_cost_pct
        - Net Return: Position * Next_Log_Return - Cost
        """
        res = df.copy()
        res['Signal'] = signals.values

        # Position held for day t+1
        res['Position'] = res['Signal'].astype(float)

        # Calculate position change (turnover)
        res['Position_Change'] = res['Position'].diff().abs()
        # Initial position entry cost
        res['Position_Change'] = res['Position_Change'].fillna(res['Position'])

        # Transaction cost penalty
        res['Cost'] = res['Position_Change'] * self.transaction_cost_pct

        # Strategy Daily Return
        res['Strategy_Return'] = (res['Position'] * res[return_col]) - res['Cost']

        # Cumulative Growth
        res['Cum_Market'] = np.exp(res[return_col].cumsum())
        res['Cum_Strategy'] = np.exp(res['Strategy_Return'].cumsum())

        return res
