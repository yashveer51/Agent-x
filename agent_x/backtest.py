import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

class BacktestEngine:
    """
    Deterministic wealth-based backtest simulation engine.
    Calculates exact wealth trajectory, position changes, turnover, and transaction costs.

    Execution Timeline & Holding Models:
    1. 'open_to_close' (default intraday session model):
       - Position entered at Open[t+1] and exited at Close[t+1].
       - Executable return R_{exec, t+1} = Close[t+1] / Open[t+1] - 1.
       - Overnight gap (Open[t+1] / Close[t] - 1) is NOT credited to the position.
       - Since position is closed at Close[t+1] every day, each active intraday trade
         incurs roundtrip transaction costs (entry turn at Open[t+1] and exit turn at Close[t+1],
         total turnover = 2 * |target_exposure| per session).
    2. 'close_to_close' (continuous holding model):
       - Position entered at Close[t] (or Open[t+1]) and held continuously across sessions.
       - Transaction costs are charged only on exposure changes across sessions
         (turnover = |target_exposure_t - target_exposure_{t-1}|).
    """

    def __init__(
        self,
        transaction_cost_pct: float = 0.0010,
        slippage_pct: float = 0.0000,
        fixed_commission: float = 0.0,
        initial_wealth: float = 1.0,
        use_log_return: bool = False,
        holding_model: str = "open_to_close"
    ):
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.fixed_commission = fixed_commission
        self.initial_wealth = initial_wealth
        self.use_log_return = use_log_return
        if holding_model not in ("open_to_close", "close_to_close"):
            raise ValueError(f"Unknown holding_model: {holding_model}. Must be 'open_to_close' or 'close_to_close'.")
        self.holding_model = holding_model

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        return_col: str = "Next_Executable_Return"
    ) -> pd.DataFrame:
        """
        Run wealth-consistent backtest on dataframe with signals.
        """
        res = df.copy()

        # Align signals
        res['Signal'] = signals.reindex(res.index).fillna(0.0).values
        res['Target_Exposure'] = res['Signal'].astype(float)

        # Ensure return column exists if Open and Close available
        if return_col not in res.columns:
            if self.use_log_return and 'Next_Log_Return' in res.columns:
                res[return_col] = res['Next_Log_Return']
            elif not self.use_log_return and 'Next_Simple_Return' in res.columns:
                res[return_col] = res['Next_Simple_Return']
            elif 'Next_Executable_Return' in res.columns:
                res[return_col] = res['Next_Executable_Return']
            elif 'Open' in res.columns and 'Close' in res.columns:
                next_open = res['Open'].shift(-1)
                next_close = res['Close'].shift(-1)
                if self.use_log_return:
                    res[return_col] = np.log(next_close / next_open)
                else:
                    res[return_col] = (next_close / next_open) - 1.0
            else:
                raise ValueError(f"Return column '{return_col}' not found in dataframe and cannot be computed.")

        valid_mask = res[return_col].notna()

        n = len(res)
        wealth = np.zeros(n)
        gross_wealth = np.zeros(n)
        portfolio_returns = np.zeros(n)
        gross_returns = np.zeros(n)
        costs = np.zeros(n)
        position_changes = np.zeros(n)
        trades = np.zeros(n, dtype=int)

        current_wealth = self.initial_wealth
        current_gross_wealth = self.initial_wealth
        prev_exposure = 0.0

        for i in range(n):
            if not valid_mask.iloc[i]:
                wealth[i] = current_wealth
                gross_wealth[i] = current_gross_wealth
                portfolio_returns[i] = 0.0
                gross_returns[i] = 0.0
                costs[i] = 0.0
                position_changes[i] = 0.0
                prev_exposure = 0.0
                continue

            target_exposure = res['Target_Exposure'].iloc[i]
            r_exec = res[return_col].iloc[i]

            if self.use_log_return:
                simple_r_exec = float(np.exp(r_exec) - 1.0)
            else:
                simple_r_exec = float(r_exec)

            # Turnover and position change calculation according to holding model
            if self.holding_model == "open_to_close":
                # Intraday session trade: entry turn at Open + exit turn at Close = 2 * |exposure|
                pos_change = 2.0 * abs(target_exposure)
            else:
                # Continuous position holding across sessions
                pos_change = abs(target_exposure - prev_exposure)

            position_changes[i] = pos_change

            if pos_change > 1e-6:
                trades[i] = 1

            total_cost_pct = self.transaction_cost_pct + self.slippage_pct

            # Cost in portfolio wealth units
            trade_cost = (pos_change * total_cost_pct * current_wealth) + (self.fixed_commission if pos_change > 1e-6 else 0.0)
            costs[i] = trade_cost / current_wealth if current_wealth > 0 else 0.0

            gross_r = target_exposure * simple_r_exec
            gross_returns[i] = gross_r
            current_gross_wealth = current_gross_wealth * (1.0 + gross_r)
            gross_wealth[i] = current_gross_wealth

            net_wealth_start = max(0.0, current_wealth - trade_cost)
            current_wealth = net_wealth_start * (1.0 + gross_r)
            wealth[i] = current_wealth

            prev_wealth = wealth[i-1] if i > 0 else self.initial_wealth
            portfolio_returns[i] = (current_wealth - prev_wealth) / prev_wealth if prev_wealth > 0 else 0.0

            prev_exposure = target_exposure

        res['Position'] = res['Target_Exposure']
        res['Position_Change'] = position_changes
        res['Trade'] = trades
        res['Cost'] = costs
        res['Gross_Return'] = gross_returns
        res['Strategy_Return'] = portfolio_returns
        res['Wealth'] = wealth
        res['Gross_Wealth'] = gross_wealth

        res['Cum_Strategy'] = wealth / self.initial_wealth
        res['Cum_Market'] = gross_wealth / self.initial_wealth

        return res
