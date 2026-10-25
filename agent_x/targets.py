import pandas as pd
import numpy as np

class TargetBuilder:
    """
    Defines targets with clear timeline boundaries.

    Execution Conventions Supported:
    1. Open-to-Close ('open_to_close'):
       - Features generated at or prior to day t Close.
       - Signal decision made after day t Close.
       - Position entered at day t+1 Open, exited at day t+1 Close.
       - Executable return = Close[t+1] / Open[t+1] - 1 (or log(Close[t+1] / Open[t+1])).
       - Excludes overnight move (Open[t+1] / Close[t]).
    2. Close-to-Close ('close_to_close'):
       - Return measured over day t+1 (Close[t+1] / Close[t] - 1).
    """

    @staticmethod
    def add_directional_target(
        df: pd.DataFrame,
        execution_convention: str = "open_to_close",
        use_log_return: bool = False
    ) -> pd.DataFrame:
        """
        Calculates future return for execution on day t+1 and creates binary target label.
        Excludes observations with missing future execution prices / labels.

        Default `use_log_return` is False to match simple return convention across
        BacktestEngine and PerformanceMetrics.

        Explicitly computes both `Next_Simple_Return` and `Next_Log_Return`, and sets
        `Next_Executable_Return` according to `use_log_return`.
        """
        df = df.copy()

        if execution_convention == "open_to_close" and 'Open' in df.columns:
            # Position entered at Open[t+1] and exited at Close[t+1]
            next_open = df['Open'].shift(-1)
            next_close = df['Close'].shift(-1)

            simple_ret = (next_close / next_open) - 1.0
            log_ret = np.log(next_close / next_open)

        elif execution_convention == "close_to_close" or 'Open' not in df.columns:
            # Position held Close[t] to Close[t+1]
            simple_ret = (df['Close'].shift(-1) / df['Close']) - 1.0
            log_ret = np.log(df['Close'].shift(-1) / df['Close'])
        else:
            raise ValueError(f"Unknown execution convention: {execution_convention}")

        df['Next_Simple_Return'] = simple_ret
        df['Next_Log_Return'] = log_ret

        if use_log_return:
            df['Next_Executable_Return'] = df['Next_Log_Return']
        else:
            df['Next_Executable_Return'] = df['Next_Simple_Return']

        # Binary Direction Target: 1 if Next_Executable_Return > 0 else 0
        # Exclude observations where future label is NaN
        valid_mask = df['Next_Executable_Return'].notna()
        df['Target'] = np.nan
        df.loc[valid_mask, 'Target'] = (df.loc[valid_mask, 'Next_Executable_Return'] > 0).astype(int)

        return df
