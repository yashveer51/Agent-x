import pandas as pd
import numpy as np

class TargetBuilder:
    """
    Defines targets with clear timeline boundaries.

    Execution Convention:
    - Features generated at or prior to day t Close.
    - Signal decision made after day t Close.
    - Position entered/held for day t+1.
    - Return measured over day t+1 (Close[t+1] / Close[t] - 1 or Next_Log_Return).
    """

    @staticmethod
    def add_directional_target(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Next day log return: Close[t+1] / Close[t]
        df['Next_Log_Return'] = np.log(df['Close'].shift(-1) / df['Close'])

        # Binary Direction Target: 1 if Next_Log_Return > 0 else 0
        df['Target'] = (df['Next_Log_Return'] > 0).astype(int)

        return df
