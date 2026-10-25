# Agent X Quantitative Research Report: Verified_Agent_X_Audit

**Generated Date**: 2026-10-25 04:42:39
**Commit SHA**: `6780e98c242dd73da8f5063344024cf79b4ab331`

## Executive Summary

Agent X evaluates machine learning stock predictions using chronological walk-forward validation and realistic transaction costs under Open-to-Close execution timing.

## Out-of-Sample Performance Summary

| Asset | Model | Gross Return | Net Return | Benchmark Net (Executable) | Benchmark Frictionless | Sharpe Ratio | Sharpe 95% CI | Max Drawdown | Total Costs |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | random_forest | 76.65% | 34.43% | 113.70% | 113.92% | 0.48 | [-0.65, 1.42] | -24.81% | 0.2730 |
| AAPL | logistic_regression | 100.13% | 58.20% | 113.70% | 113.92% | 0.72 | [-0.45, 1.67] | -23.01% | 0.2350 |
| AAPL | majority_class | 113.92% | 113.70% | 113.70% | 113.92% | 0.97 | [-0.23, 2.07] | -29.84% | 0.0010 |
| MSFT | random_forest | 20.11% | -4.48% | 50.41% | 50.56% | 0.04 | [-0.89, 0.84] | -36.12% | 0.2290 |
| MSFT | logistic_regression | 52.29% | 19.66% | 50.41% | 50.56% | 0.36 | [-0.69, 1.20] | -42.29% | 0.2410 |
| MSFT | majority_class | 50.56% | 50.41% | 50.41% | 50.56% | 0.60 | [-0.29, 1.42] | -28.77% | 0.0010 |
| SPY | random_forest | 31.01% | 13.77% | 24.93% | 25.05% | 0.33 | [-0.70, 1.20] | -18.45% | 0.1410 |
| SPY | logistic_regression | 18.69% | 4.95% | 24.93% | 25.05% | 0.17 | [-0.73, 0.98] | -17.17% | 0.1230 |
| SPY | majority_class | 25.05% | 24.93% | 24.93% | 25.05% | 0.48 | [-0.44, 1.32] | -14.35% | 0.0010 |

## Final Isolated Holdout Performance Summary

| Asset | Model | Holdout Net Return | Holdout Benchmark Net | Holdout Sharpe Ratio | Holdout Max Drawdown |
|---|---|---|---|---|---|
| AAPL | random_forest | 34.46% | 35.26% | 1.81 | -10.05% |
| AAPL | logistic_regression | 35.08% | 35.26% | 1.75 | -10.12% |
| AAPL | majority_class | 35.26% | 35.26% | 1.74 | -10.98% |
| MSFT | random_forest | -6.59% | -2.63% | -0.35 | -18.28% |
| MSFT | logistic_regression | -1.67% | -2.63% | -0.02 | -18.72% |
| MSFT | majority_class | -2.63% | -2.63% | -0.08 | -20.27% |
| SPY | random_forest | -2.10% | 1.30% | -0.19 | -9.40% |
| SPY | logistic_regression | -2.58% | 1.30% | -0.23 | -9.06% |
| SPY | majority_class | 1.30% | 1.30% | 0.18 | -8.46% |

## Audit & Research Conclusion

**Classification**: **No demonstrated edge after transaction costs**: Friction costs, realistic execution timing, and market noise eliminate directional predictive advantages.
