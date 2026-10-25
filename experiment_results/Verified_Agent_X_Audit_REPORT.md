# Agent X Quantitative Research Report: Verified_Agent_X_Audit

**Generated Date**: 2026-10-25 10:43:47
**Commit SHA**: `215ca1cd0ebbc3b789c7b87d7535f49a4a7bdf8e`

## Executive Summary

Agent X evaluates machine learning stock predictions using chronological walk-forward validation and realistic transaction costs under Open-to-Close execution timing.

## Out-of-Sample Performance Summary

| Asset | Model | Gross Return | Net Return | Benchmark Net (Executable) | Benchmark Frictionless | Sharpe Ratio | Sharpe 95% CI | Max Drawdown | Total Costs |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | random_forest | 76.65% | -55.44% | -62.89% | 113.92% | -0.88 | [-2.03, -0.00] | -59.73% | 1.3760 |
| AAPL | logistic_regression | 100.13% | -48.91% | -62.89% | 113.92% | -0.79 | [-1.99, 0.18] | -57.53% | 1.3640 |
| AAPL | majority_class | 113.92% | -62.89% | -62.89% | 113.92% | -0.96 | [-2.18, 0.07] | -70.23% | 1.7500 |
| MSFT | random_forest | 18.63% | -62.33% | -73.88% | 50.56% | -1.26 | [-2.30, -0.46] | -65.30% | 1.1460 |
| MSFT | logistic_regression | 52.29% | -51.64% | -73.88% | 50.56% | -0.95 | [-2.10, -0.11] | -65.32% | 1.1460 |
| MSFT | majority_class | 50.56% | -73.88% | -73.88% | 50.56% | -1.45 | [-2.41, -0.77] | -76.31% | 1.7500 |
| SPY | random_forest | 23.63% | -70.52% | -77.72% | 28.47% | -2.42 | [-3.49, -1.74] | -72.14% | 1.4320 |
| SPY | logistic_regression | 23.08% | -74.13% | -77.72% | 28.47% | -2.69 | [-3.77, -2.02] | -76.61% | 1.5580 |
| SPY | majority_class | 28.47% | -77.72% | -77.72% | 28.47% | -2.62 | [-3.75, -1.99] | -79.76% | 1.7500 |

## Final Isolated Holdout Performance Summary

| Asset | Model | Holdout Net Return | Holdout Benchmark Net | Holdout Sharpe Ratio | Holdout Max Drawdown |
|---|---|---|---|---|---|
| AAPL | random_forest | -12.14% | -17.92% | -0.67 | -24.36% |
| AAPL | logistic_regression | -16.54% | -17.92% | -0.91 | -28.51% |
| AAPL | majority_class | -17.92% | -17.92% | -0.99 | -29.69% |
| MSFT | random_forest | -36.82% | -40.91% | -2.83 | -38.42% |
| MSFT | logistic_regression | -38.14% | -40.91% | -2.88 | -41.22% |
| MSFT | majority_class | -40.91% | -40.91% | -3.12 | -43.81% |
| SPY | random_forest | -33.49% | -38.53% | -4.53 | -34.16% |
| SPY | logistic_regression | -39.23% | -38.53% | -5.21 | -39.84% |
| SPY | majority_class | -38.53% | -38.53% | -5.01 | -39.99% |

## Audit & Research Conclusion

**Classification**: **No demonstrated edge after transaction costs**: Friction costs, realistic execution timing, and market noise eliminate directional predictive advantages.
