# Agent X Quantitative Research Report: Verified_Agent_X_Audit

**Generated Date**: 2026-10-25 05:50:54
**Commit SHA**: `acf360542f98557a6d8c719a86b487108a5b0785`

## Executive Summary

Agent X evaluates machine learning stock predictions using chronological walk-forward validation and realistic transaction costs under Open-to-Close execution timing.

## Out-of-Sample Performance Summary

| Asset | Model | Gross Return | Net Return | Benchmark Net (Executable) | Benchmark Frictionless | Sharpe Ratio | Sharpe 95% CI | Max Drawdown | Total Costs |
|---|---|---|---|---|---|---|---|---|---|
| AAPL | random_forest | 76.65% | -55.44% | -62.89% | 113.92% | -0.88 | [-2.03, -0.00] | -59.73% | 1.3760 |
| AAPL | logistic_regression | 100.13% | -48.91% | -62.89% | 113.92% | -0.79 | [-1.99, 0.18] | -57.53% | 1.3640 |
| AAPL | majority_class | 113.92% | -62.89% | -62.89% | 113.92% | -0.96 | [-2.18, 0.07] | -70.23% | 1.7500 |
| MSFT | random_forest | 20.52% | -61.96% | -73.88% | 50.56% | -1.25 | [-2.29, -0.44] | -64.96% | 1.1520 |
| MSFT | logistic_regression | 52.29% | -51.64% | -73.88% | 50.56% | -0.95 | [-2.10, -0.11] | -65.32% | 1.1460 |
| MSFT | majority_class | 50.56% | -73.88% | -73.88% | 50.56% | -1.45 | [-2.41, -0.77] | -76.31% | 1.7500 |
| SPY | random_forest | 17.97% | -72.70% | -78.30% | 25.08% | -2.51 | [-3.55, -1.74] | -73.44% | 1.4620 |
| SPY | logistic_regression | 8.44% | -77.07% | -78.30% | 25.08% | -3.01 | [-3.88, -2.21] | -77.07% | 1.5520 |
| SPY | majority_class | 25.08% | -78.30% | -78.30% | 25.08% | -2.68 | [-3.77, -2.02] | -79.81% | 1.7500 |

## Final Isolated Holdout Performance Summary

| Asset | Model | Holdout Net Return | Holdout Benchmark Net | Holdout Sharpe Ratio | Holdout Max Drawdown |
|---|---|---|---|---|---|
| AAPL | random_forest | -12.14% | -17.92% | -0.67 | -24.36% |
| AAPL | logistic_regression | -16.54% | -17.92% | -0.91 | -28.51% |
| AAPL | majority_class | -17.92% | -17.92% | -0.99 | -29.69% |
| MSFT | random_forest | -37.16% | -40.91% | -2.85 | -38.76% |
| MSFT | logistic_regression | -38.14% | -40.91% | -2.88 | -41.22% |
| MSFT | majority_class | -40.91% | -40.91% | -3.12 | -43.81% |
| SPY | random_forest | -37.58% | -38.53% | -4.99 | -38.21% |
| SPY | logistic_regression | -39.23% | -38.53% | -5.21 | -39.84% |
| SPY | majority_class | -38.53% | -38.53% | -5.01 | -39.99% |

## Audit & Research Conclusion

**Classification**: **No demonstrated edge after transaction costs**: Friction costs, realistic execution timing, and market noise eliminate directional predictive advantages.
