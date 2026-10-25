import os
import pandas as pd
import numpy as np
from agent_x.data import DataFetcher
from agent_x.features import FeatureEngineer
from agent_x.targets import TargetBuilder
from agent_x.validation import WalkForwardValidator
from agent_x.models import ModelFactory
from agent_x.backtest import BacktestEngine
from agent_x.metrics import PerformanceMetrics
from agent_x.experiment import ExperimentTracker

def run_research_experiment(
    symbols: list[str] = ["AAPL", "MSFT", "SPY"],
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    model_types: list[str] = ["random_forest", "logistic_regression", "majority_class"],
    transaction_cost_pct: float = 0.0010,
    use_cache: bool = True
):
    print(f"--- Running Agent X Research Experiment ---")
    fetcher = DataFetcher()
    tracker = ExperimentTracker()

    summary_results = []

    for symbol in symbols:
        print(f"\n[ASSET] Processing symbol: {symbol}")
        try:
            df = fetcher.fetch_symbol(symbol, start_date=start_date, end_date=end_date, use_cache=use_cache)
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")
            continue

        fetch_info = fetcher.last_fetch_info.get(symbol, {})
        print(f"Data source: {fetch_info.get('data_source')}, range: {fetch_info.get('min_date')} to {fetch_info.get('max_date')} ({fetch_info.get('n_observations')} rows)")

        # Feature engineering & Target building
        df_feat, feature_cols = FeatureEngineer.compute_features(df)
        df_target = TargetBuilder.add_directional_target(df_feat, execution_convention="open_to_close", use_log_return=False)

        # Drop invalid target rows (e.g. last row)
        df_valid = df_target.dropna(subset=['Target'] + feature_cols).copy()

        validator = WalkForwardValidator(min_train_size=500, test_size=125, step_size=125, purge_gap=1, holdout_size=250)

        for model_type in model_types:
            print(f"  [MODEL] Evaluating model: {model_type}")

            fold_predictions = []
            fold_test_dfs = []

            for train_df, test_df, fold in validator.split(df_valid):
                # Preprocessing/scaling fitted strictly on train fold
                X_train, X_test, scaler = validator.fit_transform_features(train_df, test_df, feature_cols)
                y_train = train_df['Target'].values

                model = ModelFactory.get_model(model_type=model_type, random_state=42)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                fold_predictions.extend(preds)
                fold_test_dfs.append(test_df)

            if not fold_test_dfs:
                print(f"    No folds generated for {symbol} {model_type}")
                continue

            concat_test_df = pd.concat(fold_test_dfs)
            signals_series = pd.Series(fold_predictions, index=concat_test_df.index)

            # Backtest strategy
            engine = BacktestEngine(transaction_cost_pct=transaction_cost_pct, use_log_return=False)
            backtest_res = engine.run_backtest(concat_test_df, signals_series)

            # Calculate financial metrics
            metrics = PerformanceMetrics.calculate_financial_metrics(
                returns=backtest_res['Strategy_Return'],
                costs=backtest_res['Cost'],
                wealth=backtest_res['Wealth'],
                is_log_return=False
            )

            # Benchmark 1: Investable Executable Buy & Hold (incorporates realistic transaction cost on entry/exit)
            bm_investable_engine = BacktestEngine(transaction_cost_pct=transaction_cost_pct, use_log_return=False)
            bm_signals = pd.Series(1.0, index=concat_test_df.index)
            bm_res = bm_investable_engine.run_backtest(concat_test_df, bm_signals)
            bm_metrics = PerformanceMetrics.calculate_financial_metrics(
                returns=bm_res['Strategy_Return'],
                wealth=bm_res['Wealth'],
                is_log_return=False
            )

            # Benchmark 2: Frictionless Buy & Hold
            bm_frictionless_engine = BacktestEngine(transaction_cost_pct=0.0000, use_log_return=False)
            bm_fric_res = bm_frictionless_engine.run_backtest(concat_test_df, bm_signals)
            bm_fric_metrics = PerformanceMetrics.calculate_financial_metrics(
                returns=bm_fric_res['Strategy_Return'],
                wealth=bm_fric_res['Wealth'],
                is_log_return=False
            )

            sharpe_ci = PerformanceMetrics.block_bootstrap_sharpe_ci(backtest_res['Strategy_Return'], random_seed=42)

            res_dict = {
                'symbol': symbol,
                'model': model_type,
                'gross_return': float(np.prod(1.0 + backtest_res['Gross_Return']) - 1.0),
                'net_return': metrics['total_return'],
                'benchmark_net_return': bm_metrics['total_return'],
                'benchmark_frictionless_return': bm_fric_metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sharpe_ci': sharpe_ci,
                'max_drawdown': metrics['max_drawdown'],
                'total_costs': metrics['total_costs'],
                'n_observations': len(concat_test_df),
                'data_info': fetch_info
            }

            # Evaluate untouched final holdout period if available
            holdout_df = validator.get_holdout(df_valid)
            if holdout_df is not None and not holdout_df.empty:
                # Train model on all walk-forward validation data
                X_train_full, X_holdout, _ = validator.fit_transform_features(df_valid.iloc[:-validator.holdout_size], holdout_df, feature_cols)
                y_train_full = df_valid.iloc[:-validator.holdout_size]['Target'].values

                holdout_model = ModelFactory.get_model(model_type=model_type, random_state=42)
                holdout_model.fit(X_train_full, y_train_full)

                holdout_preds = holdout_model.predict(X_holdout)
                holdout_signals = pd.Series(holdout_preds, index=holdout_df.index)

                holdout_backtest = engine.run_backtest(holdout_df, holdout_signals)
                holdout_metrics = PerformanceMetrics.calculate_financial_metrics(
                    returns=holdout_backtest['Strategy_Return'],
                    costs=holdout_backtest['Cost'],
                    wealth=holdout_backtest['Wealth'],
                    is_log_return=False
                )

                holdout_bm_signals = pd.Series(1.0, index=holdout_df.index)
                holdout_bm_backtest = bm_investable_engine.run_backtest(holdout_df, holdout_bm_signals)
                holdout_bm_metrics = PerformanceMetrics.calculate_financial_metrics(
                    returns=holdout_bm_backtest['Strategy_Return'],
                    wealth=holdout_bm_backtest['Wealth'],
                    is_log_return=False
                )

                res_dict['holdout_net_return'] = holdout_metrics['total_return']
                res_dict['holdout_bm_net_return'] = holdout_bm_metrics['total_return']
                res_dict['holdout_sharpe_ratio'] = holdout_metrics['sharpe_ratio']
                res_dict['holdout_max_drawdown'] = holdout_metrics['max_drawdown']

            summary_results.append(res_dict)
            print(f"    Net Return: {res_dict['net_return']:.2%}, Benchmark: {res_dict['benchmark_net_return']:.2%}, Sharpe: {res_dict['sharpe_ratio']:.2f}, CI: [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]")

    try:
        import subprocess
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        commit_sha = "UNKNOWN_COMMIT"

    report_path = tracker.generate_markdown_report(summary_results, exp_name="Verified_Agent_X_Audit", commit_sha=commit_sha)
    json_path = tracker.save_experiment({"results": summary_results}, exp_name="Verified_Agent_X_Audit")

    print(f"\nExperiment complete. Markdown report saved to: {report_path}")
    print(f"JSON results saved to: {json_path}")

if __name__ == "__main__":
    run_research_experiment()
