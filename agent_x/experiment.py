import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

class ExperimentTracker:
    """
    Records and tracks research experiment parameters, metrics, and generates markdown reports.
    """

    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_experiment(self, experiment_data: Dict[str, Any], exp_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{exp_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(experiment_data, f, indent=2, default=str)

        return filepath

    def generate_markdown_report(
        self,
        summary_results: List[Dict[str, Any]],
        exp_name: str,
        commit_sha: Optional[str] = None
    ) -> str:
        report_path = os.path.join(self.output_dir, f"{exp_name}_REPORT.md")

        md = []
        md.append(f"# Agent X Quantitative Research Report: {exp_name}\n")
        md.append(f"**Generated Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if commit_sha:
            md.append(f"**Commit SHA**: `{commit_sha}`")
        md.append("\n## Executive Summary\n")
        md.append("Agent X evaluates machine learning stock predictions using chronological walk-forward validation and realistic transaction costs under Open-to-Close execution timing.\n")

        md.append("## Out-of-Sample Performance Summary\n")
        md.append("| Asset | Model | Gross Return | Net Return | Benchmark Net | Sharpe Ratio | Sharpe 95% CI | Max Drawdown | Total Costs |")
        md.append("|---|---|---|---|---|---|---|---|---|")

        for res in summary_results:
            ci = res.get('sharpe_ci', (np.nan, np.nan))
            ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if not np.isnan(ci[0]) else "N/A"
            bm_ret = res.get('benchmark_net_return', res.get('benchmark_return', 0.0))
            md.append(
                f"| {res['symbol']} | {res['model']} | {res['gross_return']:.2%} | {res['net_return']:.2%} | "
                f"{bm_ret:.2%} | {res['sharpe_ratio']:.2f} | {ci_str} | {res['max_drawdown']:.2%} | {res['total_costs']:.4f} |"
            )

        md.append("\n## Audit & Research Conclusion\n")

        net_returns = [r['net_return'] for r in summary_results]
        sharpes = [r['sharpe_ratio'] for r in summary_results]
        sharpe_cis = [r.get('sharpe_ci', (np.nan, np.nan)) for r in summary_results]

        ci_includes_zero_or_negative = any(
            np.isnan(ci[0]) or ci[0] <= 0.0 for ci in sharpe_cis
        )

        if all(s > 0.5 for s in sharpes) and all(r > 0 for r in net_returns) and not ci_includes_zero_or_negative:
            conclusion = "**Promising but unconfirmed**: Models demonstrate positive net Sharpe with 95% confidence intervals above zero. Further paper-trading validation recommended."
        elif any(r < 0 for r in net_returns) or any(s <= 0 for s in sharpes):
            conclusion = "**No demonstrated edge after transaction costs**: Friction costs, realistic execution timing, and market noise eliminate directional predictive advantages."
        else:
            conclusion = "**Insufficient evidence**: Performance is marginal, sample size is limited, or 95% confidence intervals cross zero."

        md.append(f"**Classification**: {conclusion}\n")

        with open(report_path, "w") as f:
            f.write("\n".join(md))

        return report_path
