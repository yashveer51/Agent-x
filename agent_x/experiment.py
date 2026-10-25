import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

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

    def generate_markdown_report(self, summary_results: List[Dict[str, Any]], exp_name: str) -> str:
        report_path = os.path.join(self.output_dir, f"{exp_name}_REPORT.md")

        md = []
        md.append(f"# Agent X Quantitative Research Report: {exp_name}\n")
        md.append(f"**Generated Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.append("## Executive Summary\n")
        md.append("Agent X evaluates machine learning stock predictions using chronological walk-forward validation and realistic transaction costs.\n")

        md.append("## Out-of-Sample Performance Summary\n")
        md.append("| Asset | Model | Gross Return | Net Return | Sharpe Ratio | Sharpe 95% CI | Max Drawdown | Total Costs |")
        md.append("|---|---|---|---|---|---|---|---|")

        for res in summary_results:
            ci = res.get('sharpe_ci', (0.0, 0.0))
            md.append(
                f"| {res['symbol']} | {res['model']} | {res['gross_return']:.2%} | {res['net_return']:.2%} | "
                f"{res['sharpe_ratio']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {res['max_drawdown']:.2%} | {res['total_costs']:.4f} |"
            )

        md.append("\n## Audit & Research Conclusion\n")

        # Check overall profitability across tested assets
        net_returns = [r['net_return'] for r in summary_results]
        sharpes = [r['sharpe_ratio'] for r in summary_results]

        if all(s > 0.5 for s in sharpes) and all(r > 0 for r in net_returns):
            conclusion = "**Promising but unconfirmed**: Models demonstrate positive net Sharpe across folds. Further paper-trading validation recommended."
        elif any(r < 0 for r in net_returns):
            conclusion = "**No demonstrated edge after transaction costs**: Friction costs and market noise eliminate directional predictive advantages."
        else:
            conclusion = "**Insufficient evidence**: Performance is marginal or statistically indistinguishable from zero."

        md.append(f"**Classification**: {conclusion}\n")

        with open(report_path, "w") as f:
            f.write("\n".join(md))

        return report_path
