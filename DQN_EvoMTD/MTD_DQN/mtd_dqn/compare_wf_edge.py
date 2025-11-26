"""Side-by-side comparison between WF–MTD DQN and wf_mtd_edge outputs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


DEFAULT_METRICS = ("defender_payoff", "attacker_payoff", "loss")


@dataclass
class RunData:
    """Container describing artefacts loaded from a run directory."""

    name: str
    root: Path
    train_metrics: pd.DataFrame
    eval_summary: Dict[str, float]
    table_summaries: Dict[str, pd.DataFrame]


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare WF–MTD DQN and wf_mtd_edge runs produced by the repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dqn-dir", required=True, help="Path to MTD_DQN results directory.")
    parser.add_argument("--wf-dir", required=True, help="Path to wf_mtd_edge results directory.")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma separated list of training metrics to compare.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Rolling window used to smooth the training curves.",
    )
    parser.add_argument(
        "--out",
        default="comparison_outputs",
        help="Directory where comparison tables and figures will be saved.",
    )
    return parser.parse_args(args)


def _find_logs_dir(root: Path) -> Optional[Path]:
    candidates = [
        root,
        root / "logs",
        root / "results" / "logs",
        root / "results",
    ]
    for candidate in candidates:
        csv_path = candidate / "train_metrics.csv"
        if csv_path.exists():
            return candidate
    return None


def load_training_metrics(root: Path) -> pd.DataFrame:
    logs_dir = _find_logs_dir(root)
    if logs_dir is None:
        return pd.DataFrame()
    csv_path = logs_dir / "train_metrics.csv"
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    df["iteration"] = np.arange(len(df))
    return df


def load_eval_summary(root: Path) -> Dict[str, float]:
    candidates = [
        root / "eval_summary.json",
        root / "results" / "logs" / "eval_summary.json",
        root / "logs" / "eval_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _find_table_dirs(root: Path) -> List[Path]:
    table_dirs: List[Path] = []
    for candidate in root.rglob("tables"):
        if candidate.is_dir():
            table_dirs.append(candidate)
    return table_dirs


def load_table_summaries(root: Path) -> Dict[str, pd.DataFrame]:
    summaries: Dict[str, pd.DataFrame] = {}
    for table_dir in _find_table_dirs(root):
        for csv_path in table_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty:
                continue
            key = str(csv_path.relative_to(root))
            summaries[key] = df
    return summaries


def load_run(name: str, root: Path) -> RunData:
    return RunData(
        name=name,
        root=root,
        train_metrics=load_training_metrics(root),
        eval_summary=load_eval_summary(root),
        table_summaries=load_table_summaries(root),
    )


def compute_summary_statistics(df: pd.DataFrame, metric: str, window: int) -> Dict[str, float]:
    series = df[metric]
    if window > 0 and series.size > window:
        tail = series.iloc[-window:]
    else:
        tail = series
    return {
        "mean": float(tail.mean()),
        "std": float(tail.std(ddof=0)),
        "min": float(tail.min()),
        "max": float(tail.max()),
    }


def plot_metric_overlay(
    runs: Sequence[RunData],
    metric: str,
    window: int,
    output_dir: Path,
) -> Optional[Path]:
    plt.figure(figsize=(8, 5))
    has_data = False
    for run in runs:
        if metric not in run.train_metrics.columns:
            continue
        has_data = True
        series = run.train_metrics[metric]
        if window > 0 and series.size > window:
            smoothed = series.rolling(window=window, min_periods=1).mean()
        else:
            smoothed = series
        plt.plot(run.train_metrics["iteration"], smoothed, label=run.name)

    if not has_data:
        plt.close()
        return None

    plt.title(f"Comparison – {metric}")
    plt.xlabel("Training iteration")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    output_dir = ensure_dir(output_dir)
    png_path = output_dir / f"{metric}_comparison.png"
    pdf_path = output_dir / f"{metric}_comparison.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()
    return png_path


def write_summary_table(
    runs: Sequence[RunData],
    metrics: Sequence[str],
    window: int,
    output_dir: Path,
) -> Path:
    records: List[Dict[str, object]] = []
    for run in runs:
        for metric in metrics:
            if metric not in run.train_metrics.columns:
                continue
            stats = compute_summary_statistics(run.train_metrics, metric, window)
            record = {"model": run.name, "metric": metric}
            record.update(stats)
            records.append(record)
        for key, value in run.eval_summary.items():
            if isinstance(value, (int, float)):
                records.append(
                    {
                        "model": run.name,
                        "metric": f"eval::{key}",
                        "mean": float(value),
                        "std": 0.0,
                        "min": float(value),
                        "max": float(value),
                    }
                )
        for table_name, df in run.table_summaries.items():
            if "Player" in df.columns:
                grouped = df.groupby("Player")
                iterables = grouped
            else:
                iterables = [("all", df)]
            for group_name, group_df in iterables:
                numeric = group_df.select_dtypes(include=[np.number])
                if numeric.empty:
                    continue
                agg_mean = numeric.mean()
                agg_std = numeric.std(ddof=0)
                agg_min = numeric.min()
                agg_max = numeric.max()
                for col in numeric.columns:
                    records.append(
                        {
                            "model": run.name,
                            "metric": f"table::{table_name}::{group_name}::{col}",
                            "mean": float(agg_mean[col]),
                            "std": float(agg_std[col]),
                            "min": float(agg_min[col]),
                            "max": float(agg_max[col]),
                        }
                    )
    df = pd.DataFrame(records)
    csv_path = ensure_dir(output_dir) / "comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return csv_path


def compare_runs(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.out)
    dqn_run = load_run("MTD_DQN", Path(args.dqn_dir))
    wf_run = load_run("wf_mtd_edge", Path(args.wf_dir))
    runs = [dqn_run, wf_run]

    requested = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    metrics_collected: List[str] = []
    for metric in requested:
        if metric in dqn_run.train_metrics.columns or metric in wf_run.train_metrics.columns:
            metrics_collected.append(metric)
    if not metrics_collected:
        raise ValueError("None of the requested metrics are available in the provided runs.")
    metrics = sorted(set(metrics_collected))

    for metric in metrics:
        plot_metric_overlay(runs, metric, args.window, output_dir / "figures")

    summary_path = write_summary_table(runs, metrics, args.window, output_dir / "tables")
    print(f"Comparison complete. Summary table at {summary_path}")


def main() -> None:
    args = parse_args()
    compare_runs(args)


if __name__ == "__main__":
    main()
