"""Plot per-strategy response and cost breakdown."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-strategy response cost charts.")
    parser.add_argument("--data", type=Path, default=Path("comp-perf/dqn_evomtd/response_time/per_strategy_response_cost.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("comp-perf/dqn_evomtd/response_time"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    grouped = df.groupby(["def_strategy", "mu"]).mean(numeric_only=True).reset_index()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = sorted(grouped["def_strategy"].unique())
    mu_values = sorted(grouped["mu"].unique())
    x = np.arange(len(strategies))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, mu in enumerate(mu_values):
        subset = grouped[grouped["mu"] == mu].set_index("def_strategy").reindex(strategies)
        values = subset["resp_time_ms_mean"].values
        ax.bar(x + idx * width, values, width, label=f"μ={mu}")
        ax.errorbar(
            x + idx * width,
            values,
            yerr=subset["resp_time_ms_p95"].values - values,
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    ax.set_xticks(x + width * (len(mu_values) - 1) / 2, strategies)
    ax.set_ylabel("Response time mean (ms)")
    ax.set_title("Per-strategy response time")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "per_strategy_response_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(strategies))
    for metric in ["ASSC_ms_mean", "AIC_ms_mean", "NC_ms_mean"]:
        values = df.groupby("def_strategy")[metric].mean().reindex(strategies).values
        ax.bar(strategies, values, bottom=bottoms, label=metric.replace("_ms_mean", ""))
        bottoms += values
    ax.set_ylabel("Cost (ms)")
    ax.set_title("Per-strategy cost breakdown")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "per_strategy_cost_breakdown.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
