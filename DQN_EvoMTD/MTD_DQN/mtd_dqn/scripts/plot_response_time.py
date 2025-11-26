"""Plotting helpers for response-time response CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_fig6(df: pd.DataFrame, out_path: Path) -> None:
    panels = sorted(df["panel"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.flatten()
    for ax, panel in zip(axes, panels):
        subset = df[df["panel"] == panel]
        for (n_attackers, n_targets), grp in subset.groupby(["n_attackers", "n_targets"]):
            grp_sorted = grp.sort_values("time_step")
            ax.plot(
                grp_sorted["time_step"],
                grp_sorted["avg_return"],
                label=f"nA={n_attackers}, nT={n_targets}",
            )
        ax.set_title(f"Panel {panel.upper()} – mode {subset['attack_mode'].iloc[0]}")
        ax.set_xlabel("time step")
        ax.set_ylabel("avg return")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_method_bars(df: pd.DataFrame, out_path: Path, title: str) -> None:
    mu_values = sorted(df["mu"].unique())
    methods = df["method"].unique()
    fig, axes = plt.subplots(
        1,
        len(mu_values),
        figsize=(4 * len(mu_values), 5),
        sharey=True,
    )
    if len(mu_values) == 1:
        axes = [axes]
    for ax, mu in zip(axes, mu_values):
        subset = df[df["mu"] == mu]
        subset = subset.set_index("method").loc[methods]
        y = subset["metric"].values
        p95 = subset["p95"].values
        x = np.arange(len(methods))
        ax.bar(x, y, yerr=p95 - y, capsize=4)
        ax.set_xticks(x, methods, rotation=45, ha="right")
        ax.set_title(f"μ = {mu} concurrent")
        ax.set_ylabel(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot response-time CSVs.")
    parser.add_argument("--outdir", type=Path, default=Path("comp-perf/dqn_evomtd/response_time"))
    args = parser.parse_args()

    out_dir = args.outdir
    fig6_df = _load_csv(out_dir / "fig6_avg_return_sweep.csv")
    plot_fig6(fig6_df, out_dir / "fig6_avg_return_sweep.png")
    fig7_df = _load_csv(out_dir / "fig7_request_time_vs_mu.csv")
    _plot_method_bars(fig7_df, out_dir / "fig7_request_time_vs_mu.png", "request time (ms)")
    fig9_df = _load_csv(out_dir / "fig9_load_time_vs_mu.csv")
    _plot_method_bars(fig9_df, out_dir / "fig9_load_time_vs_mu.png", "load time (ms)")


if __name__ == "__main__":
    main()
