"""Comparative performance export utilities for WF–MTD baselines."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class StepLog:
    t: int
    state: Any
    attacker_action: str
    defender_action: str
    UD: float
    UA: float
    SAP: float
    SAL: float
    DC: float
    ASSC: float
    NC: float
    AIC: float
    AS_success: int
    is_absorbing: int
    power_W: Optional[float] = None
    q_vec: Optional[List[float]] = None
    p_vec: Optional[List[float]] = None
    td_loss: Optional[float] = None
    entropy: Optional[float] = None


def _ct_epsilon(q_hist: List[List[float]], p_hist: Optional[List[List[float]]], eps: float = 1e-3) -> int:
    if not q_hist or len(q_hist) < 2:
        return -1
    q_star = np.array(q_hist[-1])
    p_star = np.array(p_hist[-1]) if p_hist else None
    for t in range(len(q_hist) - 2, -1, -1):
        dq = float(np.sum(np.abs(np.array(q_hist[t]) - q_star)))
        dp = 0.0
        if p_hist and t < len(p_hist):
            dp = float(np.sum(np.abs(np.array(p_hist[t]) - p_star)))
        if dq + dp <= eps:
            return t
    return -1


def _oscillation(q_hist: List[List[float]]) -> float:
    if not q_hist or len(q_hist) < 2:
        return float("nan")
    diffs = [
        float(np.sum(np.abs(np.array(q_hist[idx + 1]) - np.array(q_hist[idx]))))
        for idx in range(len(q_hist) - 1)
    ]
    return float(np.mean(diffs))


def _time_to_compromise(steps: List[StepLog]) -> float:
    for step in steps:
        if step.is_absorbing == 1:
            return float(step.t)
    return float("inf")


def _sal_reduction(sal_series: np.ndarray, sal_baseline: Optional[float]) -> Optional[float]:
    if sal_baseline is None or sal_baseline <= 0:
        return None
    return float(100.0 * (sal_baseline - float(np.mean(sal_series))) / sal_baseline)


def _perf_watt(ud_mean: float, p_mean: Optional[float]) -> Optional[float]:
    if p_mean is None or p_mean <= 0:
        return None
    return float(ud_mean / p_mean)


def _edp(energy_j: Optional[float], ct_eps: int) -> Optional[float]:
    if energy_j is None or not math.isfinite(energy_j) or ct_eps < 0:
        return None
    return float(energy_j * ct_eps)


def _to_dataframe(steps: List[StepLog]) -> pd.DataFrame:
    records = []
    for step in steps:
        record = asdict(step)
        if isinstance(record.get("q_vec"), np.ndarray):
            record["q_vec"] = record["q_vec"].tolist()
        if isinstance(record.get("p_vec"), np.ndarray):
            record["p_vec"] = record["p_vec"].tolist()
        records.append(record)
    if records:
        return pd.DataFrame(records)
    column_names = [field.name for field in fields(StepLog)]
    return pd.DataFrame(columns=column_names)


def compute_comp_perf(
    steps: List[StepLog],
    q_hist: Optional[List[List[float]]] = None,
    p_hist: Optional[List[List[float]]] = None,
    sal_baseline: Optional[float] = None,
    power_train_energy_j: Optional[float] = None,
    uncertainty_sweep: Optional[pd.DataFrame] = None,
    out_dir: Path = Path("comp-perf"),
    model_name: str = "model",
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = _to_dataframe(steps)
    df.to_csv(out_dir / "timeseries.csv", index=False)

    ud_mean = float(df["UD"].mean()) if "UD" in df else float("nan")
    sal_mean = float(df["SAL"].mean()) if "SAL" in df else float("nan")
    asr_series = df["AS_success"] if "AS_success" in df else pd.Series(dtype=float)
    asr = float(asr_series.mean()) if not asr_series.empty else float("nan")
    ttc = _time_to_compromise(steps)

    q_hist = q_hist or []
    p_hist = p_hist or []
    ct_eps = _ct_epsilon(q_hist, p_hist, eps=1e-3)
    osc = _oscillation(q_hist)

    dc_mean = float(df["DC"].mean()) if "DC" in df else float("nan")
    assc_mean = float(df["ASSC"].mean()) if "ASSC" in df else float("nan")
    nc_mean = float(df["NC"].mean()) if "NC" in df else float("nan")
    aic_mean = float(df["AIC"].mean()) if "AIC" in df else float("nan")

    p_mean = None
    energy_run = None
    if "power_W" in df and df["power_W"].notna().any():
        p_mean = float(df["power_W"].dropna().mean())
        energy_run = float(df["power_W"].dropna().sum())

    sal_array = df["SAL"].values if "SAL" in df else np.array([])
    sal_reduction = _sal_reduction(sal_array, sal_baseline)
    perf_per_watt = _perf_watt(ud_mean, p_mean)
    edp = _edp(energy_run, ct_eps)

    if uncertainty_sweep is not None and not uncertainty_sweep.empty:
        uncertainty_sweep.to_csv(out_dir / "robustness.csv", index=False)

    summary = {
        "model": model_name,
        "UD_mean": ud_mean,
        "SAL_mean": sal_mean,
        "SAL_reduction_pct": sal_reduction,
        "ASR": asr,
        "TTC": ttc,
        "CT_epsilon": ct_eps,
        "Oscillation": osc,
        "DC_mean": dc_mean,
        "ASSC_mean": assc_mean,
        "NC_mean": nc_mean,
        "AIC_mean": aic_mean,
        "Power_mean_W": p_mean,
        "Energy_run_J": energy_run,
        "Energy_train_J": power_train_energy_j,
        "Perf_per_Watt": perf_per_watt,
        "EDP": edp,
    }

    pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False)
    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not df.empty:
        fig, ax = plt.subplots()
        ax.plot(df["t"], df["UD"], label="U_D")
        if "SAL" in df:
            ax.plot(df["t"], df["SAL"], label="SAL")
        ud_sal_cols = ["t"]
        if "UD" in df:
            ud_sal_cols.append("UD")
        if "SAL" in df:
            ud_sal_cols.append("SAL")
        df[ud_sal_cols].to_csv(tables_dir / "ud_sal_over_time.csv", index=False)
        ax.set_xlabel("plays / steps")
        ax.set_ylabel("value")
        ax.set_title(f"{model_name}: Utility & SAL")
        ax.legend()
        fig.savefig(figures_dir / "ud_sal_over_time.png", dpi=220)
        plt.close(fig)

        if "AS_success" in df:
            cumulative = df["AS_success"].expanding().mean()
            pd.DataFrame({"t": df["t"], "ASR_cumulative": cumulative}).to_csv(
                tables_dir / "asr_over_time.csv", index=False
            )
            fig, ax = plt.subplots()
            ax.plot(df["t"], cumulative, label="ASR (cumulative)")
            ax.set_xlabel("plays / steps")
            ax.set_ylabel("rate")
            ax.set_title(f"{model_name}: Attack success rate")
            ax.legend()
            fig.savefig(figures_dir / "asr_over_time.png", dpi=220)
            plt.close(fig)

        if q_hist:
            q_arr = np.array(q_hist)
            q_df = pd.DataFrame(q_arr, columns=[f"q_{idx+1}" for idx in range(q_arr.shape[1])])
            q_df.insert(0, "t", np.arange(len(q_arr)))
            q_df.to_csv(tables_dir / "q_mix_trajectory.csv", index=False)
            fig, ax = plt.subplots()
            for idx in range(q_arr.shape[1]):
                ax.plot(range(len(q_arr)), q_arr[:, idx], label=f"q_{idx+1}")
            ax.set_xlabel("plays / steps")
            ax.set_ylabel("probability")
            ax.set_title(f"{model_name}: defender mixture")
            ax.legend()
            fig.savefig(figures_dir / "q_mix_trajectory.png", dpi=220)
            plt.close(fig)

        fig, ax = plt.subplots()
        overhead_vals = [assc_mean, nc_mean, aic_mean]
        pd.DataFrame(
            {
                "component": ["ASSC", "NC", "AIC"],
                "mean": overhead_vals,
            }
        ).to_csv(tables_dir / "overhead_components.csv", index=False)
        ax.bar(["ASSC", "NC", "AIC"], overhead_vals)
        ax.set_ylabel("mean value")
        ax.set_title(f"{model_name}: Overhead components")
        fig.savefig(figures_dir / "overhead_bar.png", dpi=220)
        plt.close(fig)

        if p_mean is not None:
            fig, ax = plt.subplots()
            power_series = df["power_W"].fillna(method="ffill")
            pd.DataFrame({"t": df["t"], "power_W": power_series}).to_csv(
                tables_dir / "power_timeline.csv", index=False
            )
            ax.plot(df["t"], power_series)
            ax.set_xlabel("plays / steps")
            ax.set_ylabel("power (W)")
            ax.set_title(f"{model_name}: power usage")
            fig.savefig(figures_dir / "power_timeline.png", dpi=220)
            plt.close(fig)

    (tables_dir / "summary_metrics.tex").write_text(
        pd.DataFrame([summary]).to_latex(index=False, float_format="%.4g"), encoding="utf-8"
    )

    return summary


__all__ = [
    "StepLog",
    "compute_comp_perf",
    "_ct_epsilon",
    "_oscillation",
    "_time_to_compromise",
    "_sal_reduction",
    "_perf_watt",
    "_edp",
]
