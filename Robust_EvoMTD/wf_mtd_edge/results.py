"""
Results module for WF-MTD edge–cloud simulations.

This module assembles the full Results section by transforming the simulation
outputs (robust payoff matrices, evolutionary histories, kernels, and sensitivity
sweeps) into plots, tables, and a human-readable index.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it does not yet exist."""

    path.mkdir(parents=True, exist_ok=True)


def _safe_filename(path_name: str) -> str:
    """Generate a filesystem-safe stem."""

    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path_name)


def _validate_matrix_shapes(
    payoff_A: np.ndarray,
    payoff_D: np.ndarray,
    as_names: List[str],
    ds_names: List[str],
) -> None:
    """Ensure payoff matrices match strategy dimensions."""

    if payoff_A.shape != payoff_D.shape:
        raise ValueError(f"Payoff matrices shape mismatch: {payoff_A.shape} vs {payoff_D.shape}")
    m, n = payoff_A.shape
    if m != len(as_names) or n != len(ds_names):
        raise ValueError(
            f"Payoff matrix shape {payoff_A.shape} does not align with "
            f"{len(as_names)} attacker and {len(ds_names)} defender strategies."
        )


def _validate_history_shapes(
    history: Dict[str, np.ndarray],
    m: int,
    n: int,
) -> None:
    """Ensure evolution history arrays share consistent shapes."""

    required_keys = {"p", "q", "fa", "fd", "Ua", "Ud", "xi_m", "theta"}
    missing = required_keys - set(history)
    if missing:
        raise ValueError(f"Evolution history missing keys: {sorted(missing)}")

    p_hist = history["p"]
    q_hist = history["q"]

    if p_hist.shape[1] != m:
        raise ValueError(f"Attacker history width {p_hist.shape[1]} != {m} strategies.")
    if q_hist.shape[1] != n:
        raise ValueError(f"Defender history width {q_hist.shape[1]} != {n} strategies.")
    if history["fa"].shape != p_hist.shape:
        raise ValueError("fa history must match attacker history shape.")
    if history["fd"].shape != q_hist.shape:
        raise ValueError("fd history must match defender history shape.")
    if history["Ua"].shape[0] != p_hist.shape[0]:
        raise ValueError("Ua trajectory length must equal history length.")
    if history["Ud"].shape[0] != q_hist.shape[0]:
        raise ValueError("Ud trajectory length must equal history length.")
    if history["xi_m"].shape[0] != p_hist.shape[0]:
        raise ValueError("xi_m trajectory length must equal history length.")
    theta_hist = history["theta"]
    required_theta = {"c", "i", "a"}
    if set(theta_hist) != required_theta:
        raise ValueError(f"theta history must provide {required_theta} keys.")
    for key in required_theta:
        if theta_hist[key].shape[0] != p_hist.shape[0]:
            raise ValueError(f"theta[{key}] length must equal history length.")


def _escape_latex_name(name: str) -> str:
    """Escape LaTeX special characters in strategy names."""

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(ch, ch) for ch in name)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def render_payoff_tables(
    path: str,
    state: int,
    payoff_A: np.ndarray,
    payoff_D: np.ndarray,
    as_names: List[str],
    ds_names: List[str],
    out_dir: str,
) -> Dict[str, str]:
    """
    Save attacker/defender payoff matrices as CSV and LaTeX (booktabs).
    Return dict with file paths.

    In the Results section, these tables correspond to the per-state/path stage
    payoffs that ground the equilibrium analysis.
    """

    _validate_matrix_shapes(payoff_A, payoff_D, as_names, ds_names)

    out_path = Path(out_dir)
    _ensure_dir(out_path)

    idx = pd.Index(as_names, name="Attacker Strategy")
    cols = ds_names

    df_A = pd.DataFrame(payoff_A, index=idx, columns=cols)
    df_D = pd.DataFrame(payoff_D, index=idx, columns=cols)

    stem = f"{_safe_filename(path)}_state{state}"
    A_csv = out_path / f"{stem}_payoff_A.csv"
    D_csv = out_path / f"{stem}_payoff_D.csv"
    A_tex = out_path / f"{stem}_payoff_A.tex"
    D_tex = out_path / f"{stem}_payoff_D.tex"

    df_A.to_csv(A_csv)
    df_D.to_csv(D_csv)

    df_A.rename(index=_escape_latex_name, columns=_escape_latex_name).to_latex(
        A_tex,
        escape=False,
        float_format="%.4f",
        bold_rows=False,
        index=True,
        caption=None,
        label=None,
        position="h!",
        column_format="l" + "r" * len(cols),
        longtable=False,
        multicolumn=True,
        multicolumn_format="c",
        multirow=False,
        header=True,
    )
    df_D.rename(index=_escape_latex_name, columns=_escape_latex_name).to_latex(
        D_tex,
        escape=False,
        float_format="%.4f",
        bold_rows=False,
        index=True,
        caption=None,
        label=None,
        position="h!",
        column_format="l" + "r" * len(cols),
        longtable=False,
        multicolumn=True,
        multicolumn_format="c",
        multirow=False,
        header=True,
    )

    return {
        "payoff_A_csv": str(A_csv),
        "payoff_D_csv": str(D_csv),
        "payoff_A_tex": str(A_tex),
        "payoff_D_tex": str(D_tex),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_strategy_trajectories(
    path: str,
    state: int,
    p_hist: np.ndarray,
    q_hist: np.ndarray,
    as_names: List[str],
    ds_names: List[str],
    out_dir: str,
    dpi: int = 160,
) -> str:
    """
    Line plot of evolutionary shares p(t), q(t) with legend; save PNG+PDF; return PNG path.

    This figure summarizes the bounded-rational Wright–Fisher dynamics for both
    agents, highlighting convergence behaviour in the Results section.
    """

    if p_hist.shape[0] != q_hist.shape[0]:
        raise ValueError("Attacker and defender history lengths must match.")

    out_path = Path(out_dir)
    _ensure_dir(out_path)

    steps = np.arange(p_hist.shape[0])
    stem = f"{_safe_filename(path)}_state{state}_mix"
    png_path = out_path / f"{stem}.png"
    pdf_path = out_path / f"{stem}.pdf"

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, dpi=dpi)
    axes[0].set_title(f"Attacker Mix – {path} / State {state}")
    for idx, name in enumerate(as_names):
        axes[0].plot(steps, p_hist[:, idx], label=name)
    axes[0].set_ylabel("Probability")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    axes[1].set_title("Defender Mix")
    for idx, name in enumerate(ds_names):
        axes[1].plot(steps, q_hist[:, idx], label=name)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Probability")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return str(png_path)


def plot_payoff_heatmaps(
    path: str,
    state: int,
    payoff_A: np.ndarray,
    payoff_D: np.ndarray,
    as_names: List[str],
    ds_names: List[str],
    out_dir: str,
    dpi: int = 160,
) -> Dict[str, str]:
    """
    Two heatmaps (A and D), consistent colorbars; return dict with figure paths.

    These heatmaps visualize the robust one-step payoffs (tilde matrices) for each
    agent, tying directly to the Results comparison of strategy pairs.
    """

    _validate_matrix_shapes(payoff_A, payoff_D, as_names, ds_names)

    out_path = Path(out_dir)
    _ensure_dir(out_path)

    stem = f"{_safe_filename(path)}_state{state}_payoff"
    A_png = out_path / f"{stem}_A.png"
    A_pdf = out_path / f"{stem}_A.pdf"
    D_png = out_path / f"{stem}_D.png"
    D_pdf = out_path / f"{stem}_D.pdf"

    vmin = min(payoff_A.min(), payoff_D.min())
    vmax = max(payoff_A.max(), payoff_D.max())

    def _plot(data: np.ndarray, title: str, png: Path, pdf: Path) -> None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(ds_names)), labels=ds_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(as_names)), labels=as_names)
        ax.set_xlabel("Defender Strategy")
        ax.set_ylabel("Attacker Strategy")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white"
                    if (data[i, j] - vmin) / (vmax - vmin + 1e-9) < 0.5
                    else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

    _plot(payoff_A, f"Attacker Payoff – {path} / State {state}", A_png, A_pdf)
    _plot(payoff_D, f"Defender Payoff – {path} / State {state}", D_png, D_pdf)

    return {
        "A_png": str(A_png),
        "A_pdf": str(A_pdf),
        "D_png": str(D_png),
        "D_pdf": str(D_pdf),
    }


def plot_lateral_and_success(
    path: str,
    state: int,
    xi_hist: np.ndarray,
    theta_hist: Dict[str, np.ndarray],
    out_dir: str,
    dpi: int = 160,
) -> Dict[str, str]:
    """
    Plot xi_m(t) time series and per-attribute theta_x(t) in separate figures; return paths.

    These trajectories document the lateral spread attenuation and success
    probabilities that underpin the Results discussion of defensive efficacy.
    """

    out_path = Path(out_dir)
    _ensure_dir(out_path)

    steps = np.arange(xi_hist.shape[0])
    stem = f"{_safe_filename(path)}_state{state}"

    xi_png = out_path / f"{stem}_xi.png"
    xi_pdf = out_path / f"{stem}_xi.pdf"
    theta_png = out_path / f"{stem}_theta.png"
    theta_pdf = out_path / f"{stem}_theta.pdf"

    # xi plot
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=dpi)
    ax.plot(steps, xi_hist, color="tab:purple", linewidth=2)
    ax.set_title(f"Lateral Coefficient $\\xi_m(t)$ – {path} / State {state}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("$\\xi_m$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(xi_png, bbox_inches="tight")
    fig.savefig(xi_pdf, bbox_inches="tight")
    plt.close(fig)

    # theta plot
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=dpi)
    for key, series in theta_hist.items():
        ax.plot(steps, series, label=f"$\\theta_{{{key}}}$")
    ax.set_title(f"Success Probabilities $\\theta_x(t)$ – {path} / State {state}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("$\\theta_x$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(theta_png, bbox_inches="tight")
    fig.savefig(theta_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "xi_png": str(xi_png),
        "xi_pdf": str(xi_pdf),
        "theta_png": str(theta_png),
        "theta_pdf": str(theta_pdf),
    }


def summarize_final_mix_and_payoff(
    path: str,
    state: int,
    p_hist: np.ndarray,
    q_hist: np.ndarray,
    fa_hist: np.ndarray,
    fd_hist: np.ndarray,
    as_names: List[str],
    ds_names: List[str],
    out_dir: str,
) -> Dict[str, str]:
    """
    Save final mixes (argmax entries and full vectors) and final expected payoffs as CSV.

    These tables support the Results claims about equilibrium strategy profiles and
    expected utilities.
    """

    out_path = Path(out_dir)
    _ensure_dir(out_path)

    final_p = p_hist[-1]
    final_q = q_hist[-1]
    final_fa = fa_hist[-1]
    final_fd = fd_hist[-1]

    stem = f"{_safe_filename(path)}_state{state}"
    mix_csv = out_path / f"{stem}_final_mix.csv"
    payoff_csv = out_path / f"{stem}_final_payoff.csv"

    df_mix = pd.DataFrame(
        {
            "Strategy": as_names + ds_names,
            "Player": ["Attacker"] * len(as_names) + ["Defender"] * len(ds_names),
            "Probability": np.concatenate([final_p, final_q]),
            "Is_Argmax": [
                int(idx == final_p.argmax()) for idx in range(len(as_names))
            ]
            + [int(idx == final_q.argmax()) for idx in range(len(ds_names))],
        }
    )
    df_mix.to_csv(mix_csv, index=False)

    df_payoff = pd.DataFrame(
        {
            "Strategy": as_names + ds_names,
            "Player": ["Attacker"] * len(as_names) + ["Defender"] * len(ds_names),
            "Expected_Payoff": np.concatenate([final_fa, final_fd]),
        }
    )
    df_payoff.to_csv(payoff_csv, index=False)

    return {
        "mix_final_csv": str(mix_csv),
        "payoff_final_csv": str(payoff_csv),
    }


def plot_transition_fan(
    path: str,
    state: int,
    kernel: np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray],
    next_state_names: List[str],
    as_names: List[str],
    ds_names: List[str],
    out_dir: str,
    dpi: int = 160,
) -> str:
    """
    Plot a 'fan' or stacked bars of P(S'|S,AS_i,DS_j): average over (i,j) weighted by final p*, q*;
    annotate dominant next states; return figure path.

    This visualization explains the Results discussion of transition dynamics and
    how equilibrium mixes steer the edge–cloud system across states.
    """

    if isinstance(kernel, tuple):
        kernel_array, p_star, q_star = kernel
    else:
        kernel_array = kernel
        m = kernel_array.shape[1]
        n = kernel_array.shape[2]
        p_star = np.full(m, 1.0 / m)
        q_star = np.full(n, 1.0 / n)

    if kernel_array.ndim != 3:
        raise ValueError("Kernel must be a 3-D array of shape (S', m, n).")

    if kernel_array.shape[1] != len(as_names) or kernel_array.shape[2] != len(ds_names):
        raise ValueError("Kernel shape does not match strategy dimensions.")

    weights = np.outer(p_star, q_star)
    avg_probs = np.tensordot(kernel_array, weights, axes=([1, 2], [0, 1]))

    if avg_probs.shape[0] != len(next_state_names):
        raise ValueError("Number of next-state names does not match kernel first dimension.")

    out_path = Path(out_dir)
    _ensure_dir(out_path)

    stem = f"{_safe_filename(path)}_state{state}_transition"
    png = out_path / f"{stem}.png"
    pdf = out_path / f"{stem}.pdf"

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=dpi)
    bars = ax.bar(next_state_names, avg_probs, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(next_state_names))))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average Transition Probability")
    ax.set_title(f"Next-State Distribution – {path} / State {state}")
    ax.grid(True, axis="y", alpha=0.3)

    dominant_idx = int(np.argmax(avg_probs))
    for bar, prob in zip(bars, avg_probs):
        ax.annotate(f"{prob:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.annotate(
        f"Dominant: {next_state_names[dominant_idx]} ({avg_probs[dominant_idx]:.2f})",
        xy=(dominant_idx, avg_probs[dominant_idx]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return str(png)


def plot_sensitivity_curves(
    path: str,
    state: int,
    sweeps: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
    dpi: int = 160,
) -> Dict[str, str]:
    """
    For each sweep key (e.g., 'c_star','a','lambda_c','gamma_att'), plot metric vs grid.
    Return dict of created figure paths (may be empty).

    These plots showcase the robustness checks reported in the Results section.
    """

    if not sweeps:
        return {}

    out_path = Path(out_dir)
    _ensure_dir(out_path)

    figure_paths: Dict[str, str] = {}
    for param, payload in sweeps.items():
        grid = payload.get("grid")
        metric = payload.get("metric")
        if grid is None or metric is None:
            continue
        if grid.shape != metric.shape:
            raise ValueError(f"Sensitivity for {param} has grid/metric shape mismatch.")
        stem = f"{_safe_filename(path)}_state{state}_sens_{_safe_filename(param)}"
        png = out_path / f"{stem}.png"
        pdf = out_path / f"{stem}.pdf"

        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=dpi)
        ax.plot(grid, metric, marker="o", linewidth=1.5, color="tab:blue")
        ax.set_title(f"Sensitivity: {param} – {path} / State {state}")
        ax.set_xlabel(param)
        ax.set_ylabel("Metric")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

        figure_paths[param] = str(png)

    return figure_paths


# ---------------------------------------------------------------------------
# Report index
# ---------------------------------------------------------------------------


def build_results_index(
    config: Dict,
    artifact_index: Dict,
    out_dir: str,
) -> str:
    """
    Write results_index.md with per-path/state sections linking all figs/tables; return md path.

    The index provides a single entry point to all Results artifacts for rapid review.
    """

    out_path = Path(out_dir)
    _ensure_dir(out_path)
    md_path = out_path / "results_index.md"

    paths = config.get("paths", [])
    states_per_path = config.get("states_per_path", {})

    lines: List[str] = ["# Results Index", ""]
    for path in paths:
        lines.append(f"## Path: {path}")
        states = states_per_path.get(path, [])
        for state in states:
            key = (path, state)
            fig_entry = artifact_index["figs"].get(key, {})
            table_entry = artifact_index["tables"].get(key, {})

            lines.append(f"### State {state}")
            if fig_entry:
                lines.append("- **Figures:**")
                mix_path = fig_entry.get("mix")
                if mix_path:
                    lines.append(f"  - [Strategy Trajectories]({mix_path})")
                payoff_heat = fig_entry.get("payoff_heatmap")
                if isinstance(payoff_heat, dict):
                    lines.append(f"  - [Attacker Payoff Heatmap]({payoff_heat.get('attacker')})")
                    lines.append(f"  - [Defender Payoff Heatmap]({payoff_heat.get('defender')})")
                elif payoff_heat:
                    lines.append(f"  - [Payoff Heatmap]({payoff_heat})")
                xi_path = fig_entry.get("xi")
                if xi_path:
                    lines.append(f"  - [Lateral Coefficient]({xi_path})")
                theta_path = fig_entry.get("theta")
                if theta_path:
                    lines.append(f"  - [Success Probabilities]({theta_path})")
                transition_path = fig_entry.get("transition")
                if transition_path:
                    lines.append(f"  - [Transition Summary]({transition_path})")
                sensitivity_entry = fig_entry.get("sensitivity", {})
                for param, s_path in sensitivity_entry.items():
                    lines.append(f"  - [Sensitivity – {param}]({s_path})")
            if table_entry:
                lines.append("- **Tables:**")
                for label, t_path in table_entry.items():
                    lines.append(f"  - [{label}]({t_path})")
            lines.append("")

    summary_tex = artifact_index["tables"].get("summary_tex")
    if summary_tex:
        lines.append(f"## Global Summary\n- [Summary Table]({summary_tex})")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(md_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_all_results(
    config: Dict,
    payoff_matrices: Dict[Tuple[str, int], Dict[str, np.ndarray]],
    evolution_history: Dict[Tuple[str, int], Dict],
    state_kernel: Dict[Tuple[str, int], np.ndarray],
    sensitivity: Optional[Dict] = None,
) -> Dict:
    """
    Orchestrate all functions above for every (path,state).
    Ensure deterministic filenames. Return artifact index.
    """

    out_dir = Path(config["out_dir"])
    figs_dir = out_dir / "figs"
    tables_dir = out_dir / "tables"
    report_dir = out_dir / "report"

    for directory in (figs_dir, tables_dir, report_dir):
        _ensure_dir(directory)

    plot_cfg = config.get("plot", {})
    dpi = int(plot_cfg.get("dpi", 160))
    fontsize = plot_cfg.get("fontsize", 11)
    matplotlib.rcParams.update({"font.size": fontsize})

    paths = config.get("paths", [])
    states_per_path = config.get("states_per_path", {})
    as_names = config.get("attacker_strategies", [])
    ds_names = config.get("defender_strategies", [])

    artifact_index: Dict = {"figs": {}, "tables": {}}
    summary_records: List[Dict] = []

    for path in paths:
        for state in states_per_path.get(path, []):
            key = (path, state)

            payoff_entry = payoff_matrices.get(key)
            history_entry = evolution_history.get(key)
            kernel_entry = state_kernel.get(key)

            if payoff_entry is None:
                raise ValueError(f"Missing payoff matrices for {(path, state)}.")
            if history_entry is None:
                raise ValueError(f"Missing evolution history for {(path, state)}.")
            if kernel_entry is None:
                raise ValueError(f"Missing transition kernel for {(path, state)}.")

            payoff_A = np.asarray(payoff_entry["A"])
            payoff_D = np.asarray(payoff_entry["D"])
            _validate_matrix_shapes(payoff_A, payoff_D, as_names, ds_names)
            _validate_history_shapes(history_entry, len(as_names), len(ds_names))

            table_paths = render_payoff_tables(
                path,
                state,
                payoff_A,
                payoff_D,
                as_names,
                ds_names,
                tables_dir.as_posix(),
            )

            fig_mix = plot_strategy_trajectories(
                path,
                state,
                history_entry["p"],
                history_entry["q"],
                as_names,
                ds_names,
                figs_dir.as_posix(),
                dpi=dpi,
            )

            heatmap_paths = plot_payoff_heatmaps(
                path,
                state,
                payoff_A,
                payoff_D,
                as_names,
                ds_names,
                figs_dir.as_posix(),
                dpi=dpi,
            )

            xi_theta_paths = plot_lateral_and_success(
                path,
                state,
                history_entry["xi_m"],
                history_entry["theta"],
                figs_dir.as_posix(),
                dpi=dpi,
            )

            final_tables = summarize_final_mix_and_payoff(
                path,
                state,
                history_entry["p"],
                history_entry["q"],
                history_entry["fa"],
                history_entry["fd"],
                as_names,
                ds_names,
                tables_dir.as_posix(),
            )
            table_paths.update(final_tables)

            next_state_names = config.get("states_per_path", {}).get(path, [])
            transition_path = plot_transition_fan(
                path,
                state,
                (kernel_entry, history_entry["p"][-1], history_entry["q"][-1]),
                next_state_names,
                as_names,
                ds_names,
                figs_dir.as_posix(),
                dpi=dpi,
            )

            sens_bundle: Dict[str, Dict[str, np.ndarray]] = {}
            if sensitivity:
                for param in ("c_star", "a", "lambda_c", "lambda_i", "lambda_a", "gamma_att"):
                    key_sens = (param, path, state)
                    if key_sens in sensitivity:
                        sens_bundle[param] = sensitivity[key_sens]
            sens_paths = plot_sensitivity_curves(
                path,
                state,
                sens_bundle,
                figs_dir.as_posix(),
                dpi=dpi,
            )

            artifact_index["figs"][key] = {
                "mix": fig_mix,
                "payoff_heatmap": {
                    "attacker": heatmap_paths["A_png"],
                    "defender": heatmap_paths["D_png"],
                },
                "xi": xi_theta_paths["xi_png"],
                "theta": xi_theta_paths["theta_png"],
                "transition": transition_path,
                "sensitivity": sens_paths,
            }
            artifact_index["tables"][key] = table_paths

            summary_records.append(
                {
                    "Path": path,
                    "State": state,
                    "Attacker_Argmax": as_names[int(history_entry["p"][-1].argmax())],
                    "Attacker_Prob": history_entry["p"][-1].max(),
                    "Defender_Argmax": ds_names[int(history_entry["q"][-1].argmax())],
                    "Defender_Prob": history_entry["q"][-1].max(),
                    "Attacker_Final_U": history_entry["Ua"][-1],
                    "Defender_Final_U": history_entry["Ud"][-1],
                }
            )

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_tex_path = tables_dir / "summary.tex"
        summary_df.to_latex(
            summary_tex_path,
            index=False,
            escape=True,
            float_format="%.4f",
            bold_rows=False,
            caption="Summary of final mixes and payoffs per path/state.",
        )
        artifact_index["tables"]["summary_tex"] = str(summary_tex_path)

    md_path = build_results_index(config, artifact_index, report_dir.as_posix())
    artifact_index["report"] = md_path

    return artifact_index


# ---------------------------------------------------------------------------
# Demonstration entry point
# ---------------------------------------------------------------------------


def _synthetic_demo() -> None:
    """Run a tiny synthetic example to validate wiring."""

    rng = np.random.default_rng(42)

    config = {
        "out_dir": "demo_results",
        "paths": ["path1"],
        "states_per_path": {"path1": [1, 2]},
        "attacker_strategies": ["AS1", "AS2"],
        "defender_strategies": ["DS1", "DS2", "DS3"],
        "plot": {"dpi": 120, "fontsize": 10},
    }

    payoff_matrices = {}
    evolution_history = {}
    state_kernel = {}
    sensitivity = {}

    for state in config["states_per_path"]["path1"]:
        key = ("path1", state)
        A = rng.normal(size=(2, 3))
        D = rng.normal(size=(2, 3))
        payoff_matrices[key] = {"A": A, "D": D}

        T = 20
        p = rng.dirichlet(np.ones(2), size=T)
        q = rng.dirichlet(np.ones(3), size=T)
        fa = rng.normal(size=(T, 2))
        fd = rng.normal(size=(T, 3))
        Ua = rng.normal(size=T)
        Ud = rng.normal(size=T)
        xi = rng.uniform(0.1, 0.9, size=T)
        theta = {
            "c": rng.uniform(0.3, 0.7, size=T),
            "i": rng.uniform(0.2, 0.8, size=T),
            "a": rng.uniform(0.1, 0.9, size=T),
        }
        evolution_history[key] = {
            "p": p,
            "q": q,
            "fa": fa,
            "fd": fd,
            "Ua": Ua,
            "Ud": Ud,
            "xi_m": xi,
            "theta": theta,
        }

        kernel = rng.uniform(size=(len(config["states_per_path"]["path1"]), 2, 3))
        kernel /= kernel.sum(axis=0, keepdims=True)
        state_kernel[key] = kernel

        grid = np.linspace(0.5, 1.5, 10)
        sensitivity[("c_star", "path1", state)] = {"grid": grid, "metric": np.sin(grid)}
        sensitivity[("gamma_att", "path1", state)] = {"grid": grid, "metric": np.cos(grid)}

    artifacts = generate_all_results(
        config=config,
        payoff_matrices=payoff_matrices,
        evolution_history=evolution_history,
        state_kernel=state_kernel,
        sensitivity=sensitivity,
    )
    print("Generated artifacts:", artifacts["report"])


if __name__ == "__main__":  # pragma: no cover
    _synthetic_demo()
