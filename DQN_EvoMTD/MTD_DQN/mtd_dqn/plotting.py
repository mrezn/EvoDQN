from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir, project_to_simplex
from .wf_mtd import WrightFisherCoupler


def _load_stage_matrices(log_path: Path, n_att: int, n_def: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(log_path)
    a_matrix = np.zeros((n_att, n_def), dtype=float)
    b_matrix = np.zeros((n_att, n_def), dtype=float)
    for i in range(n_att):
        for j in range(n_def):
            mask = (df["attacker_idx"] == i) & (df["defender_idx"] == j)
            subset = df.loc[mask]
            if not subset.empty:
                a_matrix[i, j] = subset["attacker_proxy"].mean()
                b_matrix[i, j] = subset["defender_proxy"].mean()
    return a_matrix, b_matrix


def _simulate(coupler: WrightFisherCoupler, a_matrix: np.ndarray, b_matrix: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    p_hist = []
    q_hist = []
    for _ in range(steps):
        p_hist.append(coupler.mix_attacker.copy())
        q_hist.append(coupler.mix_defender.copy())
        coupler.update(a_matrix, b_matrix)
    return np.array(p_hist), np.array(q_hist)


def _new_coupler(template: WrightFisherCoupler, initial_p: Sequence[float], initial_q: Sequence[float]) -> WrightFisherCoupler:
    coupler = WrightFisherCoupler(
        attacker_strategies=template.attacker_strategies,
        defender_strategies=template.defender_strategies,
        omega_attacker=template.omega_attacker,
        omega_defender=template.omega_defender,
        eta=template.eta,
    )
    coupler.mix_attacker = project_to_simplex(initial_p)
    coupler.mix_defender = project_to_simplex(initial_q)
    return coupler


def plot_category_a(
    base_coupler: WrightFisherCoupler,
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    steps = 40
    coupler1 = _new_coupler(base_coupler, base_coupler.mix_attacker, base_coupler.mix_defender)
    p_hist1, q_hist1 = _simulate(coupler1, a_matrix, b_matrix, steps)

    attenuated = b_matrix * 0.95
    coupler2 = _new_coupler(base_coupler, base_coupler.mix_attacker, base_coupler.mix_defender)
    p_hist2, q_hist2 = _simulate(coupler2, a_matrix, attenuated, steps)

    attacker_cols = [f"AS{idx+1}" for idx in range(p_hist1.shape[1])]
    defender_cols = [f"DS{idx+1}" for idx in range(q_hist1.shape[1])]
    df_state1_att = pd.DataFrame(p_hist1, columns=attacker_cols)
    df_state1_att.insert(0, "iteration", np.arange(len(p_hist1)))
    df_state1_att.to_csv(output_dir / "category_a_state1_attacker_mix.csv", index=False)
    df_state1_def = pd.DataFrame(q_hist1, columns=defender_cols)
    df_state1_def.insert(0, "iteration", np.arange(len(q_hist1)))
    df_state1_def.to_csv(output_dir / "category_a_state1_defender_mix.csv", index=False)

    df_state2_att = pd.DataFrame(p_hist2, columns=attacker_cols)
    df_state2_att.insert(0, "iteration", np.arange(len(p_hist2)))
    df_state2_att.to_csv(output_dir / "category_a_state2_attacker_mix.csv", index=False)
    df_state2_def = pd.DataFrame(q_hist2, columns=defender_cols)
    df_state2_def.insert(0, "iteration", np.arange(len(q_hist2)))
    df_state2_def.to_csv(output_dir / "category_a_state2_defender_mix.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for idx, strategy in enumerate(base_coupler.attacker_strategies):
        axes[0].plot(p_hist1[:, idx], label=f"AS{idx+1}: {strategy}")
        axes[1].plot(p_hist2[:, idx], label=f"AS{idx+1}: {strategy}")
    axes[0].set_title("Category A  Path 1 State 1 mixes")
    axes[1].set_title("Category A  Path 1 State 2 mixes")
    axes[1].set_xlabel("Replicator iteration")
    for ax, data in zip(axes, (q_hist1, q_hist2)):
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    png_path = output_dir / "category_a_mix.png"
    pdf_path = output_dir / "category_a_mix.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_category_b(
    base_coupler: WrightFisherCoupler,
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    inits = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ]
    steps = 40
    fig, axes = plt.subplots(len(inits), 1, figsize=(8, 9), sharex=True)
    csv_frames: List[pd.DataFrame] = []
    for ax, init in zip(axes, inits):
        coupler = _new_coupler(base_coupler, base_coupler.mix_attacker, init)
        _, q_hist = _simulate(coupler, a_matrix, b_matrix, steps)
        df = pd.DataFrame(q_hist, columns=[f"DS{idx+1}" for idx in range(q_hist.shape[1])])
        df.insert(0, "iteration", np.arange(len(q_hist)))
        df["initial_mix"] = ",".join(map(str, init))
        csv_frames.append(df)
        for idx, strategy in enumerate(base_coupler.defender_strategies):
            ax.plot(q_hist[:, idx], label=f"DS{idx+1}: {strategy}")
        ax.set_ylabel("Probability")
        ax.set_title(f"Initial defender mix {init}")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Replicator iteration")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    png_path = output_dir / "category_b_init_effects.png"
    pdf_path = output_dir / "category_b_init_effects.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    if csv_frames:
        pd.concat(csv_frames, ignore_index=True).to_csv(output_dir / "category_b_defender_mix.csv", index=False)


def plot_category_c(
    base_coupler: WrightFisherCoupler,
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    omegas = [0.2, 0.5, 0.8, 1.0]
    steps = 40
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline_coupler = WrightFisherCoupler(
        attacker_strategies=base_coupler.attacker_strategies,
        defender_strategies=base_coupler.defender_strategies,
        omega_attacker=base_coupler.omega_attacker,
        omega_defender=base_coupler.omega_defender,
        eta=base_coupler.eta,
    )
    for omega in omegas:
        baseline_coupler.omega_attacker = omega
        baseline_coupler.omega_defender = omega
        run_coupler = _new_coupler(baseline_coupler, baseline_coupler.mix_attacker, baseline_coupler.mix_defender)
        _, q_hist = _simulate(run_coupler, a_matrix, b_matrix, steps)
        deviations = np.linalg.norm(q_hist - q_hist[-1], ord=1, axis=1)
        ax.plot(deviations, label=f"?={omega}")
        pd.DataFrame(
            {"iteration": np.arange(len(deviations)), "deviation": deviations, "omega": omega}
        ).to_csv(output_dir / f"category_c_convergence_omega_{omega:.1f}.csv", index=False)
    ax.set_title("Category C Convergence vs bounded rationality")
    ax.set_xlabel("Replicator iteration")
    ax.set_ylabel("L1 deviation from equilibrium")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    png_path = output_dir / "category_c_rationality.png"
    pdf_path = output_dir / "category_c_rationality.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_category_d(
    base_coupler: WrightFisherCoupler,
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    steps = 40
    bounded = _new_coupler(base_coupler, base_coupler.mix_attacker, base_coupler.mix_defender)
    bounded_hist = _simulate(bounded, a_matrix, b_matrix, steps)[1]

    rational = WrightFisherCoupler(
        attacker_strategies=base_coupler.attacker_strategies,
        defender_strategies=base_coupler.defender_strategies,
        omega_attacker=1.0,
        omega_defender=1.0,
        eta=base_coupler.eta,
    )
    rational.mix_attacker = base_coupler.mix_attacker.copy()
    rational.mix_defender = base_coupler.mix_defender.copy()
    rational_hist = _simulate(rational, a_matrix, b_matrix, steps)[1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.max(bounded_hist, axis=1), label="Bounded rational")
    ax.plot(np.max(rational_hist, axis=1), label="Fully rational", linestyle="--")
    ax.set_title("Category D Method vs full rational baseline")
    ax.set_xlabel("Replicator iteration")
    ax.set_ylabel("Best defender strategy probability")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    png_path = output_dir / "category_d_comparison.png"
    pdf_path = output_dir / "category_d_comparison.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_training_convergence(
    logs_path: Path,
    output_dir: Path,
    dpi: int,
    style: str = "default",
) -> Optional[Path]:
    """Generate loss/performance convergence chart from training logs."""
    if not logs_path.exists():
        return None

    plt.style.use(style)
    output_dir = ensure_dir(output_dir)

    df = pd.read_csv(logs_path)
    if "loss" not in df.columns:
        return None

    df = df.copy()
    df["iteration"] = np.arange(len(df))

    loss_series = df["loss"].replace([np.inf, -np.inf], np.nan).dropna()
    if loss_series.empty:
        return None

    df["loss_smoothed"] = df["loss"].rolling(window=25, min_periods=1).mean()

    payoff_col = "defender_payoff" if "defender_payoff" in df.columns else None
    if payoff_col:
        payoff = df[payoff_col].rolling(window=25, min_periods=1).mean()
        payoff_min = payoff.min()
        payoff_max = payoff.max()
        if payoff_max - payoff_min > 1e-9:
            payoff_norm = (payoff - payoff_min) / (payoff_max - payoff_min)
        else:
            payoff_norm = payoff * 0.0 + 0.5
        df["payoff_norm"] = payoff_norm

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["iteration"], df["loss_smoothed"], color="tab:blue", label="TD loss (smoothed)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("TD loss")
    ax1.grid(True, linestyle="--", alpha=0.4)

    if payoff_col:
        ax2 = ax1.twinx()
        ax2.plot(df["iteration"], df["payoff_norm"], color="tab:orange", label="Defender payoff (normalised)")
        ax2.set_ylabel("Normalised payoff")
        ax2.set_ylim(0, 1)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    fig.suptitle("DQN Training Loss and Performance Convergence")
    fig.tight_layout()

    png_path = output_dir / "training_convergence.png"
    pdf_path = output_dir / "training_convergence.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    export_cols = ["iteration", "loss_smoothed"]
    if payoff_col:
        export_cols.append("payoff_norm")
    df[export_cols].to_csv(output_dir / "training_convergence.csv", index=False)
    return png_path


def plot_all(
    config: dict,
    output_dirs: Dict[str, Path],
    coupler: WrightFisherCoupler,
    train_logs: Path,
) -> None:
    plot_cfg = config.get("plotting", {})
    dpi = int(plot_cfg.get("dpi", 200))
    style = plot_cfg.get("style", "default")
    plt.style.use(style)

    figures_dir = ensure_dir(output_dirs["figures"])
    a_matrix, b_matrix = _load_stage_matrices(train_logs, len(coupler.attacker_strategies), len(coupler.defender_strategies))

    plot_category_a(coupler, a_matrix, b_matrix, figures_dir, dpi)
    plot_category_b(coupler, a_matrix, b_matrix, figures_dir, dpi)
    plot_category_c(coupler, a_matrix, b_matrix, figures_dir, dpi)
    plot_category_d(coupler, a_matrix, b_matrix, figures_dir, dpi)

