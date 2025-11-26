#!/usr/bin/env python3
"""Generate additional WF-MTD results without DQN control."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Dataclasses and static configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameParams:
    resource_value: float = 1.0
    mu_y: float = 0.3
    sq: float = 1.0
    k: float = 1.2
    alpha_incentive: float = 0.05
    beta_reg_range: Tuple[float, float] = (0.02, 0.06)
    eta: float = 0.15
    window: int = 10


@dataclass(frozen=True)
class StateDef:
    name: str
    gamma: float
    degree: int
    transitions: Dict[Tuple[int, int], np.ndarray]

    @property
    def xi(self) -> float:
        return self.gamma * float(self.degree)


@dataclass(frozen=True)
class UncertaintyRanges:
    c_star_jitter: float = 0.1
    a_jitter: float = 0.1
    pi_jitter: float = 0.1
    lambda_ranges: Dict[str, Tuple[float, float]] = None
    cost_feature_jitter: float = 0.1
    defender_cost_jitter: float = 0.05

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.lambda_ranges is None:
            object.__setattr__(self, "lambda_ranges", {
                "C": (0.4, 0.6),
                "I": (0.3, 0.5),
                "A": (0.5, 0.7),
            })


ATTACK_STRATEGIES: List[Dict[str, object]] = [
    {"name": "Overflow", "latex": r"AS$_1$ Overflow", "weights": np.array([0.2, 0.6, 0.2], dtype=np.float64), "pi": 1.3, "features": np.array([0.8, 0.7, 0.6, 0.5, 0.3], dtype=np.float64)},
    {"name": "DestroyData", "latex": r"AS$_2$ DestroyData", "weights": np.array([0.1, 0.7, 0.2], dtype=np.float64), "pi": 1.5, "features": np.array([1.1, 0.9, 1.2, 0.8, 0.6], dtype=np.float64)},
    {"name": "Recon+Lateral", "latex": r"AS$_3$ Recon+Lateral", "weights": np.array([0.4, 0.3, 0.3], dtype=np.float64), "pi": 1.1, "features": np.array([0.9, 0.8, 0.9, 0.6, 0.4], dtype=np.float64)},
]
ATTACK_COST_WEIGHTS = np.array([0.3, 0.2, 0.2, 0.15, 0.15], dtype=np.float64)

DEFENDER_STRATEGIES: List[Dict[str, object]] = [
    {"name": "ASD1_IPHop", "latex": r"DS$_1$ ASD1\\_IPHop", "c_star": 1.0, "a": 0.9, "assc": 0.9, "aic": 0.35},
    {"name": "ASD1+ASD3_IP+ProtoHop", "latex": r"DS$_2$ ASD1+ASD3", "c_star": 1.6, "a": 1.2, "assc": 1.25, "aic": 0.45},
    {"name": "ASD1+TimeRand", "latex": r"DS$_3$ ASD1+TimeRand", "c_star": 1.2, "a": 1.0, "assc": 1.05, "aic": 0.4},
]

ATTRIBUTE_NAMES = ("C", "I", "A")
ATTRIBUTE_VALUES = np.array([0.8, 1.0, 0.9], dtype=np.float64)

STATE_DEFS: Dict[str, StateDef] = {
    "S1": StateDef(
        name="S1",
        gamma=0.2,
        degree=2,
        transitions={
            (0, 1): np.array([0.7, 0.3], dtype=np.float64),
            (1, 1): np.array([0.55, 0.45], dtype=np.float64),
            (2, 1): np.array([0.6, 0.4], dtype=np.float64),
        },
    ),
    "S2": StateDef(
        name="S2",
        gamma=0.15,
        degree=3,
        transitions={
            (1, 1): np.array([0.6, 0.4], dtype=np.float64),
            (1, 0): np.array([0.4, 0.6], dtype=np.float64),
            (2, 1): np.array([0.45, 0.55], dtype=np.float64),
        },
    ),
}


# ---------------------------------------------------------------------------
# Robust payoff computation
# ---------------------------------------------------------------------------


def _sample_dirichlet(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gamma = rng.gamma(alpha, 1.0)
    total = float(np.sum(gamma))
    if total <= 0.0:
        return np.full_like(alpha, 1.0 / alpha.size)
    return gamma / total


def sample_uncertainty(rng: np.random.Generator, ranges: UncertaintyRanges, as_idx: int, ds_idx: int) -> Dict[str, object]:
    attack = ATTACK_STRATEGIES[as_idx]
    defend = DEFENDER_STRATEGIES[ds_idx]

    scenario = {
        "c_star": defend["c_star"] * rng.uniform(1.0 - ranges.c_star_jitter, 1.0 + ranges.c_star_jitter),
        "a_param": defend["a"] * rng.uniform(1.0 - ranges.a_jitter, 1.0 + ranges.a_jitter),
        "pi_i": attack["pi"] * rng.uniform(1.0 - ranges.pi_jitter, 1.0 + ranges.pi_jitter),
        "lambda_vec": np.array([rng.uniform(*ranges.lambda_ranges[attr]) for attr in ATTRIBUTE_NAMES], dtype=np.float64),
        "weights": _sample_dirichlet(np.asarray(attack["weights"], dtype=np.float64) * 50.0, rng),
        "features": np.asarray(attack["features"], dtype=np.float64) * rng.uniform(
            1.0 - ranges.cost_feature_jitter,
            1.0 + ranges.cost_feature_jitter,
            size=len(attack["features"]),
        ),
        "assc": defend["assc"] * rng.uniform(1.0 - ranges.defender_cost_jitter, 1.0 + ranges.defender_cost_jitter),
        "aic": defend["aic"] * rng.uniform(1.0 - ranges.defender_cost_jitter, 1.0 + ranges.defender_cost_jitter),
        "beta_reg": rng.uniform(*GameParams().beta_reg_range),
    }
    return scenario


def _beta_value(c_star: float, a_param: float, pi_i: float) -> float:
    argument = -c_star * a_param * pi_i
    numerator = 1.0 - np.exp(argument)
    denominator = 1.0 + np.exp(argument)
    beta = numerator / denominator if denominator != 0 else 0.0
    return float(np.clip(beta, 0.0, 1.0 - 1e-9))


def compute_payoffs_under_scenario(state: StateDef, scenario: Dict[str, object], params: GameParams) -> Tuple[float, float]:
    beta = _beta_value(scenario["c_star"], scenario["a_param"], scenario["pi_i"])
    lambda_vec = np.asarray(scenario["lambda_vec"], dtype=np.float64)
    weights = np.asarray(scenario["weights"], dtype=np.float64)
    theta = np.clip(1.0 - lambda_vec * beta, 0.0, 1.0)

    sal = (1.0 + state.xi) * params.resource_value * float(np.sum(theta * weights * ATTRIBUTE_VALUES))
    ac = float(np.dot(ATTACK_COST_WEIGHTS, np.asarray(scenario["features"], dtype=np.float64)))
    penalty = scenario["beta_reg"] * float(np.mean(theta))
    attacker_payoff = sal - ac - penalty

    sap_terms = (params.mu_y * theta + (1.0 - theta)) * (1.0 - weights) * ATTRIBUTE_VALUES
    sap = max(0.0, 1.0 - state.xi) * params.resource_value * float(np.sum(sap_terms))

    nc = params.sq * (1.0 - 1.0 / (1.0 + np.exp(-(scenario["a_param"] - params.k))))
    defender_payoff = sap - (scenario["assc"] + nc + scenario["aic"]) + params.alpha_incentive
    return attacker_payoff, defender_payoff


def robust_stage_entries(state: StateDef, rng: np.random.Generator, samples: int, params: GameParams, ranges: UncertaintyRanges) -> Tuple[np.ndarray, np.ndarray]:
    n_att = len(ATTACK_STRATEGIES)
    n_def = len(DEFENDER_STRATEGIES)
    A = np.empty((n_att, n_def), dtype=np.float64)
    B = np.empty((n_att, n_def), dtype=np.float64)

    for i in range(n_att):
        for j in range(n_def):
            attacker_vals: List[float] = []
            defender_vals: List[float] = []
            for _ in range(samples):
                scenario = sample_uncertainty(rng, ranges, i, j)
                u_a, u_d = compute_payoffs_under_scenario(state, scenario, params)
                attacker_vals.append(u_a)
                defender_vals.append(u_d)
            A[i, j] = float(np.min(attacker_vals))
            B[i, j] = float(np.min(defender_vals))
    return A, B


# ---------------------------------------------------------------------------
# Wright–Fisher dynamics
# ---------------------------------------------------------------------------


def wf_step(p: np.ndarray, q: np.ndarray, A: np.ndarray, B: np.ndarray, omega_a: float, omega_d: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    f_as = A @ q
    f_ds = B.T @ p

    F_as = (1.0 - omega_a) + omega_a * f_as
    F_ds = (1.0 - omega_d) + omega_d * f_ds

    denom_as = float(np.dot(p, F_as))
    denom_ds = float(np.dot(q, F_ds))
    if denom_as <= 0.0:
        denom_as = 1.0
    if denom_ds <= 0.0:
        denom_ds = 1.0

    growth_p = p * (F_as / denom_as)
    growth_q = q * (F_ds / denom_ds)

    p_new = _project_simplex(p + eta * (growth_p - p))
    q_new = _project_simplex(q + eta * (growth_q - q))
    return p_new, q_new


def _project_simplex(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.maximum(probabilities, 1e-12)
    total = float(np.sum(clipped))
    if total <= 0.0:
        return np.full_like(clipped, 1.0 / clipped.size)
    return clipped / total


def simulate(state: StateDef, A: np.ndarray, B: np.ndarray, p0: np.ndarray, q0: np.ndarray, episodes: int, omega_a: float, omega_d: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    p = p0.copy()
    q = q0.copy()
    p_hist = np.zeros((episodes, len(p)), dtype=np.float64)
    q_hist = np.zeros((episodes, len(q)), dtype=np.float64)
    for t in range(episodes):
        p_hist[t] = p
        q_hist[t] = q
        p, q = wf_step(p, q, A, B, omega_a, omega_d, eta)
    return p_hist, q_hist


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------


def detect_convergence(prob_history: np.ndarray, window: int) -> Tuple[Optional[int], Optional[int]]:
    if prob_history.shape[0] < window:
        return None, None
    argmax_seq = np.argmax(prob_history, axis=1)
    for start in range(prob_history.shape[0] - window + 1):
        block = argmax_seq[start : start + window]
        if np.all(block == block[0]):
            return start, int(block[0])
    return None, None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _configure_matplotlib() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.5,
        "mathtext.fontset": "stix",
        "font.family": "sans-serif",
    })


def _annotate(ax: plt.Axes, iteration: Optional[int], strategy: Optional[str], series: np.ndarray) -> None:
    if iteration is None or strategy is None:
        return
    idx = min(iteration, len(series) - 1)
    ax.annotate(
        f"{strategy} dominant @ t={iteration + 1}",
        xy=(iteration, series[idx]),
        xytext=(iteration + 5, min(1.0, series[idx] + 0.08)),
        arrowprops=dict(arrowstyle="->", color=ax.lines[-1].get_color()),
        fontsize=10,
    )


def plot_state_evolution(
    out_dir: Path,
    filename: str,
    state_name: str,
    p_hist: np.ndarray,
    q_hist: np.ndarray,
    conv_p: Tuple[Optional[int], Optional[int]],
    conv_q: Tuple[Optional[int], Optional[int]],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    steps = np.arange(p_hist.shape[0])

    for idx, attack in enumerate(ATTACK_STRATEGIES):
        axes[0].plot(steps, p_hist[:, idx], label=attack["latex"])
    if conv_p[0] is not None and conv_p[1] is not None:
        _annotate(axes[0], conv_p[0], ATTACK_STRATEGIES[conv_p[1]]["name"], p_hist[:, conv_p[1]])
    axes[0].set_ylabel("Attacker probability")
    axes[0].set_title(f"Evolution trajectories (State {state_name}, Path 1)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for idx, defend in enumerate(DEFENDER_STRATEGIES):
        axes[1].plot(steps, q_hist[:, idx], label=defend["latex"])
    if conv_q[0] is not None and conv_q[1] is not None:
        _annotate(axes[1], conv_q[0], DEFENDER_STRATEGIES[conv_q[1]]["name"], q_hist[:, conv_q[1]])
    axes[1].set_ylabel("Defender probability")
    axes[1].set_xlabel("Games (episodes)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def save_state_evolution_csv(
    out_dir: Path,
    filename: str,
    p_hist: np.ndarray,
    q_hist: np.ndarray,
) -> None:
    path = out_dir / filename
    header = ["iteration"]
    header.extend(f"p_{ATTACK_STRATEGIES[i]['name']}" for i in range(len(ATTACK_STRATEGIES)))
    header.extend(f"q_{DEFENDER_STRATEGIES[j]['name']}" for j in range(len(DEFENDER_STRATEGIES)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for t in range(p_hist.shape[0]):
            row = [t + 1]
            row.extend(map(float, p_hist[t]))
            row.extend(map(float, q_hist[t]))
            writer.writerow(row)


def plot_init_sensitivity(
    out_dir: Path,
    filename: str,
    state_cases: Dict[str, List[Dict[str, object]]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, state_name in zip(axes, ["S1", "S2"]):
        for entry in state_cases[state_name]:
            series = np.asarray(entry["series"], dtype=np.float64)
            ax.plot(np.arange(len(series)), series, label=entry["label"])
            if entry["conv_iter"] is not None:
                _annotate(ax, entry["conv_iter"], entry["dominant"], series)
        ax.set_title(f"Init-probability sensitivity ({state_name})")
        ax.set_xlabel("Games (episodes)")
        ax.set_ylabel("Dominant DS probability")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def save_init_sensitivity_csv(
    out_dir: Path,
    filename: str,
    state_cases: Dict[str, List[Dict[str, object]]],
) -> None:
    path = out_dir / filename
    header = ["state", "case", "dominant_strategy", "iteration", "probability"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for state_name, entries in state_cases.items():
            for entry in entries:
                series = np.asarray(entry["series"], dtype=np.float64)
                for idx, value in enumerate(series, start=1):
                    writer.writerow(
                        [
                            state_name,
                            entry["label"],
                            entry["dominant"],
                            idx,
                            float(value),
                        ]
                    )


def plot_rationality_sweep(
    out_dir: Path,
    filename: str,
    entries: List[Dict[str, object]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for entry in entries:
        series = np.asarray(entry["series"], dtype=np.float64)
        ax.plot(np.arange(len(series)), series, label=entry["label"])
        if entry["conv_iter"] is not None:
            _annotate(ax, entry["conv_iter"], entry["dominant"], series)
    ax.set_title("Effect of rationality (State S1)")
    ax.set_xlabel("Games (episodes)")
    ax.set_ylabel("max$_j$ q$_j$(t)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def save_rationality_sweep_csv(
    out_dir: Path,
    filename: str,
    entries: List[Dict[str, object]],
) -> None:
    path = out_dir / filename
    header = ["omega", "iteration", "max_q", "dominant_strategy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for entry in entries:
            omega_label = entry["label"]
            omega_value = omega_label.split("=")[-1]
            series = np.asarray(entry["series"], dtype=np.float64)
            for idx, value in enumerate(series, start=1):
                writer.writerow([omega_value, idx, float(value), entry["dominant"]])


def plot_bounded_vs_full(
    out_dir: Path,
    filename: str,
    bounded: Dict[str, object],
    full: Dict[str, object],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bounded_q = np.asarray(bounded["q_series"], dtype=np.float64)
    bounded_p = np.asarray(bounded["p_series"], dtype=np.float64)
    full_q = np.asarray(full["q_series"], dtype=np.float64)
    full_p = np.asarray(full["p_series"], dtype=np.float64)

    ax.plot(np.arange(len(bounded_q)), bounded_q, label="Bounded: max$_j$ q$_j$", color="tab:blue")
    ax.plot(np.arange(len(bounded_p)), bounded_p, linestyle="--", label="Bounded: max$_i$ p$_i$", color="tab:blue")
    ax.plot(np.arange(len(full_q)), full_q, label="Full: max$_j$ q$_j$", color="tab:orange")
    ax.plot(np.arange(len(full_p)), full_p, linestyle="--", label="Full: max$_i$ p$_i$", color="tab:orange")

    if bounded["conv_q"] is not None:
        _annotate(ax, bounded["conv_q"], bounded["dominant"], bounded_q)
    if full["conv_q"] is not None:
        _annotate(ax, full["conv_q"], full["dominant"], full_q)

    ax.set_title("Convergence speed: bounded vs fully rational")
    ax.set_xlabel("Games (episodes)")
    ax.set_ylabel("Probability")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def save_bounded_vs_full_csv(
    out_dir: Path,
    filename: str,
    bounded: Dict[str, object],
    full: Dict[str, object],
) -> None:
    path = out_dir / filename
    header = ["mode", "metric", "iteration", "value", "dominant_strategy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for mode_name, entry in (("bounded", bounded), ("full", full)):
            q_series = np.asarray(entry["q_series"], dtype=np.float64)
            p_series = np.asarray(entry["p_series"], dtype=np.float64)
            for idx, value in enumerate(q_series, start=1):
                writer.writerow([mode_name, "max_q", idx, float(value), entry["dominant"]])
            for idx, value in enumerate(p_series, start=1):
                writer.writerow([mode_name, "max_p", idx, float(value), entry["dominant"]])


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _final_distribution(hist: np.ndarray) -> List[float]:
    return hist[-1].round(6).tolist()


def _dominant_name(index: Optional[int], strategies: Sequence[Dict[str, object]]) -> Optional[str]:
    if index is None:
        return None
    return str(strategies[index]["name"])


# ---------------------------------------------------------------------------
# Master generation routine
# ---------------------------------------------------------------------------


def generate_additional_results(out_dir: str, episodes: int, robust_samples: int, eta: float = 0.15, omega: float = 0.5, window: int = 10) -> Dict[str, object]:
    _configure_matplotlib()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(12345)
    params = GameParams(eta=eta, window=window)
    ranges = UncertaintyRanges()

    robust_entries = {
        state_name: robust_stage_entries(state_def, rng, robust_samples, params, ranges)
        for state_name, state_def in STATE_DEFS.items()
    }

    summary: Dict[str, object] = {}

    uniform_att = np.full(len(ATTACK_STRATEGIES), 1.0 / len(ATTACK_STRATEGIES))
    uniform_def = np.full(len(DEFENDER_STRATEGIES), 1.0 / len(DEFENDER_STRATEGIES))

    for fig_key, state_name, filename in [
        ("fig3", "S1", "fig3_state1_evolution.png"),
        ("fig4", "S2", "fig4_state2_evolution.png"),
    ]:
        A, B = robust_entries[state_name]
        state_def = STATE_DEFS[state_name]
        p_hist, q_hist = simulate(state_def, A, B, uniform_att, uniform_def, episodes, omega, omega, params.eta)
        conv_p = detect_convergence(p_hist, window)
        conv_q = detect_convergence(q_hist, window)
        plot_state_evolution(output, filename, state_name, p_hist, q_hist, conv_p, conv_q)
        save_state_evolution_csv(output, filename.replace(".png", ".csv"), p_hist, q_hist)
        summary[fig_key] = {
            "state": state_name,
            "attacker": {"dominant": _dominant_name(conv_p[1], ATTACK_STRATEGIES), "iteration": None if conv_p[0] is None else int(conv_p[0])},
            "defender": {"dominant": _dominant_name(conv_q[1], DEFENDER_STRATEGIES), "iteration": None if conv_q[0] is None else int(conv_q[0])},
            "final": {"p": _final_distribution(p_hist), "q": _final_distribution(q_hist)},
        }

    init_cases = [
        (np.array([0.8, 0.1, 0.1], dtype=np.float64), "q0=[0.8,0.1,0.1]"),
        (np.array([0.1, 0.8, 0.1], dtype=np.float64), "q0=[0.1,0.8,0.1]"),
        (np.array([0.1, 0.1, 0.8], dtype=np.float64), "q0=[0.1,0.1,0.8]"),
    ]
    sensitivity: Dict[str, List[Dict[str, object]]] = {"S1": [], "S2": []}
    for state_name in ("S1", "S2"):
        A, B = robust_entries[state_name]
        state_def = STATE_DEFS[state_name]
        entries: List[Dict[str, object]] = []
        for q0, label in init_cases:
            p_hist, q_hist = simulate(state_def, A, B, uniform_att, q0, episodes, omega, omega, params.eta)
            conv_q = detect_convergence(q_hist, window)
            dominant_idx = conv_q[1] if conv_q[1] is not None else int(np.argmax(q_hist[-1]))
            entries.append({
                "label": f"{label} → {DEFENDER_STRATEGIES[dominant_idx]['name']}",
                "series": q_hist[:, dominant_idx].tolist(),
                "conv_iter": None if conv_q[0] is None else int(conv_q[0]),
                "dominant": DEFENDER_STRATEGIES[dominant_idx]["name"],
            })
        sensitivity[state_name] = entries
    plot_init_sensitivity(output, "fig5_init_prob_sensitivity.png", sensitivity)
    save_init_sensitivity_csv(output, "fig5_init_prob_sensitivity.csv", sensitivity)
    summary["fig5"] = sensitivity

    sweep_entries: List[Dict[str, object]] = []
    A_S1, B_S1 = robust_entries["S1"]
    for omega_val in (0.2, 0.5, 0.9):
        p_hist, q_hist = simulate(STATE_DEFS["S1"], A_S1, B_S1, uniform_att, uniform_def, episodes, omega_val, omega_val, params.eta)
        conv_q = detect_convergence(q_hist, window)
        dominant_idx = conv_q[1] if conv_q[1] is not None else int(np.argmax(q_hist[-1]))
        sweep_entries.append({
            "label": f"ω={omega_val:.1f}",
            "series": np.max(q_hist, axis=1).tolist(),
            "conv_iter": None if conv_q[0] is None else int(conv_q[0]),
            "dominant": DEFENDER_STRATEGIES[dominant_idx]["name"],
        })
    plot_rationality_sweep(output, "fig6_rationality_sweep.png", sweep_entries)
    save_rationality_sweep_csv(output, "fig6_rationality_sweep.csv", sweep_entries)
    summary["fig6"] = sweep_entries

    bounded_hist = simulate(STATE_DEFS["S1"], A_S1, B_S1, uniform_att, uniform_def, episodes, omega, omega, params.eta)
    full_hist = simulate(STATE_DEFS["S1"], A_S1, B_S1, uniform_att, uniform_def, episodes, 1.0, 1.0, params.eta)

    def collect_entry(hist_pair: Tuple[np.ndarray, np.ndarray]) -> Dict[str, object]:
        p_hist, q_hist = hist_pair
        conv_q = detect_convergence(q_hist, window)
        dominant_idx = conv_q[1] if conv_q[1] is not None else int(np.argmax(q_hist[-1]))
        return {
            "q_series": np.max(q_hist, axis=1).tolist(),
            "p_series": np.max(p_hist, axis=1).tolist(),
            "conv_q": None if conv_q[0] is None else int(conv_q[0]),
            "dominant": DEFENDER_STRATEGIES[dominant_idx]["name"],
        }

    bounded_entry = collect_entry(bounded_hist)
    full_entry = collect_entry(full_hist)
    plot_bounded_vs_full(output, "fig7_bounded_vs_full.png", bounded_entry, full_entry)
    save_bounded_vs_full_csv(output, "fig7_bounded_vs_full.csv", bounded_entry, full_entry)
    summary["fig7"] = {"bounded": bounded_entry, "full": full_entry}

    json_path = output / "summary_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Additional WF-MTD results written to", output)
    for key in ("fig3", "fig4"):
        info = summary[key]
        print(f"{key}: attacker→{info['attacker']['dominant']}@t={info['attacker']['iteration']}  defender→{info['defender']['dominant']}@t={info['defender']['iteration']}")
    for state_name, entries in summary["fig5"].items():
        for entry in entries:
            print(f"Fig5 {state_name} {entry['label']} converges to {entry['dominant']} @t={entry['conv_iter']}")
    for entry in summary["fig6"]:
        print(f"Fig6 {entry['label']} dominant {entry['dominant']} @t={entry['conv_iter']}")
    print("Fig7 bounded dominant:", summary["fig7"]["bounded"]["dominant"], "t=", summary["fig7"]["bounded"]["conv_q"])
    print("Fig7 full dominant:", summary["fig7"]["full"]["dominant"], "t=", summary["fig7"]["full"]["conv_q"])
    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate additional WF-MTD figures without the DQN controller")
    parser.add_argument("--out", dest="out", default="figs", help="Output directory")
    parser.add_argument("--episodes", type=int, default=120, help="Number of WF iterations")
    parser.add_argument("--robust_samples", type=int, default=64, help="Samples per payoff entry")
    parser.add_argument("--eta", type=float, default=0.15, help="WF step size")
    parser.add_argument("--omega", type=float, default=0.5, help="Baseline bounded-rational weight")
    parser.add_argument("--window", type=int, default=10, help="Convergence stability window")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_cli(argv)
    generate_additional_results(
        out_dir=args.out,
        episodes=args.episodes,
        robust_samples=args.robust_samples,
        eta=args.eta,
        omega=args.omega,
        window=args.window,
    )


if __name__ == "__main__":
    main()
