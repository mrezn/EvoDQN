"""Plotting utilities for WF-MTD simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from evolution import EvolutionLog
from model_types import ModelConfig


def _prepare_arrays(logs: Iterable[EvolutionLog]):
    """Convert evolution logs into numpy arrays for plotting."""

    logs = list(logs)
    if not logs:
        raise ValueError("No evolution logs to plot.")
    steps = np.array([log.step for log in logs])
    attacker_mix = np.array([log.attacker_mix for log in logs])
    defender_mix = np.array([log.defender_mix for log in logs])
    attacker_payoff = np.array([log.attacker_payoff for log in logs])
    defender_payoff = np.array([log.defender_payoff for log in logs])
    rho = np.array([log.rho for log in logs])
    return steps, attacker_mix, defender_mix, attacker_payoff, defender_payoff, rho


def plot_trajectories(logs: Iterable[EvolutionLog], config: ModelConfig, path: Path | None = None) -> None:
    steps, attacker_mix, defender_mix, attacker_payoff, defender_payoff, rho = _prepare_arrays(logs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, name in enumerate(config.strategies.attack):
        axes[0, 0].plot(steps, attacker_mix[:, idx], label=name)
    axes[0, 0].set_title("Attacker mix")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Probability")
    axes[0, 0].legend()

    for idx, name in enumerate(config.strategies.defense):
        axes[0, 1].plot(steps, defender_mix[:, idx], label=name)
    axes[0, 1].set_title("Defender mix")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Probability")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, attacker_payoff, label="Attacker", color="crimson")
    axes[1, 0].plot(steps, defender_payoff, label="Defender", color="navy")
    axes[1, 0].set_title("Stage payoffs")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Payoff")
    axes[1, 0].legend()

    for idx, state in enumerate(config.kernel.states):
        axes[1, 1].plot(steps, rho[:, idx], label=state)
    axes[1, 1].set_title("State distribution")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].legend()

    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    else:
        plt.show()
    plt.close(fig)


def plot_strategy_probabilities(
    logs: Iterable[EvolutionLog],
    config: ModelConfig,
    path: Path | None = None,
) -> None:
    """Plot attacker and defender strategy probabilities for each stage.

    This helper focuses on the Wright-Fisher mixes, drawing a dedicated
    subplot for each side so per-stage trajectories are easy to inspect.
    """

    steps, attacker_mix, defender_mix, *_ = _prepare_arrays(logs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    for idx, name in enumerate(config.strategies.attack):
        axes[0].plot(steps, attacker_mix[:, idx], label=name)
    axes[0].set_title("Attacker strategy probabilities")
    axes[0].set_xlabel("Stage")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    for idx, name in enumerate(config.strategies.defense):
        axes[1].plot(steps, defender_mix[:, idx], label=name)
    axes[1].set_title("Defender strategy probabilities")
    axes[1].set_xlabel("Stage")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    else:
        plt.show()
    plt.close(fig)
