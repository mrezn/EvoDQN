"""Bounded-rational attacker agent."""

from __future__ import annotations

from typing import Dict

import numpy as np

from model_types import ModelConfig
from utils import project_to_simplex


class AttackerAgent:
    """Attacker population governed by bounded-rational WF dynamics."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.omega = config.run.omega_a
        self.mix = np.full(len(config.strategies.attack), 1.0 / len(config.strategies.attack))
        self.fitness = np.ones_like(self.mix)

    def expected_payoffs(self, tilde: np.ndarray, defender_mix: np.ndarray) -> np.ndarray:
        return tilde @ defender_mix

    def selection_probabilities(self, payoffs: np.ndarray) -> np.ndarray:
        fitness = (1.0 - self.omega) + self.omega * payoffs
        weighted = self.mix * fitness
        denom = weighted.sum()
        if denom <= 0:
            denom = 1.0
        self.fitness = fitness
        return weighted / denom

    def update_mix(self, tilde: np.ndarray, defender_mix: np.ndarray, eta: float) -> np.ndarray:
        payoffs = self.expected_payoffs(tilde, defender_mix)
        selection = self.selection_probabilities(payoffs)
        new_mix = self.mix + eta * (selection - self.mix)
        self.mix = project_to_simplex(new_mix)
        return self.mix

    def log_state(self) -> Dict[str, float]:
        return {
            f"attack_{name}": float(prob)
            for name, prob in zip(self.config.strategies.attack, self.mix)
        }
