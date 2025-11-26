"""Uncertainty handling for robust WF-MTD optimisation."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import truncnorm

from model_types import ModelConfig
from utils import ensure_rng, l1_distance


class UncertaintyModel:
    """Helper that samples parameters from ambiguity sets or priors."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.rng = ensure_rng(config.run.seed)

    def theta_candidates(
        self,
        state: int,
        attack: int,
        defense: int,
        mode: str,
        for_attacker: bool,
    ) -> np.ndarray:
        bounds_tensor = self.config.uncertainty.theta_a_bounds if for_attacker else self.config.uncertainty.theta_d_bounds
        lower, upper = bounds_tensor[state, attack, defense]
        if mode == "bayes":
            alpha_beta = self.config.uncertainty.beta_prior or (2.0, 2.0)
            alpha, beta = alpha_beta
            samples = self.config.uncertainty.samples
            draws = self.rng.beta(alpha, beta, size=samples)
            scaled = lower + (upper - lower) * draws
            return scaled
        grid = np.linspace(lower, upper, num=max(2, self.config.uncertainty.grid_size))
        return grid

    def transition_candidates(
        self,
        state: int,
        attack: int,
        defense: int,
        mode: str,
        base_kernel: np.ndarray,
    ) -> List[np.ndarray]:
        alpha = self.config.uncertainty.dirichlet_alpha[state, attack, defense]
        base_row = base_kernel[state, attack, defense]
        candidates: List[np.ndarray] = []
        if mode == "bayes":
            for _ in range(self.config.uncertainty.samples):
                draw = self.rng.dirichlet(alpha + 1e-6)
                candidates.append(draw)
        else:
            candidates.append(base_row)
            epsilon = self.config.uncertainty.tv_radius
            for _ in range(self.config.uncertainty.samples):
                idx = self.rng.integers(0, base_row.shape[0])
                lam = self.rng.uniform(0.0, min(0.5 * epsilon, 0.49))
                vertex = np.zeros_like(base_row)
                vertex[idx] = 1.0
                candidate = (1.0 - lam) * base_row + lam * vertex
                candidate = candidate / candidate.sum()
                if l1_distance(candidate, base_row) <= epsilon + 1e-8:
                    candidates.append(candidate)
        if not candidates:
            candidates.append(base_row)
        return candidates

    def robust_minimize(
        self,
        theta_values: Sequence[float],
        transition_candidates: Iterable[np.ndarray],
        objective,
    ) -> Tuple[float, np.ndarray, float]:
        """Enumerate candidates and return the smallest objective value."""

        best_value = float("inf")
        best_transition = None
        best_theta = None
        for theta in theta_values:
            for transition in transition_candidates:
                value = objective(theta, transition)
                if value < best_value:
                    best_value = value
                    best_transition = transition
                    best_theta = theta
        if best_transition is None or best_theta is None:
            raise RuntimeError("Failed to evaluate any uncertainty candidates.")
        return best_value, best_transition, best_theta

    def truncated_normal(self, mean: float, std: float, lower: float, upper: float, size: int) -> np.ndarray:
        """Utility for drawing from a truncated normal distribution."""

        a, b = (lower - mean) / std, (upper - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=self.rng)
