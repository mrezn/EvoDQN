"""Edge-cloud environment dynamics for the WF-MTD model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from model_types import ModelConfig


@dataclass
class LateralMetrics:
    """Container for lateral movement coefficients."""

    xi: Dict[int, np.ndarray]
    amplifiers: Dict[int, np.ndarray]


class EdgeCloudEnvironment:
    """Edge-cloud substrate handling graph topology and state transitions."""

    def __init__(self, config: ModelConfig, gamma_att: float = 0.6) -> None:
        self.config = config
        self.gamma_att = gamma_att
        self.graph = nx.from_numpy_array(config.adjacency, create_using=nx.Graph)
        self._hosts_online = config.initial_b.astype(bool)
        self.state_count = len(config.kernel.states)
        self.attack_count = len(config.strategies.attack)
        self.defense_count = len(config.strategies.defense)
        self.current_state = config.kernel.state_index(config.kernel.initial_state)
        self.rho = np.zeros(self.state_count)
        self.rho[self.current_state] = 1.0

    @property
    def hosts_online(self) -> np.ndarray:
        return self._hosts_online.copy()

    def set_hosts(self, mask: np.ndarray) -> None:
        if mask.shape[0] != self._hosts_online.shape[0]:
            raise ValueError("Host vector has incompatible dimension.")
        self._hosts_online = mask.astype(bool)

    def compute_reachability(self) -> np.ndarray:
        """Compute c_ij(t) = r_ij(t) * b_i(t) * b_j(t)."""

        adjacency = self.config.adjacency
        mask = np.outer(self._hosts_online.astype(int), self._hosts_online.astype(int))
        return adjacency * mask

    def compute_lateral_metrics(self, m_max: int = 3) -> LateralMetrics:
        """Compute xi_m coefficients up to order *m_max* as stated in ?2.1."""

        if m_max < 1 or m_max > 3:
            raise ValueError("This implementation supports m in {1, 2, 3}.")
        c_matrix = self.compute_reachability()
        base = self.gamma_att * c_matrix.astype(float)
        xi: Dict[int, np.ndarray] = {1: base.sum(axis=1)}
        n = base.shape[0]
        if m_max >= 2:
            xi2 = np.zeros(n)
            row_sums = base.sum(axis=1)
            for i in range(n):
                acc = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    inner = row_sums[j] - base[j, i]
                    acc += base[i, j] * inner
                xi2[i] = acc
            xi[2] = xi2
        if m_max >= 3:
            xi3 = np.zeros(n)
            row_sums = base.sum(axis=1)
            for i in range(n):
                acc = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    for k in range(n):
                        if k == j:
                            continue
                        inner = row_sums[k] - base[k, j]
                        acc += base[i, j] * base[j, k] * inner
                xi3[i] = acc
            xi[3] = xi3
        amplifiers = {order: 1.0 + values for order, values in xi.items()}
        return LateralMetrics(xi=xi, amplifiers=amplifiers)

    def sample_next_state(self, attack_idx: int, defense_idx: int, kernel: np.ndarray, rng: np.random.Generator) -> int:
        """Sample the next state index given a transition kernel row."""

        distribution = kernel[self.current_state, attack_idx, defense_idx]
        distribution = distribution / distribution.sum()
        next_state = rng.choice(self.state_count, p=distribution)
        self.current_state = int(next_state)
        return self.current_state

    def update_state_distribution(self, p: np.ndarray, q: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Update rho_{k+1}(S') = ?_S rho_k(S) ?_{i,j} p_i q_j P(S' | S, i, j)."""

        new_rho = np.zeros_like(self.rho)
        for s in range(self.state_count):
            if self.rho[s] <= 0.0:
                continue
            next_dist = np.einsum("i,j,ijs->s", p, q, kernel[s])
            new_rho += self.rho[s] * next_dist
        if new_rho.sum() > 0:
            new_rho /= new_rho.sum()
        self.rho = new_rho
        return self.rho

    def reset(self) -> None:
        self.current_state = self.config.kernel.state_index(self.config.kernel.initial_state)
        self.rho = np.zeros(self.state_count)
        self.rho[self.current_state] = 1.0
