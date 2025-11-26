from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .utils import project_to_simplex


@dataclass
class StageGameAverager:


    num_attacker: int
    num_defender: int
    attacker_values: list[list[list[float]]] = field(init=False)
    defender_values: list[list[list[float]]] = field(init=False)

    def __post_init__(self) -> None:
        self.attacker_values = [[[] for _ in range(self.num_defender)] for _ in range(self.num_attacker)]
        self.defender_values = [[[] for _ in range(self.num_defender)] for _ in range(self.num_attacker)]

    def add(self, attacker_idx: int, defender_idx: int, attacker_proxy: float, defender_proxy: float) -> None:
        self.attacker_values[attacker_idx][defender_idx].append(float(attacker_proxy))
        self.defender_values[attacker_idx][defender_idx].append(float(defender_proxy))

    def as_matrices(self) -> tuple[np.ndarray, np.ndarray]:
 
        a_matrix = np.zeros((self.num_attacker, self.num_defender), dtype=np.float64)
        b_matrix = np.zeros((self.num_attacker, self.num_defender), dtype=np.float64)
        for i in range(self.num_attacker):
            for j in range(self.num_defender):
                if self.attacker_values[i][j]:
                    a_matrix[i, j] = float(np.mean(self.attacker_values[i][j]))
                if self.defender_values[i][j]:
                    b_matrix[i, j] = float(np.mean(self.defender_values[i][j]))
        return a_matrix, b_matrix

    def reset(self) -> None:
        """Clear the accumulated observations."""
        self.__post_init__()


@dataclass
class WrightFisherCoupler:


    attacker_strategies: Sequence[str]
    defender_strategies: Sequence[str]
    omega_attacker: float
    omega_defender: float
    eta: float
    mix_attacker: np.ndarray = field(init=False)
    mix_defender: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n_att = len(self.attacker_strategies)
        n_def = len(self.defender_strategies)
        self.mix_attacker = np.full(n_att, 1.0 / n_att, dtype=np.float64)
        self.mix_defender = np.full(n_def, 1.0 / n_def, dtype=np.float64)

    def compute_fitness(self, a_matrix: np.ndarray, b_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        attacker_fitness = (1.0 - float(self.omega_attacker)) + float(self.omega_attacker) * (
            a_matrix @ self.mix_defender
        )
        defender_fitness = (1.0 - float(self.omega_defender)) + float(self.omega_defender) * (
            self.mix_attacker @ b_matrix
        )
        return attacker_fitness, defender_fitness

    def update(self, a_matrix: np.ndarray, b_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  
        a_matrix = np.asarray(a_matrix, dtype=np.float64)
        b_matrix = np.asarray(b_matrix, dtype=np.float64)
        attacker_fitness, defender_fitness = self.compute_fitness(a_matrix, b_matrix)

        mean_attacker = float(np.dot(self.mix_attacker, attacker_fitness))
        mean_defender = float(np.dot(self.mix_defender, defender_fitness))

        updated_attacker = self.mix_attacker * (1.0 + float(self.eta) * (attacker_fitness - mean_attacker))
        updated_defender = self.mix_defender * (1.0 + float(self.eta) * (defender_fitness - mean_defender))

        self.mix_attacker = project_to_simplex(updated_attacker)
        self.mix_defender = project_to_simplex(updated_defender)
        return self.mix_attacker.copy(), self.mix_defender.copy()

    def mixes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the current attacker and defender mixes."""
        return self.mix_attacker.copy(), self.mix_defender.copy()
