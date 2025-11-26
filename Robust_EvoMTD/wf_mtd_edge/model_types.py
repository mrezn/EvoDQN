"""Typed configuration models for the WF-MTD edge-cloud package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

AttributeDict = Dict[str, float]


@dataclass
class AttributeParams:
    """Container for attribute-related parameters with optional overrides."""

    lambda_values: AttributeDict
    weights: AttributeDict
    values: AttributeDict
    resource_importance: float
    pi: AttributeDict = field(default_factory=dict)
    weight_overrides: Dict[str, AttributeDict] = field(default_factory=dict)
    lambda_overrides: Dict[str, AttributeDict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keys = set(self.lambda_values)
        if not (keys == set(self.weights) == set(self.values)):
            raise ValueError("Attribute dictionaries must share identical keys.")
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(f"Attribute weights must sum to 1. Got {weight_sum}.")
        if self.resource_importance <= 0:
            raise ValueError("Resource importance (C_r) must be strictly positive.")
        for attack, overrides in self.weight_overrides.items():
            if set(overrides) != keys:
                raise ValueError(f"Weight override for {attack} must cover attributes {keys}.")
            total = sum(overrides.values())
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"Weights for {attack} must sum to 1. Got {total}.")
        for defense, overrides in self.lambda_overrides.items():
            if set(overrides) != keys:
                raise ValueError(f"Lambda override for {defense} must cover attributes {keys}.")
            if any(value < 0.0 or value > 1.0 for value in overrides.values()):
                raise ValueError("Lambda overrides must stay within [0, 1].")

    def weights_for(self, attack: str) -> AttributeDict:
        return self.weight_overrides.get(attack, self.weights)

    def lambda_for(self, defense: str) -> AttributeDict:
        return self.lambda_overrides.get(defense, self.lambda_values)


@dataclass
class MTDParams:
    """Parameters governing Moving-Target-Defense efficacy and costs."""

    c_star: float
    a: float
    mu_y: AttributeDict
    SQ: float
    k: float
    c_star_overrides: Dict[str, float] = field(default_factory=dict)
    a_overrides: Dict[str, float] = field(default_factory=dict)

    def mu_for(self, defense: str) -> float:
        return self.mu_y.get(defense, 0.0)

    def c_star_for(self, defense: str) -> float:
        return self.c_star_overrides.get(defense, self.c_star)

    def a_for(self, defense: str) -> float:
        return self.a_overrides.get(defense, self.a)


@dataclass
class UncertaintyParams:
    """Specification of reward and transition uncertainty sets."""

    theta_a_bounds: np.ndarray
    theta_d_bounds: np.ndarray
    dirichlet_alpha: np.ndarray
    tv_radius: float
    samples: int
    grid_size: int
    beta_prior: Optional[Tuple[float, float]] = None
    logit_scale: Optional[float] = None


@dataclass
class StrategySets:
    """Names of attacker and defender pure strategies."""

    attack: List[str]
    defense: List[str]

    def sizes(self) -> Tuple[int, int]:
        return len(self.attack), len(self.defense)


@dataclass
class Beliefs:
    """Belief structure over defender types, signals, and IDS errors."""

    type_probs: AttributeDict
    signal_probs: AttributeDict
    attack_given_type: Dict[str, AttributeDict]
    misdiagnosis: np.ndarray

    def normalize(self) -> None:
        type_sum = sum(self.type_probs.values())
        if not np.isclose(type_sum, 1.0, atol=1e-6):
            raise ValueError("Type probabilities must sum to 1.")
        signal_sum = sum(self.signal_probs.values())
        if not np.isclose(signal_sum, 1.0, atol=1e-6):
            raise ValueError("Signal probabilities must sum to 1.")
        for tau, mapping in self.attack_given_type.items():
            cond_sum = sum(mapping.values())
            if not np.isclose(cond_sum, 1.0, atol=1e-6):
                raise ValueError(f"Conditional attack probabilities for {tau} must sum to 1.")


@dataclass
class Kernel:
    """Baseline transition kernel and ambiguity specification."""

    states: List[str]
    base: np.ndarray
    initial_state: str

    def state_index(self, state: str) -> int:
        try:
            return self.states.index(state)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown state {state}.") from exc


@dataclass
class RunConfig:
    """Simulation-level configuration parameters."""

    horizon: int
    discount: float
    eta: float
    steps: int
    omega_a: float
    omega_d: float
    seed: int
    tolerance: float
    beta_reg: float
    alpha_incentive: float
    info_shared: float
    mode: str = "robust"
    log_path: Optional[str] = None
    trajectory_path: Optional[str] = None

    def validate(self) -> None:
        if not (0.0 < self.discount <= 1.0):
            raise ValueError("Discount factor must lie in (0, 1].")
        if not (0.0 < self.eta <= 1.0):
            raise ValueError("Evolution stepsize eta must lie in (0, 1].")
        if not (0.0 <= self.omega_a <= 1.0 and 0.0 <= self.omega_d <= 1.0):
            raise ValueError("Omega parameters must lie in [0, 1].")
        if self.horizon <= 0 or self.steps <= 0:
            raise ValueError("Horizon and step counts must be positive integers.")
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be strictly positive.")


@dataclass
class ModelConfig:
    """Aggregated configuration object used across modules."""

    attributes: AttributeParams
    mtd: MTDParams
    uncertainty: UncertaintyParams
    strategies: StrategySets
    beliefs: Beliefs
    kernel: Kernel
    run: RunConfig
    adjacency: np.ndarray
    hosts: List[str]
    initial_b: np.ndarray
    attack_cost_weights: AttributeDict
    attack_cost_features: Dict[str, AttributeDict]
    defense_costs: Dict[str, AttributeDict]


__all__ = [
    "AttributeParams",
    "MTDParams",
    "UncertaintyParams",
    "StrategySets",
    "Beliefs",
    "Kernel",
    "RunConfig",
    "ModelConfig",
]
