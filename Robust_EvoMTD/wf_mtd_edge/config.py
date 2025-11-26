"""Configuration loading for the WF-MTD edge-cloud model."""

from __future__ import annotations


from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from pydantic import BaseModel, Field, root_validator, validator

from model_types import (
    AttributeParams,
    Beliefs,
    Kernel,
    ModelConfig,
    MTDParams,
    RunConfig,
    StrategySets,
    UncertaintyParams,
)


class NetworkBlock(BaseModel):
    states: List[str]
    initial_state: str
    adjacency: List[List[int]]
    hosts: List[str]
    initial_b: List[int]
    base_kernel: List[List[List[List[float]]]]

    @validator("initial_state")
    def _validate_initial_state(cls, value: str, values: Dict[str, object]) -> str:
        if "states" in values and value not in values["states"]:
            raise ValueError("Initial state must appear in the state list.")
        return value


class AttributeBlock(BaseModel):
    lambda_values: Dict[str, float] = Field(alias="lambda")
    weights: Dict[str, float]
    values: Dict[str, float]
    resource_importance: float

    class Config:
        allow_population_by_field_name = True


class AttackStrategyBlock(BaseModel):
    name: str
    W: Optional[Dict[str, float]] = None
    weights: Optional[Dict[str, float]] = None
    pi: float
    cost: Optional[Dict[str, float]] = None
    features: Optional[Dict[str, float]] = None
    signal_profile: Optional[Dict[str, float]] = None

    # @root_validator
    def _ensure_costs(cls, values: Dict[str, object]) -> Dict[str, object]:
        if not values.get("cost") and not values.get("features"):
            raise ValueError("Attack strategy must provide either 'cost' or 'features'.")
        return values


class DefenseStrategyBlock(BaseModel):
    name: str
    c_star: Optional[float] = None
    a: Optional[float] = None
    lambda_values: Optional[Dict[str, float]] = Field(default=None, alias="lambda")
    mu: Optional[float] = None
    costs: Dict[str, float]

    class Config:
        allow_population_by_field_name = True


class StrategyBlock(BaseModel):
    attack: List[AttackStrategyBlock]
    defense: List[DefenseStrategyBlock]


class BeliefBlock(BaseModel):
    type_probs: Dict[str, float]
    signal_probs: Dict[str, float]
    attack_given_type: Dict[str, Dict[str, float]]
    misdiagnosis: List[List[float]]

    # @root_validator
    def _check_shapes(cls, values: Dict[str, object]) -> Dict[str, object]:
        atk_types = set(values["attack_given_type"].keys())
        type_probs = set(values["type_probs"].keys())
        if atk_types != type_probs:
            raise ValueError("Attack conditional table must align with defender type keys.")
        matrix = np.asarray(values["misdiagnosis"], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Misdiagnosis matrix must be square.")
        values["misdiagnosis"] = matrix.tolist()
        return values


class MTDBlock(BaseModel):
    c_star: float
    a: float
    mu: Dict[str, float]
    SQ: float
    k: float


class UncertaintyBlock(BaseModel):
    theta_a_bounds: Dict[str, object]
    theta_d_bounds: Dict[str, object]
    dirichlet_alpha: Dict[str, object]
    tv_radius: float
    samples: int
    grid_size: int
    beta_prior: Optional[List[float]] = None
    logit_scale: Optional[float] = None


class RunBlock(BaseModel):
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


class RootConfig(BaseModel):
    network: NetworkBlock
    attributes: AttributeBlock
    strategies: StrategyBlock
    beliefs: BeliefBlock
    mtd: MTDBlock
    uncertainty: UncertaintyBlock
    run: RunBlock
    attack_cost_weights: Dict[str, float]
    defense_costs: Dict[str, Dict[str, float]]


def _index_or_all(name: Optional[str], labels: List[str]) -> List[int]:
    if name is None:
        return list(range(len(labels)))
    if name not in labels:
        raise ValueError(f"Unknown label {name}. Available: {labels}.")
    return [labels.index(name)]


def _expand_bounds(
    base: Dict[str, object],
    states: List[str],
    attacks: List[str],
    defenses: List[str],
    default_key: str = "default",
) -> np.ndarray:
    """Expand a dictionary specification to a dense bounds tensor."""

    default_bounds = base.get(default_key, [0.9, 1.1])
    bounds_array = np.tile(np.asarray(default_bounds, dtype=float), (len(states), len(attacks), len(defenses), 1))
    overrides_data = base.get("overrides", [])
    for override in overrides_data:
        state = override.get("state")
        attack = override.get("attack")
        defense = override.get("defense")
        bounds = override.get("bounds")
        if bounds is None:
            continue
        if len(bounds) != 2:
            raise ValueError("Bounds overrides must provide two values.")
        state_indices = _index_or_all(state, states)
        attack_indices = _index_or_all(attack, attacks)
        defense_indices = _index_or_all(defense, defenses)
        for s in state_indices:
            for a in attack_indices:
                for d in defense_indices:
                    bounds_array[s, a, d, :] = bounds
    return bounds_array


def _expand_dirichlet(
    base: Dict[str, object],
    states: List[str],
    attacks: List[str],
    defenses: List[str],
) -> np.ndarray:
    """Expand Dirichlet concentration parameters to a dense tensor."""

    default_row = np.asarray(base.get("default"), dtype=float)
    if default_row.shape[0] != len(states):
        raise ValueError("Default Dirichlet row must match the number of states.")
    tensor = np.tile(default_row, (len(states), len(attacks), len(defenses), 1))
    overrides = base.get("overrides", [])
    for override in overrides:
        row = np.asarray(override.get("row"), dtype=float)
        if row.size != len(states):
            raise ValueError("Override rows must match the number of states.")
        state = override.get("state")
        attack = override.get("attack")
        defense = override.get("defense")
        state_indices = _index_or_all(state, states)
        attack_indices = _index_or_all(attack, attacks)
        defense_indices = _index_or_all(defense, defenses)
        for s in state_indices:
            for a in attack_indices:
                for d in defense_indices:
                    tensor[s, a, d, :] = row
    return tensor


def load_config(path: str | Path) -> ModelConfig:
    """Load a YAML configuration and build a :class:`ModelConfig`."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data = RootConfig.parse_obj(raw)

    attack_blocks = data.strategies.attack
    defense_blocks = data.strategies.defense
    attack_names = [block.name for block in attack_blocks]
    defense_names = [block.name for block in defense_blocks]

    attack_weight_overrides: Dict[str, Dict[str, float]] = {}
    for block in attack_blocks:
        weights = block.W or block.weights
        if weights:
            attack_weight_overrides[block.name] = weights

    defense_lambda_overrides: Dict[str, Dict[str, float]] = {}
    for block in defense_blocks:
        if block.lambda_values:
            defense_lambda_overrides[block.name] = block.lambda_values

    attributes = AttributeParams(
        lambda_values=data.attributes.lambda_values,
        weights=data.attributes.weights,
        values=data.attributes.values,
        resource_importance=data.attributes.resource_importance,
        pi={block.name: block.pi for block in attack_blocks},
        weight_overrides=attack_weight_overrides,
        lambda_overrides=defense_lambda_overrides,
    )

    mtd = MTDParams(
        c_star=data.mtd.c_star,
        a=data.mtd.a,
        mu_y={name: data.mtd.mu.get(name, 0.0) for name in defense_names},
        SQ=data.mtd.SQ,
        k=data.mtd.k,
        c_star_overrides={block.name: block.c_star for block in defense_blocks if block.c_star is not None},
        a_overrides={block.name: block.a for block in defense_blocks if block.a is not None},
    )
    for block in defense_blocks:
        if block.mu is not None:
            mtd.mu_y[block.name] = block.mu

    uncertainty = UncertaintyParams(
        theta_a_bounds=_expand_bounds(data.uncertainty.theta_a_bounds, data.network.states, attack_names, defense_names),
        theta_d_bounds=_expand_bounds(data.uncertainty.theta_d_bounds, data.network.states, attack_names, defense_names),
        dirichlet_alpha=_expand_dirichlet(data.uncertainty.dirichlet_alpha, data.network.states, attack_names, defense_names),
        tv_radius=data.uncertainty.tv_radius,
        samples=data.uncertainty.samples,
        grid_size=data.uncertainty.grid_size,
        beta_prior=tuple(data.uncertainty.beta_prior) if data.uncertainty.beta_prior else None,
        logit_scale=data.uncertainty.logit_scale,
    )

    strategies = StrategySets(attack=attack_names, defense=defense_names)

    beliefs = Beliefs(
        type_probs=data.beliefs.type_probs,
        signal_probs=data.beliefs.signal_probs,
        attack_given_type=data.beliefs.attack_given_type,
        misdiagnosis=np.asarray(data.beliefs.misdiagnosis, dtype=float),
    )
    beliefs.normalize()

    kernel = Kernel(
        states=data.network.states,
        base=np.asarray(data.network.base_kernel, dtype=float),
        initial_state=data.network.initial_state,
    )

    run = RunConfig(
        horizon=data.run.horizon,
        discount=data.run.discount,
        eta=data.run.eta,
        steps=data.run.steps,
        omega_a=data.run.omega_a,
        omega_d=data.run.omega_d,
        seed=data.run.seed,
        tolerance=data.run.tolerance,
        beta_reg=data.run.beta_reg,
        alpha_incentive=data.run.alpha_incentive,
        info_shared=data.run.info_shared,
        mode=data.run.mode,
        log_path=data.run.log_path,
        trajectory_path=data.run.trajectory_path,
    )
    run.validate()

    attack_features = {}
    for block in attack_blocks:
        features = block.features or block.cost or {}
        attack_features[block.name] = features



    model = ModelConfig(
        attributes=attributes,
        mtd=mtd,
        uncertainty=uncertainty,
        strategies=strategies,
        beliefs=beliefs,
        kernel=kernel,
        run=run,
        adjacency=np.asarray(data.network.adjacency, dtype=int),
        hosts=data.network.hosts,
        initial_b=np.asarray(data.network.initial_b, dtype=int),
        attack_cost_weights=data.attack_cost_weights,
        attack_cost_features=attack_features,
        defense_costs=data.defense_costs,
    )
    return model
