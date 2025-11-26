from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

RobustMode = Literal["min", "mean"]


@dataclass(frozen=True)
class Scenario:
    """Container describing one uncertainty realisation.

    Attributes
    ----------
    defender_scale:
        Multiplicative factor applied to the nominal defender reward.
    attacker_scale:
        Multiplicative factor applied to the nominal attacker reward.
    transition:
        Stochastic matrix sampled from the ambiguity set associated with the
        current state-action tuple.
    """

    defender_scale: float
    attacker_scale: float
    transition: np.ndarray


def _logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def sample_reward_scales(
    bounds: tuple[float, float],
    logit_params: dict[str, float],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample reward multipliers from a logit-normal distribution.

    The draws are constrained to ``bounds`` and therefore materialise the set
    :math:`\Theta^D` or :math:`\Theta^A` depending on the caller.

    Complexity is :math:`\mathcal{O}(k)` where ``k = size``.
    """
    mu = float(logit_params.get("mu", 0.0))
    sigma = float(logit_params.get("sigma", 1.0))
    normal_draws = rng.normal(mu, sigma, size)
    scaled = bounds[0] + (bounds[1] - bounds[0]) * _logistic(normal_draws)
    return scaled


def sample_transition_matrix(
    base: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a transition matrix from the ambiguity set ``??``.

    Each row of the nominal matrix acts as the mean of a Dirichlet distribution
    with concentration ``alpha``. The resulting matrix is still row stochastic
    and preserves the support of the base transitions.

    Complexity is :math:`\mathcal{O}(n^2)` for ``n`` states because we iterate
    over every row and sample a Dirichlet vector of length ``n``.
    """
    base = np.asarray(base, dtype=np.float64)
    n_states = base.shape[0]
    sampled = np.zeros_like(base)
    for i in range(n_states):
        row = base[i]
        row = np.clip(row, 1e-9, 1.0)
        row /= row.sum()
        concentration = row * float(alpha)
        sampled[i] = rng.dirichlet(concentration)
    return sampled


def sample_scenarios(
    base_transition: np.ndarray,
    reward_bounds_def: tuple[float, float],
    reward_bounds_att: tuple[float, float],
    logit_params: dict[str, float],
    alpha: float,
    count: int,
    rng: np.random.Generator,
) -> list[Scenario]:
    defender_scales = sample_reward_scales(reward_bounds_def, logit_params, count, rng)
    attacker_scales = sample_reward_scales(reward_bounds_att, logit_params, count, rng)
    scenarios: list[Scenario] = []
    for i in range(count):
        transition = sample_transition_matrix(base_transition, alpha, rng)
        scenarios.append(
            Scenario(
                defender_scale=float(defender_scales[i]),
                attacker_scale=float(attacker_scales[i]),
                transition=transition,
            )
        )
    return scenarios


def aggregate(values: Sequence[float], mode: RobustMode) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot aggregate an empty set of values.")
    if mode == "min":
        return float(np.min(arr))
    if mode == "mean":
        return float(np.mean(arr))
    raise ValueError(f"Unsupported mode: {mode}")


def defender_bellman_proxy(
    rewards: Sequence[float],
    transitions: Sequence[np.ndarray],
    q_values_next: np.ndarray,
    gamma: float,
    mode: RobustMode = "min",
) -> float:

    aggregated: list[float] = []
    q_values_next = np.asarray(q_values_next, dtype=np.float64)
    for reward, transition in zip(rewards, transitions):
        expectation = float(np.sum(np.asarray(transition, dtype=np.float64) * q_values_next))
        aggregated.append(float(reward) + float(gamma) * expectation)
    return aggregate(aggregated, mode)


def attacker_bellman_proxy(
    rewards: Sequence[float],
    value_baseline: float,
    mode: RobustMode = "min",
) -> float:

    adjusted = [float(reward) + float(value_baseline) for reward in rewards]
    return aggregate(adjusted, mode)
