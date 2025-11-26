from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class PayoffComponents:
    """Container collecting intermediate quantities in the payoff computation."""

    theta_xy: np.ndarray
    sal: float
    sap: float
    attacker_cost: float
    defender_cost: float
    guidance_penalty: float


def beta_success(c_star: float, a_r: float, pi_i: float) -> float:
    """Eq. (3): logistic-shaped defender disruption success map ``�_y``."""
    exponent = math.exp(-float(c_star) * float(a_r) * float(pi_i))
    return (1.0 - exponent) / (1.0 + exponent)


def theta_xy(lambda_x: float, beta_y_val: float) -> float:
    """Eq. (4): residual attack success given the detection factor ``?_x``."""
    return 1.0 - float(lambda_x) * float(beta_y_val)


def compute_sal(
    xi_z: float,
    c_r: float,
    theta_vec: ArrayLike,
    attacker_weights: ArrayLike,
    valuations: ArrayLike,
) -> float:
    """Eq. (6): security attack loss (SAL) affecting the defender."""
    theta_arr = np.asarray(theta_vec, dtype=np.float64)
    weights = np.asarray(attacker_weights, dtype=np.float64)
    valuations_arr = np.asarray(valuations, dtype=np.float64)
    weighted = theta_arr * weights * valuations_arr
    return (1.0 + float(xi_z)) * float(c_r) * float(np.sum(weighted))


def compute_sap(
    xi_z: float,
    c_r: float,
    theta_vec: ArrayLike,
    mu_vec: ArrayLike,
    defender_weights: ArrayLike,
    valuations: ArrayLike,
) -> float:
    """Eq. (7): security assurance payoff (SAP) credited to the defender."""
    theta_arr = np.asarray(theta_vec, dtype=np.float64)
    mu_arr = np.asarray(mu_vec, dtype=np.float64)
    weights = np.asarray(defender_weights, dtype=np.float64)
    valuations_arr = np.asarray(valuations, dtype=np.float64)
    contribution = (mu_arr * theta_arr + (1.0 - theta_arr)) * (1.0 - weights)
    return (1.0 - float(xi_z)) * float(c_r) * float(np.sum(contribution * valuations_arr))


def network_cost(sq: float, a_r: float, k_s: float) -> float:
    """Eq. (8): non-compliance cost term ``NC`` using a sigmoid workload model."""
    return float(sq) * (1.0 - 1.0 / (1.0 + math.exp(-(float(a_r) - float(k_s)))))


def defender_cost(assc: float, aic: float, sq: float, a_r: float, k_s: float) -> float:
    """Aggregate defender cost ``DC = ASSC + NC + AIC`` as per Eq. (9)."""
    return float(assc) + network_cost(sq, a_r, k_s) + float(aic)


def attacker_cost(operational_cost: float, payload_cost: float = 0.0) -> float:
    """Eq. (10): attacker cost component representing assembly/computation."""
    return float(operational_cost) + float(payload_cost)


def attacker_payoff(sal: float, ac_cost: float, guidance_penalty: float) -> float:
    """Eq. (11): attacker utility ``U_A = SAL - AC - G_p``."""
    return float(sal) - float(ac_cost) - float(guidance_penalty)


def defender_payoff(sap: float, def_cost: float, alpha_I: float, incentive: float) -> float:
    """Eq. (12): defender utility ``U_D = SAP - DC + a I``."""
    return float(sap) - float(def_cost) + float(alpha_I) * float(incentive)


def summarise_payoffs(
    xi_z: float,
    c_r: float,
    attacker_weights: ArrayLike,
    defender_weights: ArrayLike,
    valuations: ArrayLike,
    mu_vec: ArrayLike,
    c_star: float,
    a_r: float,
    pi_i: float,
    lambda_vec: ArrayLike,
    sq: float,
    k_s: float,
    assc: float,
    aic: float,
    operational_cost: float,
    guidance_penalty: float,
    alpha_I: float,
    incentive: float,
) -> PayoffComponents:
    """Compose the full payoff tuple used by the environment."""
    beta_val = beta_success(c_star, a_r, pi_i)
    theta_vec = np.array([theta_xy(lmbda, beta_val) for lmbda in lambda_vec], dtype=np.float64)
    sal_val = compute_sal(xi_z, c_r, theta_vec, attacker_weights, valuations)
    sap_val = compute_sap(xi_z, c_r, theta_vec, mu_vec, defender_weights, valuations)
    def_cost = defender_cost(assc, aic, sq, a_r, k_s)
    att_cost = attacker_cost(operational_cost)
    return PayoffComponents(
        theta_xy=theta_vec,
        sal=sal_val,
        sap=sap_val,
        attacker_cost=att_cost,
        defender_cost=def_cost,
        guidance_penalty=float(guidance_penalty),
    )
