"""Payoff computations for the WF-MTD edge-cloud model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from model_types import ModelConfig


@dataclass
class StageComponents:
    """Detailed breakdown of attack/defense stage utilities."""

    sal: float
    sap: float
    ac: float
    dc: float
    penalty: float
    incentive: float
    theta_mean: float


class PayoffEngine:
    """Encapsulates payoff block computations as per §2.2–§2.5."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.attack_names = config.strategies.attack
        self.defense_names = config.strategies.defense
        self.attribute_keys = list(config.attributes.weights.keys())

    def _beta(self, attack: str, defense: str) -> float:
        pi_i = self.config.attributes.pi.get(attack, 0.0)
        c_star = self.config.mtd.c_star_for(defense)
        a_param = self.config.mtd.a_for(defense)
        argument = -c_star * a_param * pi_i
        numerator = 1.0 - np.exp(argument)
        denominator = 1.0 + np.exp(argument)
        beta = numerator / denominator if denominator != 0 else 0.0
        return float(np.clip(beta, 0.0, 1.0 - 1e-6))

    def _theta_values(self, attack: str, defense: str) -> Dict[str, float]:
        beta = self._beta(attack, defense)
        lambda_values = self.config.attributes.lambda_for(defense)
        return {
            key: 1.0 - lambda_values[key] * beta
            for key in self.attribute_keys
        }

    def _sal(
        self,
        attack: str,
        defense: str,
        xi_scalar: float,
        theta: Dict[str, float],
        theta_factor: float,
    ) -> float:
        multiplier = 1.0 + xi_scalar
        weights = self.config.attributes.weights_for(attack)
        attr_sum = sum(
            theta[key]
            * weights[key]
            * self.config.attributes.values[key]
            for key in self.attribute_keys
        )
        sal = multiplier * self.config.attributes.resource_importance * attr_sum
        return theta_factor * sal

    def _sap(
        self,
        attack: str,
        defense: str,
        xi_scalar: float,
        theta: Dict[str, float],
        theta_factor: float,
    ) -> float:
        xi_term = max(0.0, 1.0 - xi_scalar)
        weights = self.config.attributes.weights_for(attack)
        mu_y = self.config.mtd.mu_for(defense)
        attr_sum = 0.0
        for key in self.attribute_keys:
            theta_xy = theta[key]
            weight = 1.0 - weights[key]
            contribution = (
                mu_y * theta_xy + (1.0 - theta_xy)
            ) * weight * self.config.attributes.values[key]
            attr_sum += contribution
        sap = xi_term * self.config.attributes.resource_importance * attr_sum
        return theta_factor * sap

    def _attack_cost(self, attack: str) -> float:
        weights = self.config.attack_cost_weights
        features = self.config.attack_cost_features.get(attack, {})
        return sum(weights.get(name, 0.0) * features.get(name, 0.0) for name in weights)

    def _defense_cost(self, defense: str) -> float:
        costs = self.config.defense_costs.get(defense, {})
        assc = costs.get("ASSC", 0.0)
        aic = costs.get("AIC", 0.0)
        scc = costs.get("SCC", 0.0)
        nc = self.config.mtd.SQ * (1.0 - 1.0 / (1.0 + np.exp(-(self.config.mtd.a_for(defense) - self.config.mtd.k))))
        return assc + aic + scc + nc

    def stage_utilities(
        self,
        attack_idx: int,
        defense_idx: int,
        xi_scalar: float,
        theta_attacker: float = 1.0,
        theta_defender: float = 1.0,
    ) -> Tuple[float, float, StageComponents]:
        attack = self.attack_names[attack_idx]
        defense = self.defense_names[defense_idx]
        theta_values = self._theta_values(attack, defense)
        sal = self._sal(attack, defense, xi_scalar, theta_values, theta_attacker)
        sap = self._sap(attack, defense, xi_scalar, theta_values, theta_defender)
        ac = self._attack_cost(attack)
        dc = self._defense_cost(defense)
        theta_mean = float(np.mean(list(theta_values.values())))
        penalty = self.config.run.beta_reg * theta_mean
        incentive = self.config.run.alpha_incentive * self.config.run.info_shared
        ua = sal - ac - penalty
        ud = sap - dc + incentive
        components = StageComponents(
            sal=sal,
            sap=sap,
            ac=ac,
            dc=dc,
            penalty=penalty,
            incentive=incentive,
            theta_mean=theta_mean,
        )
        return ua, ud, components

    def payoff_matrices(
        self,
        xi_scalar: float,
        theta_attacker: float = 1.0,
        theta_defender: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dense payoff matrices for all strategy pairs."""

        atk_count = len(self.attack_names)
        def_count = len(self.defense_names)
        Ua = np.zeros((atk_count, def_count))
        Ud = np.zeros((atk_count, def_count))
        for i in range(atk_count):
            for j in range(def_count):
                ua, ud, _ = self.stage_utilities(i, j, xi_scalar, theta_attacker, theta_defender)
                Ua[i, j] = ua
                Ud[i, j] = ud
        return Ua, Ud
