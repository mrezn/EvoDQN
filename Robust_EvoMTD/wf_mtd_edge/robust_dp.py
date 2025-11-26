"""Robust dynamic programming for the WF-MTD model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment import EdgeCloudEnvironment
from payoffs import PayoffEngine
from model_types import ModelConfig
from uncertainty import UncertaintyModel


@dataclass
class RobustDPResult:
    tilde_a: np.ndarray
    tilde_b: np.ndarray
    transitions: np.ndarray
    attacker_thetas: np.ndarray
    defender_thetas: np.ndarray
    value_a: np.ndarray
    value_d: np.ndarray


class RobustDynamicProgrammer:
    """Backward induction under reward and transition uncertainty."""

    def __init__(self, config: ModelConfig, environment: EdgeCloudEnvironment) -> None:
        self.config = config
        self.environment = environment
        self.payoffs = PayoffEngine(config)
        self.uncertainty = UncertaintyModel(config)
        self.state_count = len(config.kernel.states)
        self.attack_count = len(config.strategies.attack)
        self.defense_count = len(config.strategies.defense)

    def solve(self, mode: str) -> RobustDPResult:
        horizon = self.config.run.horizon
        gamma = self.config.run.discount
        tilde_a = np.zeros((self.state_count, self.attack_count, self.defense_count))
        tilde_b = np.zeros_like(tilde_a)
        transitions = np.zeros((self.state_count, self.attack_count, self.defense_count, self.state_count))
        attacker_thetas = np.zeros_like(tilde_a)
        defender_thetas = np.zeros_like(tilde_a)
        value_a = np.zeros((horizon + 1, self.state_count))
        value_d = np.zeros((horizon + 1, self.state_count))

        metrics = self.environment.compute_lateral_metrics()
        xi_scalar = float(np.mean(metrics.xi[1]))

        base_kernel = self.config.kernel.base

        for stage in range(horizon - 1, -1, -1):
            for s in range(self.state_count):
                for i in range(self.attack_count):
                    for j in range(self.defense_count):
                        theta_candidates_att = self.uncertainty.theta_candidates(s, i, j, mode, True)
                        theta_candidates_def = self.uncertainty.theta_candidates(s, i, j, mode, False)
                        transition_candidates = self.uncertainty.transition_candidates(s, i, j, mode, base_kernel)

                        def attacker_objective(theta, transition):
                            ua, _, _ = self.payoffs.stage_utilities(i, j, xi_scalar, theta_attacker=theta, theta_defender=1.0)
                            return ua + gamma * float(np.dot(transition, value_a[stage + 1]))

                        best_a, best_transition, best_theta_a = self.uncertainty.robust_minimize(
                            theta_candidates_att, transition_candidates, attacker_objective
                        )
                        tilde_a[s, i, j] = best_a
                        transitions[s, i, j] = best_transition
                        attacker_thetas[s, i, j] = best_theta_a

                        def defender_objective(theta, transition):
                            _, ud, _ = self.payoffs.stage_utilities(i, j, xi_scalar, theta_attacker=1.0, theta_defender=theta)
                            return ud + gamma * float(np.dot(transition, value_d[stage + 1]))

                        best_d, _, best_theta_d = self.uncertainty.robust_minimize(
                            theta_candidates_def, transition_candidates, defender_objective
                        )
                        tilde_b[s, i, j] = best_d
                        defender_thetas[s, i, j] = best_theta_d

                value_a[stage, s] = np.max(tilde_a[s])
                value_d[stage, s] = np.max(tilde_b[s])

        return RobustDPResult(
            tilde_a=tilde_a,
            tilde_b=tilde_b,
            transitions=transitions,
            attacker_thetas=attacker_thetas,
            defender_thetas=defender_thetas,
            value_a=value_a,
            value_d=value_d,
        )
