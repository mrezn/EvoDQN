"""Evolutionary simulation orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from agents import AttackerAgent, DefenderAgent
from environment import EdgeCloudEnvironment
from payoffs import PayoffEngine
from robust_dp import RobustDPResult, RobustDynamicProgrammer
from model_types import ModelConfig
from utils import write_json_lines


@dataclass
class EvolutionLog:
    step: int
    attacker_payoff: float
    defender_payoff: float
    rho: List[float]
    attacker_mix: List[float]
    defender_mix: List[float]


class EvolutionEngine:
    """Coordinate robust DP outputs with WF evolutionary dynamics."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.environment = EdgeCloudEnvironment(config)
        self.payoff_engine = PayoffEngine(config)
        self.dp = RobustDynamicProgrammer(config, self.environment)
        self.attacker = AttackerAgent(config)
        self.defender = DefenderAgent(config)
        self.logs: List[EvolutionLog] = []
        self.dp_result: Optional[RobustDPResult] = None
        self.history_data: Dict[str, object] = {}

    def run(self, mode: Optional[str] = None, steps: Optional[int] = None) -> RobustDPResult:
        mode = mode or self.config.run.mode
        steps = steps or self.config.run.steps
        self.dp_result = self.dp.solve(mode)
        eta = self.config.run.eta

        p_series: List[np.ndarray] = []
        q_series: List[np.ndarray] = []
        fa_series: List[np.ndarray] = []
        fd_series: List[np.ndarray] = []
        Ua_series: List[float] = []
        Ud_series: List[float] = []
        xi_series: List[float] = []
        theta_keys = list(self.payoff_engine.attribute_keys)
        theta_series: Dict[str, List[float]] = {key: [] for key in theta_keys}

        for step in range(steps):
            weighted_tilde_a = np.tensordot(self.environment.rho, self.dp_result.tilde_a, axes=(0, 0))
            weighted_tilde_b = np.tensordot(self.environment.rho, self.dp_result.tilde_b, axes=(0, 0))

            prev_attack_mix = self.attacker.mix.copy()
            prev_defense_mix = self.defender.mix.copy()

            self.attacker.update_mix(weighted_tilde_a, self.defender.mix, eta)
            self.defender.update_mix(weighted_tilde_b, self.attacker.mix, eta)

            attacker_payoff = float(self.attacker.mix @ weighted_tilde_a @ self.defender.mix)
            defender_payoff = float(self.attacker.mix @ weighted_tilde_b @ self.defender.mix)

            fa_vector = weighted_tilde_a @ self.defender.mix
            fd_vector = weighted_tilde_b.T @ self.attacker.mix

            metrics = self.environment.compute_lateral_metrics()
            xi_values = metrics.xi.get(1) if getattr(metrics, "xi", None) else None
            if xi_values is None and metrics.xi:
                xi_val = float(np.mean(next(iter(metrics.xi.values()))))
            elif xi_values is None:
                xi_val = 0.0
            else:
                xi_val = float(np.mean(xi_values))

            theta_acc = {key: 0.0 for key in theta_keys}
            for i, attack_name in enumerate(self.config.strategies.attack):
                for j, defense_name in enumerate(self.config.strategies.defense):
                    theta_map = self.payoff_engine._theta_values(attack_name, defense_name)  # type: ignore[attr-defined]
                    weight = float(self.attacker.mix[i] * self.defender.mix[j])
                    for key in theta_keys:
                        theta_acc[key] += theta_map[key] * weight

            for key in theta_keys:
                theta_series[key].append(theta_acc[key])

            p_series.append(self.attacker.mix.copy())
            q_series.append(self.defender.mix.copy())
            fa_series.append(fa_vector.copy())
            fd_series.append(fd_vector.copy())
            Ua_series.append(attacker_payoff)
            Ud_series.append(defender_payoff)
            xi_series.append(xi_val)

            self.environment.update_state_distribution(self.attacker.mix, self.defender.mix, self.dp_result.transitions)

            self.logs.append(
                EvolutionLog(
                    step=step,
                    attacker_payoff=attacker_payoff,
                    defender_payoff=defender_payoff,
                    rho=self.environment.rho.tolist(),
                    attacker_mix=self.attacker.mix.tolist(),
                    defender_mix=self.defender.mix.tolist(),
                )
            )

            delta = max(
                float(np.max(np.abs(self.attacker.mix - prev_attack_mix))),
                float(np.max(np.abs(self.defender.mix - prev_defense_mix))),
            )
            if delta < self.config.run.tolerance:
                break

        if p_series:
            self.history_data = {
                "p": np.vstack(p_series),
                "q": np.vstack(q_series),
                "fa": np.vstack(fa_series),
                "fd": np.vstack(fd_series),
                "Ua": np.array(Ua_series),
                "Ud": np.array(Ud_series),
                "xi_m": np.array(xi_series),
                "theta": {key: np.array(vals) for key, vals in theta_series.items()},
            }
        else:
            self.history_data = {
                "p": np.empty((0, len(self.config.strategies.attack))),
                "q": np.empty((0, len(self.config.strategies.defense))),
                "fa": np.empty((0, len(self.config.strategies.attack))),
                "fd": np.empty((0, len(self.config.strategies.defense))),
                "Ua": np.empty(0),
                "Ud": np.empty(0),
                "xi_m": np.empty(0),
                "theta": {key: np.empty(0) for key in theta_keys},
            }

        if self.config.run.trajectory_path:
            write_json_lines(self.config.run.trajectory_path, (log.__dict__ for log in self.logs))

        return self.dp_result

    def summary(self) -> Dict[str, object]:
        if not self.logs:
            raise RuntimeError("Evolution has not been run yet.")
        last_log = self.logs[-1]
        return {
            "attacker_mix": dict(zip(self.config.strategies.attack, last_log.attacker_mix)),
            "defender_mix": dict(zip(self.config.strategies.defense, last_log.defender_mix)),
            "attacker_payoff": last_log.attacker_payoff,
            "defender_payoff": last_log.defender_payoff,
            "rho": dict(zip(self.config.kernel.states, last_log.rho)),
        }
