from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np

from . import payoffs
from .uncertainty import Scenario, sample_scenarios


@dataclass
class EnvState:
    """Container returned by :meth:`EdgeCloudMTDEnv.reset` and ``step``."""

    features: np.ndarray
    index: int


class EdgeCloudMTDEnv:

    COMPONENTS = ("confidentiality", "integrity", "availability")

    def __init__(self, config: dict, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.rng = rng or np.random.default_rng(seed=config.get("seeds", {}).get("global_seed", None))

        env_cfg = config.get("env", {})
        self.attacker_strategies: List[str] = list(env_cfg.get("strategies", {}).get("G_A", []))
        self.defender_strategies: List[str] = list(env_cfg.get("strategies", {}).get("G_D", []))
        self.num_attacker_actions = len(self.attacker_strategies)
        self.num_defender_actions = len(self.defender_strategies)
        if self.num_attacker_actions == 0 or self.num_defender_actions == 0:
            raise ValueError("Strategies must be defined in the configuration.")

        self.gamma = float(env_cfg.get("hop_attenuation", 0.85))
        self.lateral_hops = tuple(env_cfg.get("lateral_hops", [1, 2, 3]))
        self.detection_rates = np.array(
            [
                env_cfg.get("detection_rates", {}).get("lambda_c", 0.35),
                env_cfg.get("detection_rates", {}).get("lambda_i", 0.42),
                env_cfg.get("detection_rates", {}).get("lambda_a", 0.51),
            ],
            dtype=np.float64,
        )

        valuation_cfg = env_cfg.get("valuation", {})
        self.valuations = np.array(
            [
                valuation_cfg.get("R_c", 2.5),
                valuation_cfg.get("R_i", 3.0),
                valuation_cfg.get("R_a", 3.5),
            ],
            dtype=np.float64,
        )
        self.C_r = float(valuation_cfg.get("resource_importance", 1.8))

        defender_knobs = env_cfg.get("defender_knobs", {})
        self.c_star = float(defender_knobs.get("c_star", 0.8))
        self.a_r = float(defender_knobs.get("a_r", 1.2))
        self.SQ = float(defender_knobs.get("SQ", 0.6))
        self.k_s = float(defender_knobs.get("k_s", 0.4))
        self.ASSC = float(defender_knobs.get("ASSC", 0.3))
        self.AIC = float(defender_knobs.get("AIC", 0.25))
        self.alpha_I = float(defender_knobs.get("alpha_I", 0.2))
        self.incentive_bias = float(defender_knobs.get("I", 1.0))

        attacker_capability_cfg = env_cfg.get("attacker_capability", {})
        self.attacker_capability = {
            strategy: float(attacker_capability_cfg.get(strategy, 0.5))
            for strategy in self.attacker_strategies
        }

        self.attack_component_weights = self._init_attack_component_weights()
        self.defense_component_weights = self._init_defense_component_weights()

        path_structure = env_cfg.get("path_structure", {})
        self.path1_states = int(path_structure.get("path1_states", 5))
        self.path2_states = int(path_structure.get("path2_states", 4))

        self.graph = self._build_graph()
        self.adjacency = nx.to_numpy_array(self.graph, dtype=float)
        self.num_states = self.adjacency.shape[0]
        self.state_index = {node: idx for idx, node in enumerate(self.graph.nodes)}

        self.nominal_transition = self._build_nominal_transition()
        self.base_transition_bank = self._build_transition_bank()

        self.node_activity = np.zeros(self.num_states, dtype=np.float64)
        self.attacker_mix_stats = np.full(self.num_attacker_actions, 1.0 / self.num_attacker_actions)

        self.current_state_idx = 0
        self.time_step = 0
        self._recent_attack_outcomes: list[int] = []
        self.last_sal: float | None = None

        ambiguity_cfg = env_cfg.get("ambiguity", {})
        self.U_scenarios = int(ambiguity_cfg.get("U_scenarios", 8))
        self.dirichlet_alpha = float(ambiguity_cfg.get("dirichlet_alpha", 2.0))
        reward_bounds = ambiguity_cfg.get("reward_bounds", {})
        self.reward_bounds_def = tuple(reward_bounds.get("defender", [0.8, 1.2]))
        self.reward_bounds_att = tuple(reward_bounds.get("attacker", [0.6, 1.4]))
        self.logit_params = ambiguity_cfg.get("logit_normal", {"mu": 0.0, "sigma": 0.25})

    def _build_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        start_node = "entry"
        graph.add_node(start_node, kind="entry")

        prev_node = start_node
        for idx in range(self.path1_states):
            node = f"p1_s{idx}"
            graph.add_node(node, path=1, depth=idx)
            graph.add_edge(prev_node, node)
            prev_node = node
        graph.nodes[prev_node]["terminal"] = True

        prev_node = start_node
        for idx in range(self.path2_states):
            node = f"p2_s{idx}"
            graph.add_node(node, path=2, depth=idx)
            graph.add_edge(prev_node, node)
            if idx > 0:
                graph.add_edge(f"p1_s{min(idx, self.path1_states - 1)}", node)
            prev_node = node
        graph.nodes[prev_node]["terminal"] = True
        return graph

    def _build_nominal_transition(self) -> np.ndarray:
        adjacency = self.adjacency.copy()
        nominal = np.zeros_like(adjacency)
        for i in range(self.num_states):
            row = adjacency[i]
            if row.sum() == 0:
                nominal[i, i] = 1.0
            else:
                nominal[i] = row / row.sum()
        return nominal

    def _build_transition_bank(self) -> Dict[tuple[int, int], np.ndarray]:
        bank: Dict[tuple[int, int], np.ndarray] = {}
        for i, att in enumerate(self.attacker_strategies):
            for j, df in enumerate(self.defender_strategies):
                modifier = np.eye(self.num_states)
                if "hop" in df:
                    modifier *= 0.92
                if att == "overflow":
                    modifier += 0.05 * self.nominal_transition
                elif att == "destroy":
                    modifier += 0.02 * self.adjacency
                elif att == "lateral":
                    modifier += 0.03 * np.linalg.matrix_power(self.adjacency + np.eye(self.num_states), 2)
                matrix = self.nominal_transition * modifier
                matrix = np.clip(matrix, 1e-6, None)
                matrix /= matrix.sum(axis=1, keepdims=True)
                bank[(i, j)] = matrix
        return bank

    def _compute_xi(self, state_idx: int) -> float:
        xi = 0.0
        A = self.adjacency
        power = np.eye(self.num_states)
        for hop in self.lateral_hops:
            power = power @ A
            xi += (self.gamma ** hop) * power[state_idx].sum()
        normaliser = self.num_states * sum(self.gamma ** hop for hop in self.lateral_hops)
        return float(xi / normaliser)

    def _build_feature_vector(self, state_idx: int) -> np.ndarray:
        one_hot = np.zeros(self.num_states, dtype=np.float64)
        one_hot[state_idx] = 1.0
        degree = self.adjacency[state_idx]
        xi = self._compute_xi(state_idx)

        feature_list: List[float] = []
        feature_list.extend(one_hot.tolist())
        feature_list.extend((degree / degree.sum() if degree.sum() > 0 else degree).tolist())
        feature_list.append(xi)
        feature_list.extend(self.attacker_mix_stats.tolist())
        feature_list.extend(self.detection_rates.tolist())
        feature_list.extend(self.valuations.tolist())
        feature_list.extend([self.c_star, self.a_r, self.SQ, self.k_s, self.ASSC, self.AIC, self.C_r])
        return np.asarray(feature_list, dtype=np.float64)

    def _init_attack_component_weights(self) -> Dict[str, np.ndarray]:
        return {
            "overflow": np.array([0.6, 0.3, 0.1], dtype=np.float64),
            "destroy": np.array([0.2, 0.5, 0.3], dtype=np.float64),
            "lateral": np.array([0.3, 0.3, 0.4], dtype=np.float64),
        }

    def _init_defense_component_weights(self) -> Dict[str, np.ndarray]:
        return {
            "iphop": np.array([0.4, 0.3, 0.3], dtype=np.float64),
            "ip_proto_hop": np.array([0.3, 0.4, 0.3], dtype=np.float64),
            "time_rand": np.array([0.3, 0.2, 0.5], dtype=np.float64),
        }

    def reset(self) -> EnvState:
        self.node_activity.fill(0.0)
        self.attacker_mix_stats = np.full(self.num_attacker_actions, 1.0 / self.num_attacker_actions)
        self.current_state_idx = 0
        self.time_step = 0
        features = self._build_feature_vector(self.current_state_idx)
        return EnvState(features=features, index=self.current_state_idx)

    def sample_uncertainty(self, attacker_idx: int, defender_idx: int, count: int | None = None) -> list[Scenario]:
        base = self.base_transition_bank[(attacker_idx, defender_idx)]
        return sample_scenarios(
            base_transition=base,
            reward_bounds_def=self.reward_bounds_def,
            reward_bounds_att=self.reward_bounds_att,
            logit_params=self.logit_params,
            alpha=self.dirichlet_alpha,
            count=count or self.U_scenarios,
            rng=self.rng,
        )

    def _mu_vector(self) -> np.ndarray:
        decay = 0.9 ** np.arange(len(self.attacker_strategies))
        return self.detection_rates * (1.0 + 0.05 * decay.mean())

    def _attacker_capability(self, attacker_idx: int) -> float:
        strategy = self.attacker_strategies[attacker_idx]
        return self.attacker_capability.get(strategy, 0.5)

    def _attacker_operational_cost(self, attacker_idx: int) -> float:
        return 0.3 + 0.1 * attacker_idx

    def _guidance_penalty(self, attacker_idx: int) -> float:
        return 0.05 * (attacker_idx + 1)

    def _defender_incentive(self) -> float:
        return self.incentive_bias + 0.1 * np.sin(self.time_step)

    def _payoff_summary(
        self,
        state_idx: int,
        attacker_idx: int,
        defender_idx: int,
        incentive_val: float,
    ) -> payoffs.PayoffComponents:
        strategy_a = self.attacker_strategies[attacker_idx]
        strategy_d = self.defender_strategies[defender_idx]
        xi_val = self._compute_xi(state_idx)
        pi_i = self._attacker_capability(attacker_idx)
        mu_vec = self._mu_vector()
        lambda_vec = self.detection_rates
        return payoffs.summarise_payoffs(
            xi_z=xi_val,
            c_r=self.C_r,
            attacker_weights=self.attack_component_weights[strategy_a],
            defender_weights=self.defense_component_weights[strategy_d],
            valuations=self.valuations,
            mu_vec=mu_vec,
            c_star=self.c_star,
            a_r=self.a_r,
            pi_i=pi_i,
            lambda_vec=lambda_vec,
            sq=self.SQ,
            k_s=self.k_s,
            assc=self.ASSC,
            aic=self.AIC,
            operational_cost=self._attacker_operational_cost(attacker_idx),
            guidance_penalty=self._guidance_penalty(attacker_idx),
            alpha_I=self.alpha_I,
            incentive=incentive_val,
        )

    def step(
        self,
        attacker_idx: int,
        defender_idx: int,
        scenario: Scenario,
    ) -> tuple[EnvState, float, float, dict]:
        incentive_val = self._defender_incentive()
        summary = self._payoff_summary(self.current_state_idx, attacker_idx, defender_idx, incentive_val)
        defender_reward = payoffs.defender_payoff(
            summary.sap * scenario.defender_scale,
            summary.defender_cost,
            self.alpha_I,
            incentive_val,
        )
        attacker_reward = payoffs.attacker_payoff(
            summary.sal * scenario.attacker_scale,
            summary.attacker_cost,
            summary.guidance_penalty,
        )

        transition_matrix = scenario.transition
        probs = transition_matrix[self.current_state_idx]
        probs = probs / probs.sum()
        next_state_idx = int(self.rng.choice(self.num_states, p=probs))

        self.node_activity[next_state_idx] += 1.0
        self.attacker_mix_stats *= 0.9
        self.attacker_mix_stats[attacker_idx] += 0.1
        self.attacker_mix_stats /= self.attacker_mix_stats.sum()

        self.current_state_idx = next_state_idx
        self.time_step += 1
        node_names = list(self.graph.nodes)
        next_state_name = node_names[next_state_idx]
        is_absorbing = int(self.graph.nodes[next_state_name].get("terminal", False))
        self._recent_attack_outcomes.append(is_absorbing)
        if len(self._recent_attack_outcomes) > 512:
            self._recent_attack_outcomes = self._recent_attack_outcomes[-512:]
        self.last_sal = summary.sal
        features = self._build_feature_vector(self.current_state_idx)

        info = {
            "theta_xy": summary.theta_xy.copy(),
            "sal": summary.sal,
            "sap": summary.sap,
            "defender_cost": summary.defender_cost,
            "attacker_cost": summary.attacker_cost,
            "guidance_penalty": summary.guidance_penalty,
            "scenario_def_scale": scenario.defender_scale,
            "scenario_att_scale": scenario.attacker_scale,
            "alpha_I": self.alpha_I,
            "incentive": incentive_val,
        }

        return EnvState(features=features, index=self.current_state_idx), attacker_reward, defender_reward, info

    def base_transition(self, attacker_idx: int, defender_idx: int) -> np.ndarray:
        return self.base_transition_bank[(attacker_idx, defender_idx)]

    def feature_dim(self) -> int:
        return self._build_feature_vector(self.current_state_idx).shape[0]

    def action_spaces(self) -> tuple[int, int]:
        return self.num_attacker_actions, self.num_defender_actions

    def simulate_request_batch(
        self,
        mu: int,
        attack_mode: str,
        n_attackers: int,
        n_targets: int,
    ) -> dict[str, float]:
        concurrency = max(1, mu)
        base_service = 150.0 + 0.03 * concurrency
        rho = min(0.95, concurrency / 1200.0)
        queue_delay = 900.0 * rho / max(1e-3, 1.0 - rho)
        attack_factor = 25.0 * (n_attackers / max(1, n_targets))
        reconfig_latency = (self.a_r * self.c_star * 180.0) / max(1, n_targets)
        if attack_mode == "dynamic":
            reconfig_latency *= 1.15
            queue_delay *= 1.1
        resp_time = base_service + queue_delay + reconfig_latency + attack_factor
        load_time = resp_time + 0.1 * concurrency
        noise = float(self.rng.normal(0.0, 8.0))
        resp_time = max(5.0, resp_time + noise)
        load_time = max(5.0, load_time + 0.5 * noise)
        nc_val = payoffs.network_cost(self.SQ, self.a_r, self.k_s)
        return {
            "resp_time_ms": resp_time,
            "load_time_ms": load_time,
            "reconfig_latency_ms": max(1.0, reconfig_latency),
            "ASSC_ms": self.ASSC * 800.0,
            "AIC_ms": self.AIC * 700.0,
            "NC_ms": nc_val * 900.0,
            "cost_total_ms": resp_time + load_time + reconfig_latency,
        }
