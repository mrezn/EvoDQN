from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .dqn.agent import DQNAgent
from .env_edgecloud import EdgeCloudMTDEnv
from .comp_perf import StepLog
from . import payoffs
from .utils import ensure_dir
from .wf_mtd import WrightFisherCoupler


def evaluate(
    agent: DQNAgent,
    env: EdgeCloudMTDEnv,
    coupler: WrightFisherCoupler,
    config: dict,
    output_dirs: Dict[str, Path],
    mix_history: List[Dict[str, List[float]]] | None = None,
) -> Tuple[Path, List[StepLog], List[List[float]], List[List[float]]]:
    train_cfg = config.get("training", {})
    episodes = int(train_cfg.get("evaluation_every", 10))
    steps_per_episode = int(train_cfg.get("steps_per_episode", 50))

    defender_returns: List[float] = []
    attacker_returns: List[float] = []
    step_logs: List[StepLog] = []
    q_hist: List[List[float]] = []
    p_hist: List[List[float]] = []
    step_index = 0
    node_names = list(env.graph.nodes)
    nc_val = payoffs.network_cost(env.SQ, env.a_r, env.k_s)

    for _ in range(episodes):
        state = env.reset()
        defender_total = 0.0
        attacker_total = 0.0
        for _ in range(steps_per_episode):
            attacker_idx = np.random.choice(env.num_attacker_actions, p=coupler.mix_attacker)
            defender_idx = agent.select_action(state.features, mode="eval")
            scenarios = env.sample_uncertainty(attacker_idx, defender_idx)
            prev_state = state
            next_state, attacker_reward, defender_reward, info = env.step(attacker_idx, defender_idx, scenarios[0])
            defender_total += defender_reward
            attacker_total += attacker_reward
            scenario = scenarios[0]
            state_name = node_names[prev_state.index]
            next_state_name = node_names[next_state.index]
            is_absorbing = int(env.graph.nodes[next_state_name].get("terminal", False))
            q_hist.append(coupler.mix_defender.tolist())
            p_hist.append(coupler.mix_attacker.tolist())
            q_vec = coupler.mix_defender.tolist()
            p_vec = coupler.mix_attacker.tolist()
            sal_scaled = (info.get("sal", 0.0) or 0.0) * scenario.attacker_scale
            sap_scaled = (info.get("sap", 0.0) or 0.0) * scenario.defender_scale
            step_logs.append(
                StepLog(
                    t=step_index,
                    state=state_name,
                    attacker_action=env.attacker_strategies[attacker_idx],
                    defender_action=env.defender_strategies[defender_idx],
                    UD=float(defender_reward),
                    UA=float(attacker_reward),
                    SAP=float(sap_scaled),
                    SAL=float(sal_scaled),
                    DC=float(info.get("defender_cost", env.ASSC + nc_val + env.AIC)),
                    ASSC=float(env.ASSC),
                    NC=float(nc_val),
                    AIC=float(env.AIC),
                    AS_success=is_absorbing,
                    is_absorbing=is_absorbing,
                    power_W=None,
                    q_vec=q_vec,
                    p_vec=p_vec,
                    td_loss=None,
                    entropy=None,
                )
            )
            step_index += 1
            state = next_state
        defender_returns.append(defender_total / steps_per_episode)
        attacker_returns.append(attacker_total / steps_per_episode)

    convergence_metric = None
    if mix_history and len(mix_history) >= 2:
        prev = np.array(mix_history[-2]["defender_mix"], dtype=float)
        last = np.array(mix_history[-1]["defender_mix"], dtype=float)
        convergence_metric = float(np.linalg.norm(last - prev, ord=1))

    summary = {
        "defender_return_mean": float(np.mean(defender_returns)),
        "defender_return_std": float(np.std(defender_returns)),
        "attacker_return_mean": float(np.mean(attacker_returns)),
        "attacker_return_std": float(np.std(attacker_returns)),
        "convergence_l1": convergence_metric,
    }

    logs_dir = ensure_dir(output_dirs["logs"])
    eval_path = logs_dir / "eval_summary.json"
    eval_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return eval_path, step_logs, q_hist, p_hist

