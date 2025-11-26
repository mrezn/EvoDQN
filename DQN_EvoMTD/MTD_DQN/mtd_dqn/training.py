from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .dqn.agent import AgentConfig, DQNAgent
from .dqn.replay import ReplayBuffer
from .env_edgecloud import EdgeCloudMTDEnv
from .metrics.adapters import compute_asr_step, measure_request_response
from .metrics.schemas import (
    PerStrategyResponseCostRow,
    TrainingResponseTrendRow,
    schema_columns_per_strategy,
    schema_columns_training,
)
from .utils import ensure_dir, set_global_seed
from .wf_mtd import StageGameAverager, WrightFisherCoupler


@dataclass
class TrainSummary:
    logs_path: Path
    json_path: Path
    mix_history_path: Path
    mix_history: List[Dict[str, float]]
    complexity: Dict[str, str | float]
    response_trend_path: Path
    per_strategy_path: Path


def build_agent(config: dict, state_dim: int, action_dim: int) -> DQNAgent:
    dqn_cfg = config.get("dqn", {})
    agent_cfg = AgentConfig(
        gamma=float(dqn_cfg.get("gamma", 0.99)),
        tau_soft=float(dqn_cfg.get("tau_soft", 0.005)),
        tau_D=float(dqn_cfg.get("tau_D", 0.7)),
        beta_ent=float(dqn_cfg.get("beta_ent", 0.001)),
        huber_kappa=float(dqn_cfg.get("huber_kappa", 1.0)),
        weight_decay=float(dqn_cfg.get("weight_decay", 1e-5)),
        lr=float(dqn_cfg.get("lr", 3e-4)),
    )
    hidden_dims = dqn_cfg.get("hidden_dims", [256, 256])
    layer_norm = bool(dqn_cfg.get("layer_norm", True))
    return DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        config=agent_cfg,
        layer_norm=layer_norm,
    )


def _compute_stage_proxies(
    agent: DQNAgent,
    summary: Dict[str, float],
    scenarios: List,
    next_state_features: np.ndarray,
    gamma: float,
    done: bool,
) -> tuple[float, float]:
    next_state_tensor = torch.as_tensor(next_state_features, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        max_q_next = torch.max(agent.target(next_state_tensor)).item()
    discount = 0.0 if done else gamma
    defender_values = []
    attacker_values = []
    for scenario in scenarios:
        reward_def = agent._defender_reward_from_summary(summary, scenario)
        reward_att = agent._attacker_reward_from_summary(summary, scenario)
        defender_values.append(reward_def + discount * max_q_next)
        attacker_values.append(reward_att)
    return float(np.min(defender_values)), float(np.min(attacker_values))


def train(
    config: dict,
    output_dirs: Dict[str, Path],
) -> tuple[DQNAgent, EdgeCloudMTDEnv, WrightFisherCoupler, TrainSummary]:
    seeds_cfg = config.get("seeds", {})
    set_global_seed(seeds_cfg.get("global_seed", 1337), seeds_cfg.get("torch_deterministic", True))

    env = EdgeCloudMTDEnv(config)
    state = env.reset()
    state_dim = state.features.shape[0]
    _, defender_actions = env.action_spaces()

    agent = build_agent(config, state_dim, defender_actions)

    wf_cfg = config.get("env", {}).get("bounded_rationality", {})
    wf_step = float(wf_cfg.get("eta", 0.15))
    omega_A = float(wf_cfg.get("omega_A", 0.6))
    omega_D = float(wf_cfg.get("omega_D", 0.7))
    coupler = WrightFisherCoupler(
        attacker_strategies=env.attacker_strategies,
        defender_strategies=env.defender_strategies,
        omega_attacker=omega_A,
        omega_defender=omega_D,
        eta=wf_step,
    )

    replay_cfg = config.get("replay", {})
    buffer = ReplayBuffer(capacity=replay_cfg.get("capacity", 20000))
    stage_averager = StageGameAverager(env.num_attacker_actions, env.num_defender_actions)

    train_cfg = config.get("training", {})
    episodes = int(train_cfg.get("episodes", 200))
    steps_per_episode = int(train_cfg.get("steps_per_episode", 50))
    save_every = int(train_cfg.get("save_every", 50))

    warmup_steps = int(replay_cfg.get("warmup_steps", 512))
    update_freq = int(replay_cfg.get("update_freq", 4))
    batch_size = int(replay_cfg.get("batch_size", 64))

    wf_period = int(config.get("env", {}).get("wf", {}).get("K_WF", 5))

    logs_dir = ensure_dir(output_dirs["logs"])
    checkpoints_dir = ensure_dir(output_dirs["checkpoints"])
    resp_dir = Path("comp-perf/dqn_evomtd/response_time")
    resp_dir.mkdir(parents=True, exist_ok=True)
    trend_path = resp_dir / "training_response_trend.csv"
    per_strategy_path = resp_dir / "per_strategy_response_cost.csv"

    log_rows: List[Dict[str, float]] = []
    mix_history: List[Dict[str, float]] = []
    update_durations: List[float] = []
    trend_rows: List[Dict[str, object]] = []

    total_steps = 0
    last_metrics = {"loss": float("nan"), "entropy": float("nan")}

    for episode in range(episodes):
        env_state = env.reset()
        per_strategy_episode: Dict[
            Tuple[str, int, str, int, int],
            Dict[str, List[float]],
        ] = defaultdict(lambda: defaultdict(list))
        for step in range(steps_per_episode):
            attacker_idx = np.random.choice(env.num_attacker_actions, p=coupler.mix_attacker)
            defender_idx = agent.select_action(env_state.features, mode="train")
            scenarios = env.sample_uncertainty(attacker_idx, defender_idx)
            next_state, attacker_payoff, defender_payoff, summary = env.step(attacker_idx, defender_idx, scenarios[0])
            done = step == steps_per_episode - 1

            buffer.add(
                state=env_state.features,
                action=defender_idx,
                reward=defender_payoff,
                next_state=next_state.features,
                done=done,
                attacker_action=attacker_idx,
                summary=summary,
                scenarios=scenarios,
            )

            if len(buffer) >= warmup_steps and total_steps % update_freq == 0:
                start = time.perf_counter()
                metrics = agent.update(buffer, batch_size)
                update_durations.append(time.perf_counter() - start)
                last_metrics = metrics
            defender_proxy, attacker_proxy = _compute_stage_proxies(
                agent,
                summary,
                scenarios,
                next_state.features,
                agent.config.gamma,
                done,
            )
            stage_averager.add(attacker_idx, defender_idx, attacker_proxy, defender_proxy)

            row = {
                "episode": episode,
                "step": step,
                "attacker_idx": attacker_idx,
                "defender_idx": defender_idx,
                "attacker_payoff": attacker_payoff,
                "defender_payoff": defender_payoff,
                "defender_proxy": defender_proxy,
                "attacker_proxy": attacker_proxy,
                "loss": last_metrics["loss"],
                "entropy": last_metrics["entropy"],
            }
            for i, prob in enumerate(coupler.mix_attacker):
                row[f"p_{env.attacker_strategies[i]}"] = prob
            for j, prob in enumerate(coupler.mix_defender):
                row[f"q_{env.defender_strategies[j]}"] = prob
            if (episode + 1) % 100 == 0:
                log_rows.append(row)

            env_state = next_state
            total_steps += 1

            mu_schedule = [200, 400, 600, 800]
            segment = max(1, steps_per_episode // len(mu_schedule))
            mu = mu_schedule[min(len(mu_schedule) - 1, step // segment)]
            attack_mode = "dynamic" if attacker_idx % 2 else "static"
            n_attackers = max(1, attacker_idx + 1)
            n_targets = 1 + (env.current_state_idx % 3)
            metrics = measure_request_response(env, mu, attack_mode, n_attackers, n_targets)
            asr_est = compute_asr_step(env, window=50)
            def_strategy = f"GD{defender_idx + 1}"
            trend_rows.append(
                TrainingResponseTrendRow(
                    episode=episode,
                    step=step,
                    mu=mu,
                    attack_mode=attack_mode,
                    n_attackers=n_attackers,
                    n_targets=n_targets,
                    def_strategy=def_strategy,
                    resp_time_ms=float(metrics["resp_time_ms"]),
                    load_time_ms=float(metrics["load_time_ms"]),
                    reconfig_latency_ms=float(metrics["reconfig_latency_ms"]),
                    ASSC_ms=float(metrics["ASSC_ms"]),
                    AIC_ms=float(metrics["AIC_ms"]),
                    NC_ms=float(metrics["NC_ms"]),
                    cost_total_ms=float(metrics["cost_total_ms"]),
                    reward=float(defender_payoff),
                    ASR=float(asr_est),
                    SAL=float(summary.get("sal", 0.0)),
                    UD=float(defender_payoff),
                ).to_dict()
            )
            if len(trend_rows) >= 100:
                _append_rows(trend_path, trend_rows, schema_columns_training())
                trend_rows.clear()

            strat_key = (def_strategy, mu, attack_mode, n_attackers, n_targets)
            strat_stats = per_strategy_episode[strat_key]
            for key, value in metrics.items():
                strat_stats.setdefault(key, []).append(float(value))

        if (episode + 1) % wf_period == 0:
            a_matrix, b_matrix = stage_averager.as_matrices()
            coupler.update(a_matrix, b_matrix)
            stage_averager.reset()
            mix_history.append(
                {
                    "episode": episode + 1,
                    "attacker_mix": coupler.mix_attacker.tolist(),
                    "defender_mix": coupler.mix_defender.tolist(),
                }
            )

        if (episode + 1) % save_every == 0:
            checkpoint_path = checkpoints_dir / f"agent_ep{episode + 1}.pt"
            torch.save({"episode": episode + 1, **agent.state_dict()}, checkpoint_path)

        if per_strategy_episode:
            per_rows = []
            for (def_strategy, mu, attack_mode, n_attackers, n_targets), stats in per_strategy_episode.items():
                per_rows.append(
                    PerStrategyResponseCostRow(
                        def_strategy=def_strategy,
                        mu=mu,
                        attack_mode=attack_mode,
                        n_attackers=n_attackers,
                        n_targets=n_targets,
                        resp_time_ms_mean=float(np.mean(stats["resp_time_ms"])),
                        resp_time_ms_p95=float(np.percentile(stats["resp_time_ms"], 95)),
                        load_time_ms_mean=float(np.mean(stats["load_time_ms"])),
                        load_time_ms_p95=float(np.percentile(stats["load_time_ms"], 95)),
                        ASSC_ms_mean=float(np.mean(stats["ASSC_ms"])),
                        AIC_ms_mean=float(np.mean(stats["AIC_ms"])),
                        NC_ms_mean=float(np.mean(stats["NC_ms"])),
                        cost_total_ms_mean=float(np.mean(stats["cost_total_ms"])),
                    ).to_dict()
                )
            _append_rows(per_strategy_path, per_rows, schema_columns_per_strategy())
            per_strategy_episode.clear()

    logs_df = pd.DataFrame(log_rows)
    logs_path = logs_dir / "train_metrics.csv"
    logs_df.to_csv(logs_path, index=False)
    json_path = logs_dir / "train_metrics.json"
    json_path.write_text(json.dumps(log_rows, indent=2), encoding="utf-8")
    mix_history_path = logs_dir / "mix_history.json"
    mix_history_path.write_text(json.dumps(mix_history, indent=2), encoding="utf-8")

    complexity = {
        "mean_update_time_seconds": float(np.mean(update_durations)) if update_durations else 0.0,
        "per_update_theoretical": f"O(B * (U + H)) with B={batch_size}, U={env.U_scenarios}, H={sum(config.get('dqn', {}).get('hidden_dims', [256, 256]))}",
        "memory_complexity": f"O(B * (state_dim + U)) with state_dim={state_dim}",
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
    }

    summary = TrainSummary(
        logs_path=logs_path,
        json_path=json_path,
        mix_history_path=mix_history_path,
        mix_history=mix_history,
        complexity=complexity,
        response_trend_path=trend_path,
        per_strategy_path=per_strategy_path,
    )
    return agent, env, coupler, summary


def _append_rows(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = np.nan
    df = df[columns]
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)

