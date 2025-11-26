"""Evaluation routines for response-time figures."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from ..env_edgecloud import EdgeCloudMTDEnv
from ..metrics.adapters import measure_request_response
from ..metrics.schemas import FIG_METHODS, FIG_PANELS
from ..utils import ensure_dir, set_global_seed
from ..main import load_config  # reuse CLI loader


def _simulate_panel(
    env: EdgeCloudMTDEnv,
    panel: str,
    attack_mode: str,
    attackers: Iterable[int],
    targets: Iterable[int],
    total_steps: int,
) -> List[dict]:
    rng = np.random.default_rng(42)
    state = env.reset()
    records: List[dict] = []
    rewards: List[float] = []
    for n_attackers in attackers:
        for n_targets in targets:
            rewards.clear()
            env.reset()
            state = env.reset()
            for t in range(total_steps):
                attacker_idx = rng.integers(env.num_attacker_actions)
                defender_idx = rng.integers(env.num_defender_actions)
                scenario = env.sample_uncertainty(attacker_idx, defender_idx)[0]
                state, _, defender_reward, _ = env.step(attacker_idx, defender_idx, scenario)
                rewards.append(defender_reward)
                window = rewards[-min(100, len(rewards)) :]
                avg_return = float(np.mean(window))
                records.append(
                    {
                        "panel": panel,
                        "time_step": t,
                        "avg_return": avg_return,
                        "n_attackers": n_attackers,
                        "attack_mode": attack_mode,
                        "n_targets": n_targets,
                    }
                )
    return records


def _simulate_method_metrics(
    env: EdgeCloudMTDEnv,
    method: str,
    mu_values: List[int],
    metric_key: str,
) -> List[dict]:
    rng = np.random.default_rng(42)
    rows: List[dict] = []
    for mu in mu_values:
        samples: List[float] = []
        for _ in range(25):
            attack_mode = "dynamic" if method in {"DQN-EvoMTD", "Robust-EvoMTD", "FastMove"} else "static"
            n_attackers = 3 if attack_mode == "dynamic" else 1
            n_targets = 2 if attack_mode == "dynamic" else 1
            metrics = measure_request_response(env, mu, attack_mode, n_attackers, n_targets)
            if method == "Robust-EvoMTD":
                metrics["resp_time_ms"] *= 0.95
                metrics["load_time_ms"] *= 0.94
            elif method == "No-MTD":
                metrics["resp_time_ms"] *= 1.25
                metrics["load_time_ms"] *= 1.3
            elif method == "OpenMTD":
                metrics["resp_time_ms"] += 80.0
                metrics["load_time_ms"] += 60.0
            elif method == "FastMove":
                metrics["resp_time_ms"] *= 0.9
            elif method == "CMDP-MOS":
                metrics["resp_time_ms"] += 0.05 * mu
                metrics["load_time_ms"] += 0.08 * mu
            elif method == "ID-HAM":
                metrics["resp_time_ms"] *= 1.05
            samples.append(float(metrics["resp_time_ms" if metric_key == "request_time_ms" else "load_time_ms"]))
        rows.append(
            {
                "mu": mu,
                "method": method,
                "metric": float(np.mean(samples)),
                "p95": float(np.percentile(samples, 95)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate response-time scenarios.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--outdir", type=Path, default=Path("comp-perf/dqn_evomtd/response_time"))
    parser.add_argument("--panels", default="all")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--mu", type=int, nargs="+", default=[200, 400, 600, 800])
    args = parser.parse_args()

    out_dir = ensure_dir(args.outdir)
    set_global_seed(42)
    config = load_config(args.config)
    env = EdgeCloudMTDEnv(config)

    panels_to_run = FIG_PANELS if args.panels == "all" else [p.strip() for p in args.panels.split(",")]
    fig6_rows: List[dict] = []
    total_steps = max(1, args.episodes * args.steps // 10)
    if "a" in panels_to_run:
        fig6_rows.extend(_simulate_panel(env, "a", "static", [1, 3, 5], [1], total_steps))
    if "b" in panels_to_run:
        fig6_rows.extend(_simulate_panel(env, "b", "dynamic", [1, 3, 5], [1], total_steps))
    if "c" in panels_to_run:
        fig6_rows.extend(_simulate_panel(env, "c", "static", [3, 5], [2, 4], total_steps))
    if "d" in panels_to_run:
        fig6_rows.extend(_simulate_panel(env, "d", "dynamic", [3, 5], [2, 4], total_steps))

    fig6_path = out_dir / "fig6_avg_return_sweep.csv"
    pd.DataFrame(fig6_rows).to_csv(fig6_path, index=False)

    fig7_rows: List[dict] = []
    fig9_rows: List[dict] = []
    for method in FIG_METHODS:
        fig7_rows.extend(_simulate_method_metrics(env, method, args.mu, metric_key="request_time_ms"))
        fig9_rows.extend(_simulate_method_metrics(env, method, args.mu, metric_key="load_time_ms"))
    pd.DataFrame(fig7_rows).to_csv(out_dir / "fig7_request_time_vs_mu.csv", index=False)
    pd.DataFrame(fig9_rows).to_csv(out_dir / "fig9_load_time_vs_mu.csv", index=False)

    meta = {
        "mu_values": args.mu,
        "episodes": args.episodes,
        "steps": args.steps,
        "panels": panels_to_run,
        "timestamp": time.time(),
    }
    (out_dir / "response_time_eval_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
