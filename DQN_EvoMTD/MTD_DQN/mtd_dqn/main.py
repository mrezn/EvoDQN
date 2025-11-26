from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml

from . import EdgeCloudMTDEnv, WrightFisherCoupler, set_global_seed
from .evaluation import evaluate
from .plotting import plot_all, plot_training_convergence
from .comp_perf import compute_comp_perf
from .tables import export_tables
from .training import TrainSummary, build_agent, train
from .utils import ensure_dir


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_output_dirs(config: dict) -> Dict[str, Path]:
    output_cfg = config.get("output", {})
    base_dir = ensure_dir(output_cfg.get("base_dir", "results"))
    dirs = {
        "base": base_dir,
        "figures": ensure_dir(output_cfg.get("figures_dir", base_dir / "figures")),
        "tables": ensure_dir(output_cfg.get("tables_dir", base_dir / "tables")),
        "logs": ensure_dir(output_cfg.get("logs_dir", base_dir / "logs")),
        "checkpoints": ensure_dir(output_cfg.get("checkpoints_dir", base_dir / "checkpoints")),
    }
    return dirs


def _build_coupler(config: dict, env: EdgeCloudMTDEnv) -> WrightFisherCoupler:
    wf_cfg = config.get("env", {}).get("bounded_rationality", {})
    return WrightFisherCoupler(
        attacker_strategies=env.attacker_strategies,
        defender_strategies=env.defender_strategies,
        omega_attacker=float(wf_cfg.get("omega_A", 0.6)),
        omega_defender=float(wf_cfg.get("omega_D", 0.7)),
        eta=float(wf_cfg.get("eta", 0.15)),
    )


def _load_latest_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoints = sorted(checkpoints_dir.glob("agent_ep*.pt"))
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found. Run training mode first.")
    return checkpoints[-1]


def _load_mix_history(logs_dir: Path) -> list:
    path = logs_dir / "mix_history.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def run_train_mode(config: dict, output_dirs: Dict[str, Path]) -> TrainSummary:
    agent, env, coupler, summary = train(config, output_dirs)
    mix_history = summary.mix_history
    eval_path, step_logs, q_hist, p_hist = evaluate(agent, env, coupler, config, output_dirs, mix_history=mix_history)
    comp_dir = Path(__file__).resolve().parent / "comp-perf"
    comp_summary = compute_comp_perf(
        steps=step_logs,
        q_hist=q_hist,
        p_hist=p_hist,
        sal_baseline=None,
        power_train_energy_j=None,
        uncertainty_sweep=None,
        out_dir=comp_dir,
        model_name="DQN-WF-MTD",
    )
    print(f"Saved comparative perf to: {comp_dir}")
    print(f"Summary: {comp_summary}")
    plot_cfg = config.get("plotting", {})
    dpi = int(plot_cfg.get("dpi", 200))
    style = plot_cfg.get("style", "default")
    plot_training_convergence(summary.logs_path, output_dirs["figures"], dpi, style)
    plot_all(config, output_dirs, coupler, summary.logs_path)
    export_tables(config, output_dirs)
    return summary


def run_eval_mode(config: dict, output_dirs: Dict[str, Path]) -> None:
    env = EdgeCloudMTDEnv(config)
    state = env.reset()
    _, defender_actions = env.action_spaces()
    agent = build_agent(config, state.features.shape[0], defender_actions)
    coupler = _build_coupler(config, env)

    checkpoint_path = _load_latest_checkpoint(output_dirs["checkpoints"])
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    agent.load_state_dict(checkpoint)

    mix_history = _load_mix_history(output_dirs["logs"])
    _, step_logs, q_hist, p_hist = evaluate(agent, env, coupler, config, output_dirs, mix_history=mix_history)
    comp_dir = Path(__file__).resolve().parent / "comp-perf"
    comp_summary = compute_comp_perf(
        steps=step_logs,
        q_hist=q_hist,
        p_hist=p_hist,
        sal_baseline=None,
        power_train_energy_j=None,
        uncertainty_sweep=None,
        out_dir=comp_dir,
        model_name="DQN-WF-MTD",
    )
    print(f"Saved comparative perf to: {comp_dir}")
    print(f"Summary: {comp_summary}")


def run_plot_mode(config: dict, output_dirs: Dict[str, Path]) -> None:
    env = EdgeCloudMTDEnv(config)
    coupler = _build_coupler(config, env)
    plot_all(config, output_dirs, coupler, output_dirs["logs"] / "train_metrics.csv")


def run_tables_mode(config: dict, output_dirs: Dict[str, Path]) -> None:
    export_tables(config, output_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dueling DQN")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--mode", choices=["train", "eval", "plot", "tables"], default="train")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dirs = prepare_output_dirs(config)

    if args.mode == "train":
        summary = run_train_mode(config, output_dirs)
        complexity_lines = [
            "Complexity report:",
            f" per-update (theoretical): {summary.complexity['per_update_theoretical']}",
            f" memory footprint: {summary.complexity['memory_complexity']}",
            f" episodes: {summary.complexity['episodes']}",
            f" steps/episode: {summary.complexity['steps_per_episode']}",
            f" mean update time [s]: {summary.complexity['mean_update_time_seconds']:.6f}",
        ]
        print("\n".join(complexity_lines))
    elif args.mode == "eval":
        run_eval_mode(config, output_dirs)
    elif args.mode == "plot":
        run_plot_mode(config, output_dirs)
    elif args.mode == "tables":
        run_tables_mode(config, output_dirs)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
