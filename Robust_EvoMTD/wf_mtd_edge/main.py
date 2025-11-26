
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import typer
import numpy as np

import results as results_module
from config import load_config
from evolution import EvolutionEngine
from plots import plot_trajectories
from comp_perf import StepLog, compute_comp_perf


def _build_step_logs(engine: EvolutionEngine) -> tuple[List[StepLog], List[List[float]], List[List[float]]]:
    history = engine.history_data or {}
    p_arr = np.array(history.get("p", np.empty((0, 0))))
    q_arr = np.array(history.get("q", np.empty((0, 0))))
    ua_arr = np.array(history.get("Ua", np.empty(0)))
    ud_arr = np.array(history.get("Ud", np.empty(0)))
    xi_arr = np.array(history.get("xi_m", np.empty(0)))

    if p_arr.size == 0 or q_arr.size == 0:
        return [], [], []

    attack_names = engine.config.strategies.attack
    defense_names = engine.config.strategies.defense
    states = engine.config.kernel.states
    defense_costs = engine.config.defense_costs

    def _nc_for(defense: str) -> float:
        return engine.config.mtd.SQ * (
            1.0 - 1.0 / (1.0 + np.exp(-(engine.config.mtd.a_for(defense) - engine.config.mtd.k)))
        )

    step_logs: List[StepLog] = []
    q_hist: List[List[float]] = []
    p_hist: List[List[float]] = []

    steps = min(len(p_arr), len(q_arr), len(engine.logs))
    for idx in range(steps):
        p_vec = p_arr[idx]
        q_vec = q_arr[idx]
        xi_val = float(xi_arr[idx]) if idx < len(xi_arr) else 0.0
        sal_acc = 0.0
        sap_acc = 0.0
        dc_acc = 0.0
        theta_acc = 0.0

        for i, attack_name in enumerate(attack_names):
            for j, defense_name in enumerate(defense_names):
                _, _, components = engine.payoff_engine.stage_utilities(i, j, xi_val)
                weight = float(p_vec[i] * q_vec[j])
                sal_acc += components.sal * weight
                sap_acc += components.sap * weight
                dc_acc += components.dc * weight
                theta_acc += components.theta_mean * weight

        assc_exp = float(
            sum(q_vec[j] * defense_costs.get(defense_names[j], {}).get("ASSC", 0.0) for j in range(len(defense_names)))
        )
        aic_exp = float(
            sum(q_vec[j] * defense_costs.get(defense_names[j], {}).get("AIC", 0.0) for j in range(len(defense_names)))
        )
        nc_exp = float(sum(q_vec[j] * _nc_for(defense_names[j]) for j in range(len(defense_names))))

        log_entry = engine.logs[idx]
        rho = np.array(log_entry.rho)
        state_idx = int(np.argmax(rho)) if rho.size else 0
        state_name = states[state_idx] if state_idx < len(states) else f"state_{state_idx}"
        absorbing_state = states[-1] if states else "terminal"
        is_absorbing = int(state_name == absorbing_state)
        as_success = int(theta_acc >= 0.5)

        step_logs.append(
            StepLog(
                t=int(log_entry.step),
                state=state_name,
                attacker_action=attack_names[int(np.argmax(p_vec))],
                defender_action=defense_names[int(np.argmax(q_vec))],
                UD=float(ud_arr[idx]) if idx < len(ud_arr) else 0.0,
                UA=float(ua_arr[idx]) if idx < len(ua_arr) else 0.0,
                SAP=float(sap_acc),
                SAL=float(sal_acc),
                DC=float(dc_acc),
                ASSC=assc_exp,
                NC=nc_exp,
                AIC=aic_exp,
                AS_success=as_success,
                is_absorbing=is_absorbing,
                power_W=None,
                q_vec=q_vec.tolist(),
                p_vec=p_vec.tolist(),
                td_loss=None,
                entropy=None,
            )
        )
        q_hist.append(q_vec.tolist())
        p_hist.append(p_vec.tolist())

    return step_logs, q_hist, p_hist

try:  # Optional dependency for the additional figures
    from results_additional_wfmt_no_dqn import generate_additional_results
except ImportError:  # pragma: no cover - optional runtime dependency
    generate_additional_results = None  # type: ignore[assignment]

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, readable=True, help="YAML configuration file."),
    mode: Optional[str] = typer.Option(None, help="Simulation mode: bayes or robust."),
    steps: Optional[int] = typer.Option(None, help="Override number of WF steps."),
    seed: Optional[int] = typer.Option(None, help="Override random seed."),
    plot: bool = typer.Option(False, help="Generate matplotlib plots."),
    extra_out: Optional[Path] = typer.Option(None, help="Directory for additional WF-MTD figures."),
    extra_episodes: Optional[int] = typer.Option(None, help="Episodes for additional figures (default 120)."),
    extra_robust_samples: Optional[int] = typer.Option(None, help="Robust samples for additional figures (default 64)."),
    extra_eta: Optional[float] = typer.Option(None, help="WF step size η for additional figures."),
    extra_omega: Optional[float] = typer.Option(None, help="Baseline ω for additional bounded rational runs."),
    extra_window: Optional[int] = typer.Option(None, help="Convergence window for additional figures (default 10)."),
) -> None:
    """Execute the robust WF-MTD simulation specified by *config*."""

    model_config = load_config(config)
    if mode:
        model_config.run.mode = mode
    if steps:
        model_config.run.steps = steps
    if seed:
        model_config.run.seed = seed

    engine = EvolutionEngine(model_config)
    engine.run(model_config.run.mode, model_config.run.steps)
    summary = engine.summary()

    typer.echo("Attacker equilibrium mix:")
    for name, prob in summary["attacker_mix"].items():
        typer.echo(f"  {name}: {prob:.4f}")
    typer.echo("Defender equilibrium mix:")
    for name, prob in summary["defender_mix"].items():
        typer.echo(f"  {name}: {prob:.4f}")
    typer.echo(f"Attacker payoff: {summary['attacker_payoff']:.4f}")
    typer.echo(f"Defender payoff: {summary['defender_payoff']:.4f}")

    if plot:
        plot_trajectories(engine.logs, model_config)

    if getattr(engine, "dp_result", None) and getattr(engine, "history_data", None):
        try:
            if model_config.run.log_path:
                base_dir = Path(model_config.run.log_path).parent
            elif model_config.run.trajectory_path:
                base_dir = Path(model_config.run.trajectory_path).parent
            else:
                base_dir = config.parent
            results_dir = base_dir / "results"

            state_count = len(model_config.kernel.states)
            paths = ["default"]
            states_per_path = {"default": list(range(state_count))}

            payoff_matrices = {
                ("default", state_idx): {
                    "A": engine.dp_result.tilde_a[state_idx],
                    "D": engine.dp_result.tilde_b[state_idx],
                }
                for state_idx in range(state_count)
            }

            history_template = engine.history_data
            theta_template = history_template.get("theta", {})
            evolution_history = {}
            for state_idx in range(state_count):
                evolution_history[("default", state_idx)] = {
                    "p": history_template["p"],
                    "q": history_template["q"],
                    "fa": history_template["fa"],
                    "fd": history_template["fd"],
                    "Ua": history_template["Ua"],
                    "Ud": history_template["Ud"],
                    "xi_m": history_template["xi_m"],
                    "theta": {key: theta_template[key] for key in theta_template},
                }

            transitions = engine.dp_result.transitions
            state_kernel = {
                ("default", state_idx): transitions[state_idx].transpose(2, 0, 1)
                for state_idx in range(state_count)
            }

            results_config = {
                "out_dir": results_dir.as_posix(),
                "paths": paths,
                "states_per_path": states_per_path,
                "attacker_strategies": model_config.strategies.attack,
                "defender_strategies": model_config.strategies.defense,
                "plot": {"dpi": 160, "fontsize": 11},
            }

            artifacts = results_module.generate_all_results(
                config=results_config,
                payoff_matrices=payoff_matrices,
                evolution_history=evolution_history,
                state_kernel=state_kernel,
                sensitivity=None,
            )
            typer.echo(f"Results artifacts generated: {artifacts['report']}")
        except Exception as exc:  # pragma: no cover - best effort
            typer.echo(f"Results generation failed: {exc}")

    if extra_out and generate_additional_results:
        try:
            extra_summary = generate_additional_results(
                out_dir=str(extra_out),
                episodes=extra_episodes or 120,
                robust_samples=extra_robust_samples or 64,
                eta=extra_eta or model_config.run.eta,
                omega=extra_omega or model_config.run.omega_d,
                window=extra_window or 10,
            )
            typer.echo(f"Additional WF-MTD figures saved to {extra_out}")
            typer.echo(f"Additional summary keys: {', '.join(sorted(extra_summary.keys()))}")
        except Exception as exc:  # pragma: no cover - optional feature
            typer.echo(f"Additional figure generation failed: {exc}")
    elif extra_out:
        typer.echo("Additional results module is unavailable; skipping extra plots.")

    step_logs, q_hist, p_hist = _build_step_logs(engine)
    comp_dir = Path(__file__).resolve().parent / "comp-perf"
    summary = compute_comp_perf(
        steps=step_logs,
        q_hist=q_hist,
        p_hist=p_hist,
        sal_baseline=None,
        power_train_energy_j=None,
        uncertainty_sweep=None,
        out_dir=comp_dir,
        model_name="WF-MTD",
    )
    typer.echo(f"Comparative performance artifacts written to {comp_dir}")
    typer.echo(f"Summary snapshot: {summary}")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
