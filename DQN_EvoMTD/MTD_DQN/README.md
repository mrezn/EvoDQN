# DQN-EvoMTD & WF-MTD Edge Repository

This repository combines two complementary Moving Target Defense (MTD) implementations:

1. **`MTD_DQN/`** – A Dueling Double-DQN defender coupled with Wright–Fisher (WF) evolutionary dynamics under reward/transition uncertainty. Includes response-time instrumentation, plotting scripts, and evaluation tooling.
2. **`wf_mtd_edge/`** – A robust WF-MTD edge/cloud simulator driven by evolutionary stage games and optional additional figures.

Both projects can be developed independently while sharing comparison utilities (`comp-perf/`).

---

## 1. Installation

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
pip install -r MTD_DQN/requirements.txt
```

The `requirements.txt` lists the superset of dependencies required by both codebases (PyTorch, numpy, pandas, matplotlib, networkx, PyYAML, scipy, tqdm, typer).

---

## 2. Directory overview

```
MTD_DQN/
  configs/                # YAML configs for the DQN trainer/evaluator
  mtd_dqn/
    __init__.py
    main.py               # CLI entrypoint (train/eval/plot/tables)
    training.py           # Dueling DQN + WF coupling loop with response-time logging
    env_edgecloud.py      # Edge/cloud environment (simulate_request_batch, etc.)
    payoffs.py, ...       # Payoff and uncertainty utilities
    metrics/
      schemas.py          # CSV schemas for response-time artifacts
      adapters.py         # Synthetic timing + ASR adapters
    scripts/
      eval_response_time.py
      plot_response_time.py
      plot_per_strategy_costs.py
    comp_perf.py          # Comparative performance exporter (uncertainty-ready)
  results/
    figures/, tables/, logs/, checkpoints/
wf_mtd_edge/
  main.py                 # Typer CLI for WF-MTD edge simulations
  comp_perf.py            # WF counterpart exporter
  configs/, evolution.py, environment.py, ...
comp-perf/
  dqn_evomtd/response_time/  # Response-time CSV/PNG outputs + README
  ...                        # (other comparison folders)
```

---

## 3. Running the DQN WF–MTD defender

```bash
cd MTD_DQN
python -m mtd_dqn.main --config configs/base.yaml --mode train
python -m mtd_dqn.main --config configs/base.yaml --mode eval
python -m mtd_dqn.main --config configs/base.yaml --mode plot
python -m mtd_dqn.main --config configs/base.yaml --mode tables
```

Artefacts:

* `results/logs/train_metrics.csv|json` – per-step metrics and JSON logs.
* `results/logs/mix_history.json` – attacker/defender mix snapshots.
* `results/checkpoints/agent_ep*.pt` – PyTorch checkpoints.
* `results/figures/` – Category A–D plots (`*.png` + `*.pdf`) plus CSV exports (Category A–D, convergence, etc.).
* `results/tables/` – Strategy/vulnerability tables in CSV + LaTeX.
* `comp-perf/dqn_evomtd/response_time/` – Response-time trend CSVs/PNGs, per-strategy cost CSVs, Fig.6/7/9 sweeps, plus `README_response_time.md`.

Response-time scripts:

```bash
python -m mtd_dqn.scripts.eval_response_time --outdir comp-perf/dqn_evomtd/response_time --episodes 5 --steps 200 --mu 200 400
python -m mtd_dqn.scripts.plot_response_time --outdir comp-perf/dqn_evomtd/response_time
python -m mtd_dqn.scripts.plot_per_strategy_costs
```

These produce:

* `training_response_trend.csv`
* `per_strategy_response_cost.csv`
* `fig6_avg_return_sweep.csv|.png`
* `fig7_request_time_vs_mu.csv|.png`
* `fig9_load_time_vs_mu.csv|.png`
* `per_strategy_response_time.png`
* `per_strategy_cost_breakdown.png`

---

## 4. Running the WF-MTD edge simulator

```bash
cd wf_mtd_edge
python -m main --config configs/strategy_space_demo.yaml --mode robust --steps 60 --plot
```

Optional additional figures:

```bash
python -m main --config configs/strategy_space_demo.yaml --mode robust --steps 60 \
  --extra-out ./figs --extra-episodes 120 --extra-robust-samples 64
```

Artefacts (depending on config):

* `outputs/results/` – Figures (`figs/`), tables (`tables/`), LaTeX reports.
* `figs/summary_metrics.json` – aggregated evolution metrics.
* `comp-perf/` – WF comparative exports (via `comp_perf.compute_comp_perf`).

---

## 5. Comparative analysis

Both projects emit `comp-perf/<model>/` directories containing:

* `summary_metrics.json|.csv` – aggregated utilities, ASR, costs.
* `timeseries.csv` – per-step StepLog exports.
* `figures/` – Utility/SAL/ASR trajectories, mixture plots, overhead bars, etc.
* `tables/` – Matching LaTeX tables.

Use these along with `compare/compare_models.py` (if present) to create cross-model comparisons:

```bash
python -m compare.compare_models \
  --dqn_dir MTD_DQN/mtd_dqn/comp-perf \
  --wf_dir wf_mtd_edge/comp-perf \
  --out_dir compare/comp-perf
```

This writes combined figures such as `cmp_ud_over_time.png`, `cmp_asr_over_time.png`, and `summary_compare.csv|.tex`.

---

## 6. Dependencies

`requirements.txt` lists the shared dependencies:

- `torch`, `numpy`, `pandas`, `matplotlib`
- `networkx`, `PyYAML`, `scipy`, `tqdm`
- `typer`

Install once at the repo root (venv recommended). Each script uses deterministic RNG (`np.random.default_rng(42)`) for reproducibility.

---

## 7. Deployment notes

* **MTD_DQN** – Run the main CLI or evaluation scripts; outputs land under `results/` and `comp-perf/dqn_evomtd/response_time/`.
* **wf_mtd_edge** – Use the Typer CLI in `main.py`; results mirror the original WF-MTD paper layout under `outputs/`.
* Both projects can be containerised or scheduled independently. All paths are relative, so keep repo root as the working directory when invoking scripts.

Refer to `README_response_time.md` and inline docstrings for additional details on CSV schemas and evaluation knobs.
