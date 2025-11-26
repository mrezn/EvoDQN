import numpy as np
import pytest

from wf_mtd_edge.config import load_config
from wf_mtd_edge.evolution import EvolutionEngine


def test_evolution_runs_and_preserves_simplex():
    config = load_config("configs/small_edge_demo.yaml")
    config.run.steps = 10
    config.run.mode = "bayes"
    engine = EvolutionEngine(config)
    engine.run()
    summary = engine.summary()

    attacker_probs = np.array(list(summary["attacker_mix"].values()))
    defender_probs = np.array(list(summary["defender_mix"].values()))

    assert attacker_probs.sum() == pytest.approx(1.0, rel=1e-6)
    assert defender_probs.sum() == pytest.approx(1.0, rel=1e-6)
    assert (attacker_probs >= 0).all()
    assert (defender_probs >= 0).all()
    assert engine.logs, "Expected evolution logs to be populated"
