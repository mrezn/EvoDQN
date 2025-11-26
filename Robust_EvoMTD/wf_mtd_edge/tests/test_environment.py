import numpy as np
import pytest

from wf_mtd_edge.config import load_config
from wf_mtd_edge.environment import EdgeCloudEnvironment


def test_lateral_coefficients_match_manual_values():
    config = load_config("configs/small_edge_demo.yaml")
    env = EdgeCloudEnvironment(config, gamma_att=0.6)
    metrics = env.compute_lateral_metrics(m_max=3)

    assert metrics.xi[1][0] == pytest.approx(0.6, rel=1e-6)
    assert metrics.xi[2][0] == pytest.approx(0.36, rel=1e-6)
    assert metrics.xi[3][0] == pytest.approx(0.0, abs=1e-6)


def test_state_distribution_update_matches_manual_average():
    config = load_config("configs/small_edge_demo.yaml")
    env = EdgeCloudEnvironment(config)
    attacker_mix = np.array([0.5, 0.5])
    defender_mix = np.array([0.5, 0.5])
    env.update_state_distribution(attacker_mix, defender_mix, config.kernel.base)
    assert env.rho[0] == pytest.approx(0.55, abs=1e-6)
    assert env.rho[1] == pytest.approx(0.45, abs=1e-6)
