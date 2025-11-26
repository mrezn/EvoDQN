import numpy as np

from wf_mtd_edge.config import load_config
from wf_mtd_edge.uncertainty import UncertaintyModel


def test_transition_candidates_tv_within_radius():
    config = load_config("configs/small_edge_demo.yaml")
    uncertainty = UncertaintyModel(config)
    base_kernel = config.kernel.base
    candidates = uncertainty.transition_candidates(0, 0, 0, "robust", base_kernel)
    assert candidates, "Expected at least one transition candidate"
    base_row = base_kernel[0, 0, 0]
    epsilon = config.uncertainty.tv_radius
    for candidate in candidates:
        assert candidate.shape == base_row.shape
        distance = float(np.sum(np.abs(candidate - base_row)))
        assert distance <= epsilon + 1e-6


def test_theta_candidates_bayes_sampling():
    config = load_config("configs/small_edge_demo.yaml")
    uncertainty = UncertaintyModel(config)
    theta_draws = uncertainty.theta_candidates(0, 0, 0, "bayes", True)
    assert theta_draws.shape[0] == config.uncertainty.samples
    bounds = config.uncertainty.theta_a_bounds[0, 0, 0]
    assert float(theta_draws.min()) >= bounds[0] - 1e-6
    assert float(theta_draws.max()) <= bounds[1] + 1e-6
