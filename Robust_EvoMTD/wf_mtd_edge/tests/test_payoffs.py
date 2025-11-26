import math

import numpy as np
import pytest

from wf_mtd_edge.config import load_config
from wf_mtd_edge.environment import EdgeCloudEnvironment
from wf_mtd_edge.payoffs import PayoffEngine


def test_sal_sap_components_against_manual_calc():
    config = load_config("configs/small_edge_demo.yaml")
    env = EdgeCloudEnvironment(config)
    metrics = env.compute_lateral_metrics()
    xi_scalar = float(np.mean(metrics.xi[1]))

    engine = PayoffEngine(config)
    ua, ud, components = engine.stage_utilities(0, 0, xi_scalar)

    defense = config.strategies.defense[0]
    attack = config.strategies.attack[0]
    c_star = config.mtd.c_star_for(defense)
    a_param = config.mtd.a_for(defense)
    pi_val = config.attributes.pi[attack]
    beta = (1.0 - math.exp(-c_star * a_param * pi_val)) / (1.0 + math.exp(-c_star * a_param * pi_val))
    theta = {
        key: 1.0 - config.attributes.lambda_for(defense)[key] * beta
        for key in config.attributes.weights
    }
    weights = config.attributes.weights_for(attack)
    sal_expected = (1.0 + xi_scalar) * config.attributes.resource_importance * sum(
        theta[key] * weights[key] * config.attributes.values[key]
        for key in theta
    )
    sap_expected = max(0.0, 1.0 - xi_scalar) * config.attributes.resource_importance * sum(
        (
            config.mtd.mu_for(defense) * theta[key]
            + (1.0 - theta[key])
        )
        * (1.0 - weights[key])
        * config.attributes.values[key]
        for key in theta
    )

    assert components.sal == pytest.approx(sal_expected, rel=1e-6)
    assert components.sap == pytest.approx(sap_expected, rel=1e-6)
    assert ua == pytest.approx(components.sal - components.ac - components.penalty, rel=1e-6)
    assert ud == pytest.approx(components.sap - components.dc + components.incentive, rel=1e-6)
