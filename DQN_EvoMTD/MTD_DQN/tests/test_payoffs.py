import numpy as np

from mtd_dqn import payoffs


def test_beta_success_in_unit_interval() -> None:
    value = payoffs.beta_success(0.8, 1.2, 0.7)
    assert 0.0 < value < 1.0


def test_sal_increases_with_xi() -> None:
    theta = np.array([0.3, 0.4, 0.5])
    weights = np.array([0.4, 0.3, 0.3])
    valuations = np.array([2.0, 3.0, 4.0])
    sal_low = payoffs.compute_sal(0.1, 1.5, theta, weights, valuations)
    sal_high = payoffs.compute_sal(0.5, 1.5, theta, weights, valuations)
    assert sal_high > sal_low


def test_sap_remains_positive() -> None:
    theta = np.array([0.2, 0.3, 0.4])
    mu = np.array([0.5, 0.6, 0.7])
    weights = np.array([0.3, 0.3, 0.4])
    valuations = np.array([1.5, 2.0, 2.5])
    sap = payoffs.compute_sap(0.2, 1.5, theta, mu, weights, valuations)
    assert sap > 0.0
