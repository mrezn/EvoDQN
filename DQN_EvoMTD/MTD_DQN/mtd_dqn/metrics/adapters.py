from __future__ import annotations

from typing import Dict, List

import numpy as np

from .. import payoffs

_RNG = np.random.default_rng(42)


def _base_service_time(mu: int) -> float:
    service_base = 120.0  
    return service_base + 0.02 * mu


def _queue_delay(mu: int, capacity: float = 950.0) -> float:
    rho = min(0.95, mu / max(capacity, 1.0))
    if rho <= 0.0:
        return 0.0
    return 800.0 * rho / max(1e-3, 1.0 - rho)


def measure_request_response(
    env,
    mu: int,
    attack_mode: str,
    n_attackers: int,
    n_targets: int,
) -> Dict[str, float]:
    if hasattr(env, "simulate_request_batch"):
        return env.simulate_request_batch(mu, attack_mode, n_attackers, n_targets)

    assc = getattr(env, "ASSC", 0.3)
    aic = getattr(env, "AIC", 0.25)
    nc = payoffs.network_cost(getattr(env, "SQ", 0.6), getattr(env, "a_r", 1.2), getattr(env, "k_s", 0.4))
    reconfig_latency_ms = (env.c_star * env.a_r * 250.0) / max(1, n_targets)
    reconfig_latency_ms += _RNG.normal(0, 5)

    resp_time = _base_service_time(mu) + _queue_delay(mu) + reconfig_latency_ms
    resp_time += 15.0 * (n_attackers / max(1, n_targets))

    load_time = resp_time + 0.1 * mu + _RNG.normal(0, 10)
    cost_total = (assc + aic + nc) * 1000.0 + reconfig_latency_ms

    return {
        "resp_time_ms": max(5.0, resp_time),
        "load_time_ms": max(5.0, load_time),
        "reconfig_latency_ms": max(1.0, reconfig_latency_ms),
        "ASSC_ms": assc * 800.0,
        "AIC_ms": aic * 700.0,
        "NC_ms": nc * 900.0,
        "cost_total_ms": max(5.0, cost_total),
    }


def compute_asr_step(env, window: int = 50) -> float:
    outcomes: List[int] = getattr(env, "_recent_attack_outcomes", [])
    if outcomes:
        data = outcomes[-window:]
        if data:
            return float(np.mean(data))
    sal = getattr(env, "last_sal", None)
    if sal is None:
        return 0.5
    return float(np.clip(sal / (abs(sal) + 10.0), 0.0, 1.0))
