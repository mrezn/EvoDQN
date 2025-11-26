from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TrainingResponseTrendRow:
    episode: int
    step: int
    mu: int
    attack_mode: str
    n_attackers: int
    n_targets: int
    def_strategy: str
    resp_time_ms: float
    load_time_ms: float
    reconfig_latency_ms: float
    ASSC_ms: float
    AIC_ms: float
    NC_ms: float
    cost_total_ms: float
    reward: float
    ASR: float
    SAL: float
    UD: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class PerStrategyResponseCostRow:
    def_strategy: str
    mu: int
    attack_mode: str
    n_attackers: int
    n_targets: int
    resp_time_ms_mean: float
    resp_time_ms_p95: float
    load_time_ms_mean: float
    load_time_ms_p95: float
    ASSC_ms_mean: float
    AIC_ms_mean: float
    NC_ms_mean: float
    cost_total_ms_mean: float

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


FIG_METHODS = [
    "DQN-EvoMTD",
    "Robust-EvoMTD",
    "No-MTD",
    "OpenMTD",
    "FastMove",
    "CMDP-MOS",
    "ID-HAM",
]


FIG_PANELS = ["a", "b", "c", "d"]


UNC_DIR_NAME = "comp-perf/dqn_evomtd/response_time"


def schema_columns_training() -> List[str]:
    return list(TrainingResponseTrendRow.__annotations__)


def schema_columns_per_strategy() -> List[str]:
    return list(PerStrategyResponseCostRow.__annotations__)
