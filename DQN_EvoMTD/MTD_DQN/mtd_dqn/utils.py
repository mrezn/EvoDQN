
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - torch is optional in tests
    import torch
except ImportError:  # pragma: no cover - fallback when torch is absent
    torch = None


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        else:
            torch.use_deterministic_algorithms(False)


def ensure_dir(path: os.PathLike[str] | str) -> Path:

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def project_to_simplex(vector: Sequence[float], epsilon: float = 1e-12) -> np.ndarray:

    x = np.asarray(vector, dtype=np.float64)
    if np.allclose(x.sum(), 1.0, atol=1e-9) and np.all(x >= 0.0):
        return x.copy()

    x_sorted = np.sort(x)[::-1]
    cssv = np.cumsum(x_sorted)
    rho = np.nonzero(x_sorted + (1.0 - cssv) / (np.arange(1, len(x_sorted) + 1)) > 0)[0]
    if len(rho) == 0:
        tau = 0.0
    else:
        rho_idx = rho[-1]
        tau = (cssv[rho_idx] - 1.0) / (rho_idx + 1)
    projected = np.maximum(x - tau, 0.0)
    projected_sum = projected.sum()
    if projected_sum <= epsilon:
        projected = np.full_like(projected, 1.0 / projected.size)
    else:
        projected /= projected_sum
    return projected


def softmax_temperature(logits: Sequence[float], tau: float, axis: int = -1) -> np.ndarray:

    logits_arr = np.asarray(logits, dtype=np.float64)
    if tau <= 0:
        raise ValueError("Temperature tau must be positive.")
    shifted = logits_arr - np.max(logits_arr, axis=axis, keepdims=True)
    exps = np.exp(shifted / tau)
    partition = np.sum(exps, axis=axis, keepdims=True)
    return exps / partition


def running_mean(values: Iterable[float]) -> float:

    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    if count == 0:
        raise ValueError("Cannot compute the mean of an empty iterable.")
    return total / float(count)
