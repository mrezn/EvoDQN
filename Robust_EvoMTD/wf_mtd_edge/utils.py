"""Utility helpers for the WF-MTD edge-cloud model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function."""

    return 1.0 / (1.0 + np.exp(-x))


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp *value* to the closed interval [lower, upper]."""

    return max(lower, min(upper, value))


def project_to_simplex(probabilities: np.ndarray) -> np.ndarray:
    """Project a real vector onto the probability simplex.

    Implements the algorithm of Wang & Carreira-Perpi??n (2013), guaranteeing
    non-negativity and a unit sum. Used for the Wright?Fisher updates.
    """

    if probabilities.ndim != 1:
        raise ValueError("Simplex projection expects a one-dimensional array.")
    if len(probabilities) == 0:
        raise ValueError("Cannot project an empty vector.")
    sorted_probs = np.sort(probabilities)[::-1]
    cumulative = np.cumsum(sorted_probs)
    rho = np.nonzero(sorted_probs + (1.0 - cumulative) / (np.arange(len(sorted_probs)) + 1) > 0)[0]
    if len(rho) == 0:
        theta = 0.0
    else:
        rho_idx = rho[-1]
        theta = (cumulative[rho_idx] - 1.0) / (rho_idx + 1)
    projected = np.maximum(probabilities - theta, 0.0)
    projected /= projected.sum()
    return projected


def ensure_rng(seed: int | None = None) -> np.random.Generator:
    """Create a numpy Generator seeded for reproducibility."""

    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def write_json_lines(path: str | Path, rows: Iterable[dict]) -> None:
    """Write an iterable of dictionaries to *path* in JSON Lines format."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the L1 distance between two equal-dimensional vectors."""

    if p.shape != q.shape:
        raise ValueError("Arrays must share the same shape to compute L1 distance.")
    return float(np.sum(np.abs(p - q)))
