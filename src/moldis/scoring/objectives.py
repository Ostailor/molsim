from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

Goal = Literal["min", "max"]


@dataclass
class Objective:
    name: str
    goal: Goal  # "min" or "max"
    weight: float = 1.0


def apply_uncertainty_penalty(
    means: np.ndarray, sigmas: np.ndarray | None, goals: list[Goal], k: float = 1.0
) -> np.ndarray:
    """Apply uncertainty penalties: max → mean - k*sigma, min → mean + k*sigma.

    If sigmas is None, returns means unchanged.
    """
    if sigmas is None:
        return means.copy()
    penalized = means.copy()
    for j, goal in enumerate(goals):
        if goal == "max":
            penalized[:, j] = means[:, j] - k * sigmas[:, j]
        else:
            penalized[:, j] = means[:, j] + k * sigmas[:, j]
    return penalized


def compute_bounds(
    values: np.ndarray, low_q: float = 0.05, high_q: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Compute robust per-objective bounds from values using quantiles.

    Returns (mins, maxs), shape (d,), clipped to avoid zero width.
    """
    lo = np.quantile(values, low_q, axis=0)
    hi = np.quantile(values, high_q, axis=0)
    # ensure non-degenerate
    width = np.maximum(hi - lo, 1e-12)
    return lo, lo + width


def normalize(values: np.ndarray, goals: list[Goal], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Normalize each column to [0, 1] with goal-aware orientation.

    For max: (x - lo) / (hi - lo); for min: (hi - x) / (hi - lo).
    """
    width = np.maximum(hi - lo, 1e-12)
    X = (values - lo) / width
    out = np.empty_like(X)
    for j, goal in enumerate(goals):
        if goal == "max":
            out[:, j] = np.clip(X[:, j], 0.0, 1.0)
        else:
            out[:, j] = np.clip(1.0 - X[:, j], 0.0, 1.0)
    return out


def scalarize_weighted_sum(norm_values: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    w = np.asarray(list(weights), dtype=float)
    if w.ndim != 1 or w.shape[0] != norm_values.shape[1]:
        raise ValueError("weights shape must match number of objectives")
    w = w / (np.sum(w) + 1e-12)
    return norm_values @ w
