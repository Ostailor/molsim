from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .objectives import (
    Goal,
    apply_uncertainty_penalty,
    compute_bounds,
    normalize,
    scalarize_weighted_sum,
)
from .pareto import nsga2_select


@dataclass
class CandidateScores:
    ids: list[str]
    values: np.ndarray  # shape (n, d) raw means
    sigmas: np.ndarray | None  # shape (n, d) or None
    goals: list[Goal]


def select_pareto(
    ids: list[str],
    means: np.ndarray,
    sigmas: np.ndarray | None,
    goals: list[Goal],
    k: int,
    penalty_k: float = 1.0,
    seed: int | None = 0,
) -> list[str]:
    """Select k candidate IDs via NSGA-II on uncertainty-penalized values.

    Steps:
    - apply uncertainty penalties to means
    - robust-normalize each objective to [0,1] oriented to maximize
    - NSGA-II selection using maximize=True for all (since normalized orientation ensures that)
    """
    if means.ndim != 2:
        raise ValueError("means must be 2D (n, d)")
    penalized = apply_uncertainty_penalty(means, sigmas, goals, k=penalty_k)
    lo, hi = compute_bounds(penalized)
    norm = normalize(penalized, goals, lo, hi)
    maximize = [True] * norm.shape[1]
    idx = nsga2_select(norm, min(k, norm.shape[0]), maximize=maximize, seed=seed)
    return [ids[i] for i in idx]


def select_weighted_sum(
    ids: list[str],
    means: np.ndarray,
    sigmas: np.ndarray | None,
    goals: list[Goal],
    weights: Iterable[float],
    penalty_k: float = 1.0,
) -> list[str]:
    """Select by scalarization (weighted sum) on uncertainty-penalized normalized values."""
    penalized = apply_uncertainty_penalty(means, sigmas, goals, k=penalty_k)
    lo, hi = compute_bounds(penalized)
    norm = normalize(penalized, goals, lo, hi)
    scores = scalarize_weighted_sum(norm, list(weights))
    order = np.argsort(-scores)  # descending
    return [ids[i] for i in order]
