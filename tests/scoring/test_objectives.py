from __future__ import annotations

import numpy as np

from moldis.scoring.objectives import (
    apply_uncertainty_penalty,
    compute_bounds,
    normalize,
    scalarize_weighted_sum,
)
from moldis.scoring.selection import select_pareto, select_weighted_sum


def test_normalize_and_scalarize():
    means = np.array([[1.0, 10.0], [2.0, 8.0], [3.0, 6.0]])
    goals = ["max", "min"]
    lo, hi = compute_bounds(means)
    norm = normalize(means, goals, lo, hi)
    # First objective increases with means; second increases as raw decreases
    assert norm[0, 0] < norm[2, 0]
    assert norm[0, 1] < norm[2, 1]
    scores = scalarize_weighted_sum(norm, [0.5, 0.5])
    order = list(np.argsort(-scores))
    assert len(order) == 3


def test_uncertainty_penalty_and_selection():
    ids = ["a", "b", "c"]
    means = np.array([[1.0, 1.0], [0.9, 1.1], [1.1, 0.8]])
    sigmas = np.array([[0.1, 0.2], [0.05, 0.05], [0.2, 0.1]])
    goals = ["max", "max"]
    penal = apply_uncertainty_penalty(means, sigmas, goals, k=1.0)
    assert penal.shape == means.shape
    # Pareto selection returns k ids
    sel = select_pareto(ids, means, sigmas, goals, k=2, penalty_k=1.0, seed=0)
    assert len(sel) == 2
    # Weighted sum returns all sorted
    sel2 = select_weighted_sum(ids, means, sigmas, goals, weights=[0.5, 0.5])
    assert len(sel2) == 3
