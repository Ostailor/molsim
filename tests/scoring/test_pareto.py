from __future__ import annotations

import numpy as np

from moldis.scoring.pareto import fast_non_dominated_sort, hypervolume_2d, nsga2_select


def test_dominance_and_fronts_simple():
    # Two objectives to maximize both (use transformed scores)
    vals = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.8],
            [0.5, 0.9],
            [0.9, 0.4],
            [0.6, 0.6],
        ]
    )
    maximize = [True, True]
    fronts = fast_non_dominated_sort(vals, maximize)
    # First front should contain non-dominated points
    f0 = set(fronts[0])
    assert f0 == {0, 1, 2, 3}
    # NSGA-II selecting 3 should come from first front only
    sel = nsga2_select(vals, 3, maximize=maximize, seed=0)
    assert set(sel).issubset(f0)


def test_hypervolume_2d_minimization():
    # Minimize both objectives; ref point (2,2)
    pts = np.array([[1.0, 1.0], [0.5, 1.5], [1.2, 0.6]])
    hv = hypervolume_2d(pts, ref=(2.0, 2.0))
    assert hv > 0.0
