from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray, maximize: list[bool]) -> bool:
    """Return True if a dominates b given objective senses.

    For maximize=True objectives: higher is better; otherwise lower is better.
    """
    better_or_equal = True
    strictly_better = False
    for j, is_max in enumerate(maximize):
        if is_max:
            if a[j] < b[j]:
                better_or_equal = False
                break
            if a[j] > b[j]:
                strictly_better = True
        else:
            if a[j] > b[j]:
                better_or_equal = False
                break
            if a[j] < b[j]:
                strictly_better = True
    return better_or_equal and strictly_better


def fast_non_dominated_sort(values: np.ndarray, maximize: list[bool]) -> list[list[int]]:
    """Return fronts (list of index lists) using Deb's fast non-dominated sort.

    values: shape (n, d)
    maximize: list of length d
    """
    n = values.shape[0]
    S: list[list[int]] = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(values[p], values[q], maximize):
                S[p].append(q)
            elif dominates(values[q], values[p], maximize):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(front: list[int], values: np.ndarray) -> dict[int, float]:
    """Compute crowding distances for a single front (higher is better)."""
    if not front:
        return {}
    m = values.shape[1]
    dist = {i: 0.0 for i in front}
    for j in range(m):
        front_sorted = sorted(front, key=lambda idx: values[idx, j])
        vmin = values[front_sorted[0], j]
        vmax = values[front_sorted[-1], j]
        if vmax - vmin == 0:
            continue
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")
        for k in range(1, len(front_sorted) - 1):
            prev_idx = front_sorted[k - 1]
            next_idx = front_sorted[k + 1]
            dist[front_sorted[k]] += (values[next_idx, j] - values[prev_idx, j]) / (vmax - vmin)
    return dist


def nsga2_select(
    values: np.ndarray, k: int, maximize: list[bool], seed: int | None = 0
) -> list[int]:
    """Select k points using NSGA-II front + crowding distance rules."""
    fronts = fast_non_dominated_sort(values, maximize)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            cd = crowding_distance(front, values)
            # Sort by distance desc, break ties deterministically by index
            rest = sorted(front, key=lambda idx: (-cd[idx], idx))
            need = k - len(selected)
            selected.extend(rest[:need])
            break
    return selected


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float]) -> float:
    """Exact hypervolume in 2D for minimization objectives with reference point.

    Assumes smaller is better on both axes. For maximize objectives, transform before call.
    """
    if points.size == 0:
        return 0.0
    # Filter dominated points and sort by first coordinate
    P = points.copy()
    P = P[np.argsort(P[:, 0])]
    hv = 0.0
    best_y = ref[1]
    for x, y in P:
        width = max(0.0, ref[0] - x)
        height = max(0.0, best_y - y)
        hv += width * height
        best_y = min(best_y, y)
    return hv
