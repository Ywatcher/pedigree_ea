"""Pareto ranking and NSGA-II selection (all objectives minimized)."""

from __future__ import annotations

import numpy as np


def dominance(objs: np.ndarray) -> np.ndarray:
    """dom[a, b] = True if a dominates b."""
    le = (objs[:, None, :] <= objs[None, :, :]).all(axis=2)
    lt = (objs[:, None, :] < objs[None, :, :]).any(axis=2)
    return le & lt


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool((a <= b).all() and (a < b).any())


def nondominated_ranks(objs: np.ndarray) -> np.ndarray:
    """Front index per row: 0 = non-dominated, 1 = next front, ..."""
    dom = dominance(objs)
    n_dominators = dom.sum(axis=0)
    ranks = np.full(len(objs), -1)
    front = np.nonzero(n_dominators == 0)[0]
    r = 0
    while front.size:
        ranks[front] = r
        n_dominators = n_dominators - dom[front].sum(axis=0)
        n_dominators[ranks >= 0] = -1
        front = np.nonzero(n_dominators == 0)[0]
        r += 1
    return ranks


def crowding(objs: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Crowding distance within each front (inf at the extremes)."""
    dist = np.zeros(len(objs))
    for r in np.unique(ranks):
        idx = np.nonzero(ranks == r)[0]
        if idx.size <= 2:
            dist[idx] = np.inf
            continue
        for m in range(objs.shape[1]):
            order = idx[np.argsort(objs[idx, m], kind="stable")]
            span = objs[order[-1], m] - objs[order[0], m]
            dist[order[0]] = dist[order[-1]] = np.inf
            if span > 0:
                dist[order[1:-1]] += (objs[order[2:], m] - objs[order[:-2], m]) / span
    return dist


def nsga2_select(objs: np.ndarray, k: int, penalize: np.ndarray | None = None) -> np.ndarray:
    """Indices of k survivors by (rank, -crowding). Rows flagged in `penalize`
    (e.g. duplicate phenotypes) are only taken after all others."""
    ranks = nondominated_ranks(objs)
    if penalize is not None:
        ranks = np.where(penalize, ranks + len(objs), ranks)
    dist = crowding(objs, ranks)
    order = np.lexsort((-dist, ranks))
    return order[:k]


def tournament(ranks: np.ndarray, dist: np.ndarray, rng: np.random.Generator) -> int:
    a, b = rng.integers(len(ranks), size=2)
    if ranks[a] != ranks[b]:
        return int(a if ranks[a] < ranks[b] else b)
    return int(a if dist[a] >= dist[b] else b)
