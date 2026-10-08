"""Expected kinship coefficients.

Kinship phi(i, j) is the probability that alleles drawn at random from i and
j at the same locus are identical by descent. Reference values (no inbreeding):
self 0.5, parent-child and full siblings 0.25, second degree 0.125, first
cousins 0.0625. A degree-d relationship has phi = 2 ** -(d + 1).

Recursion, with people in topological order (parents first):
    phi(i, i) = (1 + phi(f, m)) / 2        (1/2 if a parent is missing)
    phi(i, j) = (phi(f, j) + phi(m, j)) / 2  for j earlier than i
A missing parent contributes 0.

Kinship never decreases when a parent-child edge is added, which lets a search
prune as soon as any pair exceeds its target (see reference/brute_force.py).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import numpy as np

from .pairs import Pair, PairData, pair_key  # noqa: F401  (re-exported)
from .pedigree import Pedigree, PedigreeError


class KinshipData(PairData[float]):
    """Kinship values for pairs of people."""

    _convert = staticmethod(float)

    def matrix(self) -> tuple[list[str], np.ndarray]:
        """Square matrix over `ids`, NaN for unknown pairs and the diagonal."""
        idx = {pid: i for i, pid in enumerate(self.ids)}
        m = np.full((len(self.ids), len(self.ids)), np.nan)
        for (a, b), v in self._values.items():
            m[idx[a], idx[b]] = m[idx[b], idx[a]] = v
        return self.ids, m


def _topo_indices(parents: Sequence[Sequence[int]]) -> list[int]:
    n = len(parents)
    indeg = [len(ps) for ps in parents]
    children: list[list[int]] = [[] for _ in range(n)]
    for i, ps in enumerate(parents):
        for p in ps:
            children[p].append(i)
    queue = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for c in children[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if len(order) != n:
        raise PedigreeError("pedigree contains a cycle")
    return order


def kinship_from_parents(parents: Sequence[Sequence[int]]) -> np.ndarray:
    """Kinship matrix for people 0..n-1, where parents[i] lists i's parent indices."""
    n = len(parents)
    phi = np.zeros((n, n))
    done: list[int] = []
    for i in _topo_indices(parents):
        ps = list(parents[i])
        phi[i, i] = 0.5 * (1.0 + phi[ps[0], ps[1]]) if len(ps) == 2 else 0.5
        if done and ps:
            row = 0.5 * phi[np.ix_(done, ps)].sum(axis=1)
            phi[i, done] = row
            phi[done, i] = row
        done.append(i)
    return phi


def kinship_matrix(ped: Pedigree) -> tuple[list[str], np.ndarray]:
    """Kinship matrix over all people in `ped`, in `ped.ids` order."""
    ids = ped.ids
    idx = {pid: i for i, pid in enumerate(ids)}
    try:
        parents = [[idx[p] for p in ped[pid].parents] for pid in ids]
    except KeyError as e:
        raise PedigreeError(f"unknown parent {e.args[0]!r}") from None
    return ids, kinship_from_parents(parents)


def expected_kinship(ped: Pedigree, ids: Sequence[str] | None = None) -> KinshipData:
    """Expected kinship for every pair among `ids` (default: observed people)."""
    ids = list(ped.observed_ids if ids is None else ids)
    all_ids, phi = kinship_matrix(ped)
    idx = {pid: i for i, pid in enumerate(all_ids)}
    values = {(a, b): phi[idx[a], idx[b]]
              for k, a in enumerate(ids) for b in ids[k + 1:]}
    return KinshipData(values, ids)


def inbreeding(ped: Pedigree) -> dict[str, float]:
    """Inbreeding coefficient F = 2 * phi(i, i) - 1 per person."""
    ids, phi = kinship_matrix(ped)
    return {pid: 2 * phi[i, i] - 1 for i, pid in enumerate(ids)}


def pair_errors(expected: KinshipData, target: KinshipData) -> dict[Pair, float]:
    """Signed error (expected - target) for every pair in `target`.

    Raises KeyError if `expected` lacks a pair the target has.
    """
    return {pair: expected[pair] - v for pair, v in target.items()}


# ---- relationship degree ------------------------------------------------

def kinship_to_degree(phi: float) -> float:
    """Continuous degree: 0 for self/MZ twins, 1 for first degree, ...; inf if phi <= 0."""
    return -math.log2(phi) - 1 if phi > 0 else math.inf


def degree_class(phi: float, max_degree: int = 3) -> int | None:
    """Nearest relationship degree, using KING-style cut-offs halfway (in log2)
    between degrees: > 0.354 -> 0, > 0.177 -> 1, > 0.0884 -> 2, > 0.0442 -> 3.
    Returns None for unrelated/more distant than `max_degree`.
    """
    for d in range(max_degree + 1):
        if phi > 2.0 ** -(d + 1.5):
            return d
    return None
