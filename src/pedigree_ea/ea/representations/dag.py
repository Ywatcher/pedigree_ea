"""Helpers for one (N, 2) parent array, shared by representations.

`Builder` adds edges one at a time and refuses those that would break the
pedigree rules (self-parent, third parent, duplicate, cycle), keeping an
ancestor matrix up to date. Decoders use it to turn arbitrary edge lists
into valid pedigrees.
"""

from __future__ import annotations

import numpy as np

from ... import batch

MISSING = batch.MISSING


def ancestors(parents: np.ndarray) -> np.ndarray:
    """anc[i, j] = i is a strict ancestor of j."""
    return batch.ancestor_matrix(parents[None])[0]


def children_of(parents: np.ndarray, p: int) -> np.ndarray:
    return np.nonzero((parents == p).any(axis=1))[0]


def n_free(parents: np.ndarray, c: int) -> int:
    return int((parents[c] < 0).sum())


def is_acyclic(parents: np.ndarray) -> bool:
    return not np.diagonal(ancestors(parents)).any()


def release_unused_latent(parents: np.ndarray, n_obs: int) -> np.ndarray:
    """Latent slots with no children are unused: clear their parents so they are free."""
    p = parents.copy()
    for _ in range(p.shape[0]):
        has_child = np.zeros(p.shape[0], dtype=bool)
        has_child[p[p >= 0]] = True
        unused = ~has_child & (p >= 0).any(axis=1)
        unused[:n_obs] = False
        if not unused.any():
            break
        p[unused] = MISSING
    return p


def unused_latent(parents: np.ndarray, n_obs: int) -> np.ndarray:
    """Latent slots with no parents and no children."""
    has_child = np.zeros(parents.shape[0], dtype=bool)
    has_child[parents[parents >= 0]] = True
    free = ~has_child & (parents < 0).all(axis=1)
    free[:n_obs] = False
    return np.nonzero(free)[0]


def finalize(parents: np.ndarray, n_obs: int) -> np.ndarray | None:
    """Normalize, release unused latent slots; None if the result has a cycle."""
    p = release_unused_latent(batch.normalize(parents), n_obs)
    return p if is_acyclic(p) else None


class Builder:
    def __init__(self, n: int):
        self.parents = np.full((n, 2), MISSING, dtype=np.int16)
        self.anc = np.zeros((n, n), dtype=bool)
        self.rejected_cycle = 0
        self.rejected_full = 0

    def can_add(self, p: int, c: int) -> str | None:
        """Reason the edge p -> c cannot be added, or None if it can."""
        if p == c or p < 0 or c < 0:
            return "invalid"
        if p in self.parents[c]:
            return "duplicate"
        if self.anc[c, p]:
            return "cycle"
        if (self.parents[c] >= 0).all():
            return "full"
        return None

    def add(self, p: int, c: int) -> bool:
        why = self.can_add(p, c)
        if why == "cycle":
            self.rejected_cycle += 1
        elif why == "full":
            self.rejected_full += 1
        if why is not None:
            return False
        k = 0 if self.parents[c, 0] < 0 else 1
        self.parents[c, k] = p
        up = self.anc[:, p].copy()
        up[p] = True
        down = self.anc[c].copy()
        down[c] = True
        self.anc |= up[:, None] & down[None, :]
        return True

    def result(self) -> np.ndarray:
        return batch.normalize(self.parents)
