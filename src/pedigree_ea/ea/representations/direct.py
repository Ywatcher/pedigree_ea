"""Direct representation: the genotype is the (N, 2) parent array itself.

Operators edit the pedigree and keep it valid (acyclic, <= 2 parents); edits
that would create a cycle are rejected. They follow the mutation list in
ideas.md, grouped by size:
    small   add_edge, remove_edge, rewire_parent
    medium  add_latent_parent, make_siblings, insert_latent, bypass_latent
    major   merge_latent, split_latent, move_branch
Crossover (optional, off by default in configs) takes each person's parent
row from either genotype; latent slots are not aligned between genotypes, so
it is mostly meaningful for observed people.
"""

from __future__ import annotations

import numpy as np

from ... import batch
from . import dag
from .base import Representation

SIZES = {
    "add_edge": "small", "remove_edge": "small", "rewire_parent": "small",
    "add_latent_parent": "medium", "make_siblings": "medium",
    "insert_latent": "medium", "bypass_latent": "medium",
    "merge_latent": "major", "split_latent": "major", "move_branch": "major",
}


class Direct(Representation):
    name = "direct"
    has_crossover = True

    def __init__(self, n_obs: int, max_latent: int, init_mutations: int | None = None):
        super().__init__(n_obs, max_latent)
        self.init_mutations = init_mutations if init_mutations is not None else n_obs

    def random(self, rng):
        g = np.full((self.n, 2), dag.MISSING, dtype=np.int16)
        for _ in range(int(rng.integers(0, self.init_mutations + 1))):
            g, _ = self.mutate(g, rng)
        return g

    def decode(self, g):
        return g

    def operators(self):
        return {name: getattr(self, name) for name in SIZES}

    def crossover(self, a, b, rng):
        take_b = rng.random(self.n) < 0.5
        child = dag.finalize(np.where(take_b[:, None], b, a), self.n_obs)
        return a if child is None else child

    # ---- helpers --------------------------------------------------------------
    def _done(self, g):
        return dag.finalize(g, self.n_obs)

    def _parent_candidates(self, g, c, anc):
        """People who could become a parent of c without a cycle or duplicate."""
        ok = np.ones(self.n, dtype=bool)
        ok[c] = False
        ok &= ~anc[c]                         # c's descendants would close a cycle
        ok[g[c][g[c] >= 0]] = False
        # Only offer in-use latent slots plus one free one, to avoid wasting slots.
        free = dag.unused_latent(g, self.n_obs)
        ok[free[1:]] = False
        return np.nonzero(ok)[0]

    def _with_free_slot(self, g):
        return np.nonzero((g < 0).any(axis=1))[0]

    def _used_latent(self, g):
        has_child = np.zeros(self.n, dtype=bool)
        has_child[g[g >= 0]] = True
        return np.nonzero(has_child[self.n_obs:])[0] + self.n_obs

    # ---- small ------------------------------------------------------------------
    def add_edge(self, g, rng):
        cands = self._with_free_slot(g)
        if not cands.size:
            return None
        c = rng.choice(cands)
        ps = self._parent_candidates(g, c, dag.ancestors(g))
        if not ps.size:
            return None
        g = g.copy()
        g[c, 1] = rng.choice(ps)
        return self._done(g)

    def remove_edge(self, g, rng):
        edges = np.argwhere(g >= 0)
        if not edges.size:
            return None
        c, k = edges[rng.integers(len(edges))]
        g = g.copy()
        g[c, k] = dag.MISSING
        return self._done(g)

    def rewire_parent(self, g, rng):
        """Replace one parent: x1 -> A becomes x3 -> A."""
        edges = np.argwhere(g >= 0)
        if not edges.size:
            return None
        c, k = edges[rng.integers(len(edges))]
        h = g.copy()
        h[c, k] = dag.MISSING
        ps = self._parent_candidates(h, c, dag.ancestors(h))
        ps = ps[ps != g[c, k]]
        if not ps.size:
            return None
        h[c, k] = rng.choice(ps)
        return self._done(h)

    # ---- medium -----------------------------------------------------------------
    def add_latent_parent(self, g, rng):
        """A new latent person becomes a parent of two people (making them half-sibs)."""
        free = dag.unused_latent(g, self.n_obs)
        cands = self._with_free_slot(g)
        if not free.size or cands.size < 2:
            return None
        lat = free[0]
        c1, c2 = rng.choice(cands, size=2, replace=False)
        g = g.copy()
        g[c1, 1] = lat
        g[c2, 1] = lat
        return self._done(g)

    def make_siblings(self, g, rng):
        """B takes A's parents (A and B become full or half siblings)."""
        with_parents = np.nonzero((g >= 0).any(axis=1))[0]
        if not with_parents.size:
            return None
        a = rng.choice(with_parents)
        others = np.setdiff1d(np.arange(self.n_obs), [a])
        if not others.size:
            return None
        b = rng.choice(others)
        h = g.copy()
        h[b] = g[a]
        return self._done(h)

    def insert_latent(self, g, rng):
        """p -> c becomes p -> L -> c (one more generation between them)."""
        edges = np.argwhere(g >= 0)
        free = dag.unused_latent(g, self.n_obs)
        if not edges.size or not free.size:
            return None
        c, k = edges[rng.integers(len(edges))]
        lat = free[0]
        h = g.copy()
        h[lat, 0] = g[c, k]
        h[c, k] = lat
        return self._done(h)

    def bypass_latent(self, g, rng):
        """Remove a latent person with one parent p, attaching their children to p."""
        used = self._used_latent(g)
        single = [x for x in used if (g[x] >= 0).sum() == 1]
        if not single:
            return None
        lat = single[rng.integers(len(single))]
        p = g[lat][g[lat] >= 0][0]
        h = g.copy()
        h[h == lat] = p
        h[lat] = dag.MISSING
        return self._done(h)

    # ---- major ------------------------------------------------------------------
    def merge_latent(self, g, rng):
        used = self._used_latent(g)
        if used.size < 2:
            return None
        keep, drop = rng.choice(used, size=2, replace=False)
        h = g.copy()
        h[h == drop] = keep
        merged = [x for x in list(g[keep]) + list(g[drop]) if x >= 0 and x != keep]
        merged = list(dict.fromkeys(merged))
        if len(merged) > 2:
            merged = list(rng.choice(merged, size=2, replace=False))
        h[keep] = (merged + [dag.MISSING] * 2)[:2]
        h[drop] = dag.MISSING
        return self._done(h)

    def split_latent(self, g, rng):
        """Move some children of a latent person to a new latent person with the same parents."""
        free = dag.unused_latent(g, self.n_obs)
        multi = [x for x in self._used_latent(g) if dag.children_of(g, x).size >= 2]
        if not free.size or not multi:
            return None
        lat = multi[rng.integers(len(multi))]
        kids = dag.children_of(g, lat)
        moved = rng.choice(kids, size=int(rng.integers(1, kids.size)), replace=False)
        new = free[0]
        h = g.copy()
        for c in moved:
            h[c][h[c] == lat] = new
        h[new] = g[lat]
        return self._done(h)

    def move_branch(self, g, rng):
        """Detach a person from their parents and reattach to 0-2 new ones."""
        c = int(rng.integers(self.n))
        h = g.copy()
        h[c] = dag.MISSING
        anc = dag.ancestors(h)
        for k in range(int(rng.integers(0, 3))):
            ps = self._parent_candidates(h, c, anc)
            if not ps.size:
                break
            h[c, k] = rng.choice(ps)
        if np.array_equal(batch.normalize(h), g):
            return None
        return self._done(h)
