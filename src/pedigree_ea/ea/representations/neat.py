"""NEAT-like representation: a list of parent->child edge genes with innovation numbers.

Genes are (innovation, parent slot, child slot, enabled). The same structural
edge (parent slot, child slot) always gets the same innovation number within a
run, so crossover can align genomes by innovation. Genomes start empty and
grow (complexification), as in NEAT.

Decoding adds enabled genes in innovation order and silently skips genes that
would break the pedigree rules (cycle, third parent); skipped genes are
neutral, inherited and may become active later.

Operators: add_connection, toggle_enable, remove_gene (small);
add_node = split an edge p -> c into p -> L -> c, add_latent_parent (medium).

OPEN (needs discussion before relying on crossover results):
- Latent identity across genomes. Latent slots are only aligned when created
  by splitting the same innovation (`_split` registry); otherwise slot L5 in
  two genomes can mean unrelated people, so crossover may combine them
  arbitrarily.
- Speciation. `distance` is implemented (NEAT compatibility distance), but no
  speciated strategy uses it yet; the baseline runs NSGA-II with crossover.
- How "fitter parent" is defined for disjoint/excess genes under multiple
  objectives. Here the first parent passed (the tournament winner) is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dag
from .base import Representation


@dataclass(frozen=True)
class Genome:
    innov: np.ndarray     # (G,) sorted
    src: np.ndarray       # (G,) parent slot
    dst: np.ndarray       # (G,) child slot
    enabled: np.ndarray   # (G,) bool

    def __len__(self) -> int:
        return len(self.innov)


def _genome(rows) -> Genome:
    rows = sorted(rows)
    if not rows:
        e = np.zeros(0, dtype=np.int32)
        return Genome(e, e.astype(np.int16), e.astype(np.int16), np.zeros(0, dtype=bool))
    innov, src, dst, en = zip(*rows)
    return Genome(np.array(innov, dtype=np.int32), np.array(src, dtype=np.int16),
                  np.array(dst, dtype=np.int16), np.array(en, dtype=bool))


def _rows(g: Genome) -> list[tuple]:
    return list(zip(g.innov.tolist(), g.src.tolist(), g.dst.tolist(), g.enabled.tolist()))


class Neat(Representation):
    name = "neat"
    has_crossover = True

    def __init__(self, n_obs: int, max_latent: int, init_mutations: int | None = None,
                 c_disjoint: float = 1.0, c_enabled: float = 0.5):
        super().__init__(n_obs, max_latent)
        self.init_mutations = init_mutations if init_mutations is not None else n_obs
        self.c_disjoint = c_disjoint
        self.c_enabled = c_enabled
        self._innov: dict[tuple[int, int], int] = {}
        self._split: dict[int, int] = {}

    def innovation(self, p: int, c: int) -> int:
        return self._innov.setdefault((int(p), int(c)), len(self._innov))

    # ---- required ---------------------------------------------------------------
    def random(self, rng):
        g = _genome([])
        for _ in range(int(rng.integers(0, self.init_mutations + 1))):
            g, _ = self.mutate(g, rng)
        return g

    def decode(self, g):
        b = dag.Builder(self.n)
        for p, c, en in zip(g.src.tolist(), g.dst.tolist(), g.enabled.tolist()):
            if en:
                b.add(p, c)
        return b.result()

    def operators(self):
        return {"add_connection": self.add_connection, "toggle_enable": self.toggle_enable,
                "remove_gene": self.remove_gene, "add_node": self.add_node,
                "add_latent_parent": self.add_latent_parent}

    # ---- optional ---------------------------------------------------------------
    def crossover(self, a, b, rng):
        """Align by innovation. Matching genes from either parent (disabled if
        disabled in either, with prob 0.75); disjoint/excess genes from `a`."""
        gb = {r[0]: r for r in _rows(b)}
        rows = []
        for r in _rows(a):
            other = gb.get(r[0])
            if other is None:
                rows.append(r)
                continue
            pick = r if rng.random() < 0.5 else other
            en = pick[3]
            if not (r[3] and other[3]):
                en = rng.random() >= 0.75
            rows.append((pick[0], pick[1], pick[2], en))
        return _genome(rows)

    def distance(self, a, b):
        ia, ib = set(a.innov.tolist()), set(b.innov.tolist())
        n = max(len(a), len(b), 1)
        non_matching = len(ia ^ ib)
        common = sorted(ia & ib)
        ea = dict(zip(a.innov.tolist(), a.enabled.tolist()))
        eb = dict(zip(b.innov.tolist(), b.enabled.tolist()))
        mismatch = np.mean([ea[i] != eb[i] for i in common]) if common else 0.0
        return self.c_disjoint * non_matching / n + self.c_enabled * mismatch

    # ---- operators --------------------------------------------------------------
    def _free_latent(self, g):
        used = set(g.src.tolist()) | set(g.dst.tolist())
        return [x for x in range(self.n_obs, self.n) if x not in used]

    def _add(self, g, edges):
        have = set(g.innov.tolist())
        rows = _rows(g)
        for p, c in edges:
            i = self.innovation(p, c)
            if i in have:
                rows = [(r[0], r[1], r[2], True if r[0] == i else r[3]) for r in rows]
            else:
                rows.append((i, int(p), int(c), True))
                have.add(i)
        return _genome(rows)

    def add_connection(self, g, rng):
        free = self._free_latent(g)
        allowed = [x for x in range(self.n) if x < self.n_obs or x not in free] + free[:1]
        p, c = rng.choice(allowed, size=2, replace=False)
        if self.innovation(p, c) in set(g.innov.tolist()):
            return None
        return self._add(g, [(p, c)])

    def toggle_enable(self, g, rng):
        if not len(g):
            return None
        k = rng.integers(len(g))
        en = g.enabled.copy()
        en[k] = ~en[k]
        return Genome(g.innov, g.src, g.dst, en)

    def remove_gene(self, g, rng):
        if not len(g):
            return None
        k = rng.integers(len(g))
        keep = np.arange(len(g)) != k
        return Genome(g.innov[keep], g.src[keep], g.dst[keep], g.enabled[keep])

    def add_node(self, g, rng):
        on = np.nonzero(g.enabled)[0]
        free = self._free_latent(g)
        if not on.size or not free:
            return None
        k = rng.choice(on)
        innov, p, c = int(g.innov[k]), int(g.src[k]), int(g.dst[k])
        lat = self._split.get(innov)
        if lat is None or lat not in free:
            lat = free[0]
            self._split.setdefault(innov, lat)
        en = g.enabled.copy()
        en[k] = False
        return self._add(Genome(g.innov, g.src, g.dst, en), [(p, lat), (lat, c)])

    def add_latent_parent(self, g, rng):
        """A new latent person becomes a parent of two people."""
        free = self._free_latent(g)
        if not free:
            return None
        in_use = [x for x in range(self.n) if x not in free]
        c1, c2 = rng.choice(in_use, size=2, replace=False)
        return self._add(g, [(free[0], c1), (free[0], c2)])
