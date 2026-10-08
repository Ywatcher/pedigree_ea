"""CGP-like representation: every person reserves 2 input genes (their parents).

Following ideas.md ("reserve 2 inputs for each individual"). The genotype is
    order   (N,)   a permutation of person slots (the CGP node order)
    inputs  (N, 2) for the person at position t, positions < t of their
                   parents, or -1 for a missing parent
Inputs may only point to earlier positions, so every genotype decodes to an
acyclic pedigree (no repair needed), like feed-forward CGP. Latent people
nobody points to are inactive: their genes mutate freely without changing the
phenotype (CGP-style neutral drift).

Operators: point_mutation (re-draw input genes), swap_order (exchange two
people in the order; references follow the people, references that would
point forward become -1). Baseline strategy: (1+lambda).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...genetics import batch
from .base import Representation


@dataclass(frozen=True)
class Genome:
    order: np.ndarray
    inputs: np.ndarray


class CGP(Representation):
    name = "cgp"

    def __init__(self, n_obs: int, max_latent: int, p_missing: float = 0.6,
                 mutation_rate: float = 0.1):
        super().__init__(n_obs, max_latent)
        self.p_missing = p_missing
        self.mutation_rate = mutation_rate

    def _draw(self, t: int, rng) -> int:
        if t == 0 or rng.random() < self.p_missing:
            return -1
        return int(rng.integers(t))

    def random(self, rng):
        order = rng.permutation(self.n).astype(np.int16)
        inputs = np.array([[self._draw(t, rng), self._draw(t, rng)] for t in range(self.n)],
                          dtype=np.int16)
        return Genome(order, inputs)

    def empty(self, rng):
        return Genome(rng.permutation(self.n).astype(np.int16),
                      np.full((self.n, 2), -1, dtype=np.int16))

    def decode(self, g):
        parents = np.full((self.n, 2), -1, dtype=np.int16)
        ok = g.inputs >= 0
        parents[g.order] = np.where(ok, g.order[np.where(ok, g.inputs, 0)], -1)
        return batch.normalize(parents)

    def operators(self):
        return {"point_mutation": self.point_mutation, "swap_order": self.swap_order}

    def point_mutation(self, g, rng):
        inputs = g.inputs.copy()
        mask = rng.random(inputs.shape) < self.mutation_rate
        if not mask.any():
            mask.flat[rng.integers(mask.size)] = True
        for t, k in np.argwhere(mask):
            inputs[t, k] = self._draw(int(t), rng)
        return Genome(g.order, inputs)

    def swap_order(self, g, rng):
        t1, t2 = sorted(rng.choice(self.n, size=2, replace=False))
        order, inputs = g.order.copy(), g.inputs.copy()
        order[[t1, t2]] = order[[t2, t1]]
        inputs[[t1, t2]] = inputs[[t2, t1]]
        swapped = inputs.copy()
        swapped[inputs == t1] = t2
        swapped[inputs == t2] = t1
        pos = np.arange(self.n)[:, None]
        swapped[swapped >= pos] = -1
        return Genome(order, swapped)
