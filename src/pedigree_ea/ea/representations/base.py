"""Representation interface.

A representation owns a genotype type, a decoder to the shared phenotype
(an (N, 2) parent array, see genetics/batch.py), and variation operators. Required:
`random`, `decode`, `operators`. Optional capabilities:
    crossover(a, b, rng)       -> genotype      (has_crossover = True)
    distance(a, b)             -> float         (for speciation / niching)
    genotype_objectives(g)     -> tuple[float]  (names in genotype_objective_names)

Genotypes are treated as immutable: operators return new genotypes.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

Genotype = Any
Operator = Callable[[Genotype, np.random.Generator], "Genotype | None"]


class Representation:
    name = "base"
    has_crossover = False
    genotype_objective_names: tuple[str, ...] = ()
    max_tries = 10

    def __init__(self, n_obs: int, max_latent: int):
        self.n_obs = n_obs
        self.max_latent = max_latent
        self.n = n_obs + max_latent

    # ---- required ---------------------------------------------------------
    def random(self, rng: np.random.Generator) -> Genotype:
        raise NotImplementedError

    def decode(self, g: Genotype) -> np.ndarray:
        raise NotImplementedError

    def operators(self) -> dict[str, Operator]:
        """name -> fn(g, rng) returning a new genotype, or None if not applicable."""
        raise NotImplementedError

    # ---- optional -----------------------------------------------------------
    def empty(self, rng: np.random.Generator) -> Genotype:
        """A genotype decoding to the pedigree with no parent links (POSS starts
        there). Default: a random genotype."""
        return self.random(rng)

    def crossover(self, a: Genotype, b: Genotype, rng: np.random.Generator) -> Genotype:
        raise NotImplementedError(f"{self.name} has no crossover")

    def distance(self, a: Genotype, b: Genotype) -> float:
        raise NotImplementedError(f"{self.name} has no distance")

    def genotype_objectives(self, g: Genotype) -> tuple[float, ...]:
        return ()

    # ---- shared -------------------------------------------------------------
    def mutate(self, g: Genotype, rng: np.random.Generator,
               weights: dict[str, float] | None = None) -> tuple[Genotype, str]:
        """Apply one operator chosen at random (by `weights` if given); retry
        if it does not apply. Returns (new genotype, operator name)."""
        ops = self.operators()
        names = list(ops)
        p = None
        if weights:
            w = np.array([weights.get(n, 0.0) for n in names], dtype=float)
            p = w / w.sum()
        for _ in range(self.max_tries):
            name = names[rng.choice(len(names), p=p)]
            new = ops[name](g, rng)
            if new is not None:
                return new, name
        return g, "none"

    def decode_batch(self, genotypes: list[Genotype]) -> np.ndarray:
        return np.stack([self.decode(g) for g in genotypes])
