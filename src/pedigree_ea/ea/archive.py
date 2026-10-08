"""Everything a run has found, independent of the population.

Every fit is kept here as soon as it is found and never removed, whatever the
population does; this is not a strategy choice. Strategies may decide
separately whether the population prioritizes fits (`fits_first`) and,
later, whether fits are used as parents (they can read `fits[i].genotype`).

- distinct phenotypes evaluated (by parent-array key; cheap, not relabel-invariant)
- fitting pedigrees, deduplicated by structure, with the evaluation they appeared at
- the Pareto front of distinct structures over the phenotype objectives
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..genetics import batch
from ..genetics.canonical import PedigreeSet
from ..genetics.pedigree import Pedigree
from .problem import Evaluation, Problem
from .pareto import dominates


@dataclass
class Found:
    pedigree: Pedigree
    objectives: np.ndarray
    at_eval: int
    genotype: Any = None      # the genotype that first produced it (for strategies that reuse fits)


class Archive:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.phenotypes: set[bytes] = set()
        self.fits: list[Found] = []
        self._fit_set = PedigreeSet(prune=True)
        self.pareto: list[Found] = []
        self.last_new_fit = 0
        self.last_new_pareto = 0
        self.last_new_phenotype = 0

    def update(self, parents: np.ndarray, ev: Evaluation, first_eval: int,
               genotypes: list | None = None) -> np.ndarray:
        """Record a batch evaluated as evaluations first_eval .. first_eval+B-1
        (`genotypes`: the batch's genotypes, kept with new fits).

        Returns flags (B, 3): new phenotype, new fitting structure, new Pareto point.
        """
        parents = batch.normalize(parents)
        new = np.zeros((parents.shape[0], 3), dtype=bool)
        for b in range(parents.shape[0]):
            at = first_eval + b
            k = batch.key(parents[b])
            if k in self.phenotypes:
                continue
            self.phenotypes.add(k)
            self.last_new_phenotype = at
            new[b, 0] = True
            if not ev.valid[b]:
                continue
            ped = None
            if ev.fits[b]:
                ped = batch.to_pedigree(parents[b], self.problem.ids)
                if self._fit_set.add(ped):
                    self.fits.append(Found(ped, ev.objectives[b], at,
                                           genotypes[b] if genotypes is not None else None))
                    self.last_new_fit = at
                    new[b, 1] = True
            if self._update_pareto(parents[b], ev.objectives[b], at, ped):
                self.last_new_pareto = at
                new[b, 2] = True
        return new

    def _update_pareto(self, parents, obj, at, ped) -> bool:
        for f in self.pareto:
            if dominates(f.objectives, obj):
                return False
        same = [f for f in self.pareto if np.array_equal(f.objectives, obj)]
        ped = ped or batch.to_pedigree(parents, self.problem.ids)
        s = PedigreeSet()
        for f in same:
            s.add(f.pedigree)
        if ped in s:
            return False
        self.pareto = [f for f in self.pareto if not dominates(obj, f.objectives)]
        self.pareto.append(Found(ped, obj.copy(), at))
        return True

    def summary(self) -> dict[str, int]:
        return {"n_phenotypes": len(self.phenotypes), "n_fits": len(self.fits),
                "n_pareto": len(self.pareto)}
