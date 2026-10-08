"""POSS-style Pareto optimization, adapted to finding many pedigrees.

POSS (Qian, Yu & Zhou, 2015; the GSEMO family) keeps no fixed-size
population: the population IS the archive of non-dominated solutions. It
starts from the empty solution; each step picks a parent uniformly from the
archive, mutates it, and adds the child unless a member dominates it,
removing the members the child dominates. The usual second objective is size
(here: number of latent people).

Adaptations for this project:
- Distinct structures with EQUAL objective values are all kept (standard POSS
  keeps one per value), because the goal is many equally good pedigrees. Use
  objectives that ignore error within tolerance (`excess_total`, `n_latent`),
  so every fit scores the same and none is pushed out by one fitting the noise.
- `lam` children per step, evaluated as one batch.
- Each child gets 1 + Poisson(`extra_mutations`) operator applications,
  mimicking bit-wise mutation's occasional multi-bit jumps.
- The archive is capped at `max_archive`; when full, a random member of the
  most crowded objective value is dropped.
- `parent_selection`: "uniform" (POSS) or "least_used" (weights members picked
  fewer times higher, to spread effort over a large archive).

The approximation guarantees of POSS assume a (nearly) submodular objective
over subsets; IBD error over pedigrees is not, so none carry over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...genetics import batch
from ...genetics.canonical import same_structure, structure_hash
from ...genetics.pedigree import Pedigree
from ..engine import Engine, RunResult

NAME = "poss"
DEFAULTS = {"lam": 8, "extra_mutations": 0.5, "max_archive": 2000,
            "parent_selection": "uniform"}
OBJECTIVES = ("excess_total", "n_latent")


@dataclass
class _Member:
    genotype: Any
    obj: np.ndarray
    key: bytes          # normalized parent array (cheap duplicate check)
    pedigree: Pedigree  # pruned, for structure identity
    shash: str
    uses: int = 0


def poss(engine: Engine, lam: int = 8, extra_mutations: float = 0.5,
         max_archive: int = 2000, parent_selection: str = "uniform",
         start_empty: bool = True) -> RunResult:
    if parent_selection not in ("uniform", "least_used"):
        raise ValueError(f"parent_selection must be 'uniform' or 'least_used', got {parent_selection!r}")
    rep, rng, ids = engine.rep, engine.rng, engine.problem.ids
    pop: list[_Member] = []
    keys: set[bytes] = set()

    def objs() -> np.ndarray:
        return np.array([m.obj for m in pop])

    def offer(g, obj: np.ndarray) -> bool:
        """Add a child unless dominated or a duplicate structure; drop members it dominates."""
        parents = batch.normalize(rep.decode(g))
        k = batch.key(parents)
        if k in keys:
            return False
        if pop:
            o = objs()
            if ((o <= obj).all(axis=1) & (o < obj).any(axis=1)).any():
                return False
        ped = batch.to_pedigree(parents, ids)
        h = structure_hash(ped)
        if any(m.shash == h and same_structure(m.pedigree, ped) for m in pop):
            keys.add(k)          # same structure under another labelling
            return False
        if pop:
            o = objs()
            dominated = (obj <= o).all(axis=1) & (obj < o).any(axis=1)
            for m in [m for m, d in zip(pop, dominated) if d]:
                keys.discard(m.key)
            pop[:] = [m for m, d in zip(pop, dominated) if not d]
        pop.append(_Member(g, obj.copy(), k, ped, h))
        keys.add(k)
        if len(pop) > max_archive:
            _drop_most_crowded()
        return True

    def _drop_most_crowded() -> None:
        groups: dict[bytes, list[int]] = {}
        for i, m in enumerate(pop):
            groups.setdefault(m.obj.tobytes(), []).append(i)
        biggest = max(groups.values(), key=len)
        i = biggest[rng.integers(len(biggest))]
        keys.discard(pop[i].key)
        pop.pop(i)

    def pick() -> _Member:
        if parent_selection == "uniform":
            m = pop[rng.integers(len(pop))]
        else:
            w = 1.0 / (1.0 + np.array([m.uses for m in pop], dtype=float))
            m = pop[rng.choice(len(pop), p=w / w.sum())]
        m.uses += 1
        return m

    start = rep.empty(rng) if start_empty else rep.random(rng)
    obj, _ = engine.evaluate([start], ["init"])
    offer(start, obj[0])
    engine.end_generation(objs(), [m.genotype for m in pop])
    while not engine.done():
        kids, ops, parents = [], [], []
        for _ in range(lam):
            g = pick().genotype
            parents.append((g,))
            n_ops = 1 + int(rng.poisson(extra_mutations))
            names = []
            for _ in range(n_ops):
                g, op = rep.mutate(g, rng)
                names.append(op)
            kids.append(g)
            ops.append(names[0] if n_ops == 1 else "multi")
        kid_objs, _ = engine.evaluate(kids, ops, parents)
        for g, obj in zip(kids, kid_objs):
            offer(g, obj)
        engine.end_generation(objs(), [m.genotype for m in pop])
    return engine.result()
