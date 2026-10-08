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
- `fits_first` (default): fits, i.e. solutions (valid and within tolerance),
  are never rejected or removed because another candidate dominates them, and
  when the archive is full a non-fit is dropped first. Fit status keeps
  candidates; it is not an objective. If it (or a constraint violation)
  becomes an objective in future, this rule and the dominance test may merge
  into one constrained-domination test.

The approximation guarantees of POSS assume a (nearly) submodular objective
over subsets; IBD error over pedigrees is not, so none carry over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...genetics import batch
from ..engine import Engine, RunResult

NAME = "poss"
DEFAULTS = {"lam": 8, "extra_mutations": 0.5, "max_archive": 2000,
            "parent_selection": "uniform", "fits_first": True}
OBJECTIVES = ("excess_total", "n_latent")


@dataclass
class _Member:
    genotype: Any
    obj: np.ndarray
    key: bytes          # normalized parent array (cheap duplicate check)
    structure: bytes    # canonical form (exact structure identity)
    fit: bool
    uses: int = 0


def poss(engine: Engine, lam: int = 8, extra_mutations: float = 0.5,
         max_archive: int = 2000, parent_selection: str = "uniform",
         start_empty: bool = True, fits_first: bool = True) -> RunResult:
    if parent_selection not in ("uniform", "least_used"):
        raise ValueError(f"parent_selection must be 'uniform' or 'least_used', got {parent_selection!r}")
    rep, rng = engine.rep, engine.rng
    pop: list[_Member] = []
    keys: set[bytes] = set()
    structures: set[bytes] = set()

    def objs() -> np.ndarray:
        return np.array([m.obj for m in pop])

    def offer(g, obj: np.ndarray, fit: bool) -> bool:
        """Add a child unless dominated (fits exempt with fits_first) or a
        duplicate structure; drop members it dominates (fits exempt)."""
        k = batch.key(batch.normalize(rep.decode(g)))
        if k in keys:
            return False
        protected = fits_first and fit
        if pop and not protected:
            o = objs()
            if ((o <= obj).all(axis=1) & (o < obj).any(axis=1)).any():
                return False
        (sk,) = engine.structure_keys([g])
        if sk in structures:
            keys.add(k)          # same structure under another labelling
            return False
        if pop:
            o = objs()
            dominated = (obj <= o).all(axis=1) & (obj < o).any(axis=1)
            if fits_first:
                dominated &= ~np.array([m.fit for m in pop])
            for m in [m for m, d in zip(pop, dominated) if d]:
                keys.discard(m.key)
                structures.discard(m.structure)
            pop[:] = [m for m, d in zip(pop, dominated) if not d]
        pop.append(_Member(g, obj.copy(), k, sk, bool(fit)))
        keys.add(k)
        structures.add(sk)
        if len(pop) > max_archive:
            _drop_most_crowded()
        return True

    def _drop_most_crowded() -> None:
        """Drop a random member of the most crowded objective value, non-fits first."""
        candidates = [i for i, m in enumerate(pop) if not (fits_first and m.fit)] or list(range(len(pop)))
        groups: dict[bytes, list[int]] = {}
        for i in candidates:
            groups.setdefault(pop[i].obj.tobytes(), []).append(i)
        biggest = max(groups.values(), key=len)
        i = biggest[rng.integers(len(biggest))]
        keys.discard(pop[i].key)
        structures.discard(pop[i].structure)
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
    obj, fit = engine.evaluate([start], ["init"])
    offer(start, obj[0], fit[0])
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
        kid_objs, kid_fits = engine.evaluate(kids, ops, parents)
        for g, obj, fit in zip(kids, kid_objs, kid_fits):
            offer(g, obj, fit)
        engine.end_generation(objs(), [m.genotype for m in pop])
    return engine.result()
