"""NSGA-II: generational (mu + lambda) with Pareto ranking and crowding.

Each generation, `pop_size` parents make `pop_size` children (binary
tournament on rank and crowding, optional crossover, one mutation). From the
pooled 2 * pop_size, the best `pop_size` survive by rank and crowding.

`fits_first` (default): in survival, fits, i.e. solutions (valid and within
tolerance), rank before every non-fit and do not compete with each other, so a
preference objective such as n_latent cannot push a solution out; duplicates
are detected by structure (canonical form), so relabelled copies do not crowd
out distinct pedigrees. Non-fits keep their usual Pareto ranks among
themselves. Parent selection still uses plain Pareto ranks, so fits are kept
but do not monopolize reproduction.
With `fits_first=False`: plain Pareto ranking, duplicates by parent array.
"""

from __future__ import annotations

import numpy as np

from ... import batch
from ..engine import Engine, RunResult
from ..pareto import crowding, fit_first_ranks, nondominated_ranks, select_by_ranks, tournament

NAME = "nsga2"
DEFAULTS = {"pop_size": 100, "crossover_prob": 0.0, "fits_first": True}
OBJECTIVES = None   # use RunConfig's default objectives


def _duplicates(keys: list[bytes]) -> np.ndarray:
    dup = np.zeros(len(keys), dtype=bool)
    seen: set[bytes] = set()
    for i, k in enumerate(keys):
        dup[i] = k in seen
        seen.add(k)
    return dup


def nsga2(engine: Engine, pop_size: int = 100, crossover_prob: float = 0.0,
          fits_first: bool = True) -> RunResult:
    rep, rng = engine.rep, engine.rng
    if crossover_prob > 0 and not rep.has_crossover:
        raise ValueError(f"{rep.name} has no crossover; set crossover_prob=0")

    # Fit status keeps candidates; it is not an objective. If it (or a
    # constraint violation) becomes an objective in future, fit_first_ranks and
    # the Pareto ranking may merge into one constrained-domination ranking.
    def ranks_of(objs, fits):
        return fit_first_ranks(objs, fits) if fits_first else nondominated_ranks(objs)

    pop = [rep.random(rng) for _ in range(pop_size)]
    objs, fits = engine.evaluate(pop, ["init"] * pop_size)
    engine.end_generation(objs, pop)
    while not engine.done():
        # Parents are chosen by plain Pareto rank: fit status protects survival
        # only, so the search keeps exploring from non-fits near other answers.
        ranks = nondominated_ranks(objs)
        dist = crowding(objs, ranks)
        kids, ops, parents = [], [], []
        for _ in range(pop_size):
            a = tournament(ranks, dist, rng)
            g, par = pop[a], (pop[a],)
            if crossover_prob > 0 and rng.random() < crossover_prob:
                b = tournament(ranks, dist, rng)
                g, par = rep.crossover(g, pop[b], rng), (pop[a], pop[b])
            g, op = rep.mutate(g, rng)
            kids.append(g)
            ops.append(op)
            parents.append(par)
        kid_objs, kid_fits = engine.evaluate(kids, ops, parents)
        allg = pop + kids
        allo = np.vstack([objs, kid_objs])
        allf = np.concatenate([fits, kid_fits])
        keys = (engine.structure_keys(allg) if fits_first
                else [batch.key(batch.normalize(rep.decode(g))) for g in allg])
        keep = select_by_ranks(allo, ranks_of(allo, allf), pop_size, penalize=_duplicates(keys))
        pop = [allg[i] for i in keep]
        objs, fits = allo[keep], allf[keep]
        engine.end_generation(objs, pop)
    return engine.result()
