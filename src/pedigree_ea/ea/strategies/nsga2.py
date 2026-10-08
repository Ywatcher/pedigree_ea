"""NSGA-II: generational (mu + lambda) with Pareto ranking and crowding.

Each generation, `pop_size` parents make `pop_size` children (binary
tournament on rank and crowding, optional crossover, one mutation). From the
pooled 2 * pop_size, the best `pop_size` survive by front and crowding;
duplicate phenotypes (identical parent arrays) survive only after all
distinct ones.
"""

from __future__ import annotations

import numpy as np

from ...genetics import batch
from ..engine import Engine, RunResult
from ..pareto import crowding, nondominated_ranks, nsga2_select, tournament

NAME = "nsga2"
DEFAULTS = {"pop_size": 100, "crossover_prob": 0.0}
OBJECTIVES = None   # use RunConfig's default objectives


def nsga2(engine: Engine, pop_size: int = 100, crossover_prob: float = 0.0) -> RunResult:
    rep, rng = engine.rep, engine.rng
    if crossover_prob > 0 and not rep.has_crossover:
        raise ValueError(f"{rep.name} has no crossover; set crossover_prob=0")
    pop = [rep.random(rng) for _ in range(pop_size)]
    objs, _ = engine.evaluate(pop, ["init"] * pop_size)
    engine.end_generation(objs, pop)
    while not engine.done():
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
        kid_objs, _ = engine.evaluate(kids, ops, parents)
        allg = pop + kids
        allo = np.vstack([objs, kid_objs])
        keys = [batch.key(batch.normalize(rep.decode(g))) for g in allg]
        dup = np.zeros(len(allg), dtype=bool)
        seen: set[bytes] = set()
        for i, k in enumerate(keys):
            dup[i] = k in seen
            seen.add(k)
        keep = nsga2_select(allo, pop_size, penalize=dup)
        pop = [allg[i] for i in keep]
        objs = allo[keep]
        engine.end_generation(objs, pop)
    return engine.result()
