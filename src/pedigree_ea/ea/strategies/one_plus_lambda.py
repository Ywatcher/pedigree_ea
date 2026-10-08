"""(1 + lambda) with Pareto acceptance and restarts.

One parent makes `lam` mutated children per step. A child that dominates the
parent replaces it; otherwise any child the parent does not dominate may
(neutral drift, as in CGP). After `restart_window` evaluations without a new
fit, the search restarts from a random genotype.
"""

from __future__ import annotations

import numpy as np

from ..engine import Engine, RunResult
from ..pareto import dominates

NAME = "1+lambda"
DEFAULTS = {"lam": 4, "restart_window": 2000}
OBJECTIVES = None


def one_plus_lambda(engine: Engine, lam: int = 4, restart_window: int | None = 2000) -> RunResult:
    rep, rng = engine.rep, engine.rng
    parent = rep.random(rng)
    p_obj, _ = engine.evaluate([parent], ["init"])
    p_obj = p_obj[0]
    last_restart = 0
    engine.end_generation(p_obj[None], [parent])
    while not engine.done():
        kids, ops = zip(*(rep.mutate(parent, rng) for _ in range(lam)))
        k_obj, _ = engine.evaluate(list(kids), list(ops), [(parent,)] * lam)
        better = [i for i in range(lam) if dominates(k_obj[i], p_obj)]
        neutral = [i for i in range(lam) if not dominates(p_obj, k_obj[i])]
        choice = better or neutral
        if choice:
            i = choice[rng.integers(len(choice))]
            parent, p_obj = kids[i], k_obj[i]
        since = engine.n_evals - max(engine.archive.last_new_fit, last_restart)
        if restart_window is not None and since >= restart_window:
            parent = rep.random(rng)
            p_obj = engine.evaluate([parent], ["restart"])[0][0]
            last_restart = engine.n_evals
        engine.end_generation(np.vstack([p_obj[None], k_obj]), [parent, *kids])
    return engine.result()
