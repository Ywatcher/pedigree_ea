"""(1 + lambda) with Pareto acceptance and restarts.

One parent makes `lam` mutated children per step. A child that dominates the
parent replaces it; otherwise any child the parent does not dominate may
(neutral drift, as in CGP). After `restart_window` evaluations without a new
fit, the search restarts from a random genotype.

`fits_first` (default): a fit (valid and within tolerance) parent is only
replaced by a fit child, and a fit child is always acceptable; moves between
fits are neutral whatever the objectives say. Fit status keeps candidates; it
is not an objective. If it (or a constraint violation) becomes an objective
in future, this rule and the dominance test may merge into one
constrained-domination test.
"""

from __future__ import annotations

import numpy as np

from ..engine import Engine, RunResult
from ..pareto import dominates

NAME = "1+lambda"
DEFAULTS = {"lam": 4, "restart_window": 2000, "fits_first": True}
OBJECTIVES = None


def one_plus_lambda(engine: Engine, lam: int = 4, restart_window: int | None = 2000,
                    fits_first: bool = True) -> RunResult:
    rep, rng = engine.rep, engine.rng
    parent = rep.random(rng)
    p_obj, p_fit = engine.evaluate([parent], ["init"])
    p_obj, p_fit = p_obj[0], bool(p_fit[0])
    last_restart = 0
    engine.end_generation(p_obj[None], [parent])
    while not engine.done():
        kids, ops = zip(*(rep.mutate(parent, rng) for _ in range(lam)))
        k_obj, k_fit = engine.evaluate(list(kids), list(ops), [(parent,)] * lam)
        fit_kids = [i for i in range(lam) if k_fit[i]]
        if fits_first and (p_fit or fit_kids):
            choice = fit_kids            # among fits every move is neutral
        else:
            better = [i for i in range(lam) if dominates(k_obj[i], p_obj)]
            neutral = [i for i in range(lam) if not dominates(p_obj, k_obj[i])]
            choice = better or neutral
        if choice:
            i = choice[rng.integers(len(choice))]
            parent, p_obj, p_fit = kids[i], k_obj[i], bool(k_fit[i])
        since = engine.n_evals - max(engine.archive.last_new_fit, last_restart)
        if restart_window is not None and since >= restart_window:
            parent = rep.random(rng)
            o, f = engine.evaluate([parent], ["restart"])
            p_obj, p_fit = o[0], bool(f[0])
            last_restart = engine.n_evals
        engine.end_generation(np.vstack([p_obj[None], k_obj]), [parent, *kids])
    return engine.result()
