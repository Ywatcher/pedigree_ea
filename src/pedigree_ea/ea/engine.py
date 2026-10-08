"""Search strategies and the bookkeeping they share.

`Engine` evaluates genotypes (decode -> Problem.evaluate, plus the
representation's genotype objectives), updates the archive, counts operator
outcomes, logs progress and checks stopping rules. Strategies only decide
what to vary and what survives:

    nsga2           generational NSGA-II (mu + lambda), optional crossover;
                    duplicate phenotypes survive only after all distinct ones
    one_plus_lambda (1 + lambda) with Pareto acceptance: a child replaces the
                    parent unless the parent dominates it (neutral drift as in
                    CGP); restarts when no new fit is found for a while
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from .. import batch
from .archive import Archive
from .logger import RunLogger
from .problem import Problem
from .representations.base import Representation
from .selection import crowding, dominates, nondominated_ranks, nsga2_select, tournament


@dataclass
class Stopping:
    """Stop when any limit is reached (None = no limit). Windows count evaluations."""
    max_evals: int | None = 20000
    max_seconds: float | None = None
    no_new_fit_window: int | None = None
    no_new_pareto_window: int | None = None
    target_fits: int | None = None   # e.g. size of a known answer key


@dataclass
class RunResult:
    n_evals: int
    seconds: float
    stop_reason: str
    archive: Archive
    history: list[dict] = field(default_factory=list)
    operator_stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"n_evals": self.n_evals, "seconds": round(self.seconds, 3),
                "stop_reason": self.stop_reason, **self.archive.summary()}


class Engine:
    def __init__(self, problem: Problem, rep: Representation, rng: np.random.Generator,
                 stopping: Stopping, logger: RunLogger | None = None):
        self.problem, self.rep, self.rng = problem, rep, rng
        self.stopping, self.logger = stopping, logger
        self.archive = Archive(problem)
        self.n_evals = 0
        self.generation = 0
        self.t0 = time.time()
        self.history: list[dict] = []
        self.op_stats = defaultdict(lambda: {"used": 0, "new_phenotypes": 0, "new_fits": 0})
        self.stop_reason = ""

    def evaluate(self, genotypes: list, ops: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Objectives (B, M) and fit flags (B,) for genotypes."""
        parents = batch.normalize(self.rep.decode_batch(genotypes))
        ev = self.problem.evaluate(parents)
        objs = ev.objectives
        if self.rep.genotype_objective_names:
            extra = np.array([self.rep.genotype_objectives(g) for g in genotypes], dtype=float)
            objs = np.hstack([objs, extra])
        new = self.archive.update(parents, ev, self.n_evals)
        if ops is not None:
            for op, (new_pheno, new_fit, _) in zip(ops, new):
                s = self.op_stats[op]
                s["used"] += 1
                s["new_phenotypes"] += int(new_pheno)
                s["new_fits"] += int(new_fit)
        self.n_evals += len(genotypes)
        return objs, ev.fits

    def end_generation(self, objs: np.ndarray) -> None:
        self.generation += 1
        rec = {"generation": self.generation, "evals": self.n_evals,
               "seconds": round(time.time() - self.t0, 3), **self.archive.summary(),
               "best": objs.min(axis=0).tolist()}
        self.history.append(rec)
        if self.logger is not None:
            self.logger.log(rec)

    def done(self) -> bool:
        s, a = self.stopping, self.archive
        checks = [
            (s.max_evals is not None and self.n_evals >= s.max_evals, "max_evals"),
            (s.max_seconds is not None and time.time() - self.t0 >= s.max_seconds, "max_seconds"),
            (s.no_new_fit_window is not None and a.fits
             and self.n_evals - a.last_new_fit >= s.no_new_fit_window, "no_new_fit_window"),
            (s.no_new_pareto_window is not None
             and self.n_evals - a.last_new_pareto >= s.no_new_pareto_window, "no_new_pareto_window"),
            (s.target_fits is not None and len(a.fits) >= s.target_fits, "target_fits"),
        ]
        for hit, reason in checks:
            if hit:
                self.stop_reason = reason
                return True
        return False

    def result(self) -> RunResult:
        return RunResult(self.n_evals, time.time() - self.t0, self.stop_reason, self.archive,
                         self.history, {k: dict(v) for k, v in self.op_stats.items()})


# ---- strategies ------------------------------------------------------------------

def nsga2(engine: Engine, pop_size: int = 100, crossover_prob: float = 0.0) -> RunResult:
    rep, rng = engine.rep, engine.rng
    if crossover_prob > 0 and not rep.has_crossover:
        raise ValueError(f"{rep.name} has no crossover; set crossover_prob=0")
    pop = [rep.random(rng) for _ in range(pop_size)]
    objs, _ = engine.evaluate(pop, ["init"] * pop_size)
    engine.end_generation(objs)
    while not engine.done():
        ranks = nondominated_ranks(objs)
        dist = crowding(objs, ranks)
        kids, ops = [], []
        for _ in range(pop_size):
            a = tournament(ranks, dist, rng)
            g = pop[a]
            if crossover_prob > 0 and rng.random() < crossover_prob:
                g = rep.crossover(g, pop[tournament(ranks, dist, rng)], rng)
            g, op = rep.mutate(g, rng)
            kids.append(g)
            ops.append(op)
        kid_objs, _ = engine.evaluate(kids, ops)
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
        engine.end_generation(objs)
    return engine.result()


def one_plus_lambda(engine: Engine, lam: int = 4, restart_window: int | None = 2000) -> RunResult:
    rep, rng = engine.rep, engine.rng
    parent = rep.random(rng)
    p_obj, _ = engine.evaluate([parent], ["init"])
    p_obj = p_obj[0]
    last_restart = 0
    engine.end_generation(p_obj[None])
    while not engine.done():
        kids, ops = zip(*(rep.mutate(parent, rng) for _ in range(lam)))
        k_obj, _ = engine.evaluate(list(kids), list(ops))
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
        engine.end_generation(np.vstack([p_obj[None], k_obj]))
    return engine.result()


STRATEGIES = {"nsga2": nsga2, "1+lambda": one_plus_lambda}


def stopping_dict(s: Stopping) -> dict:
    return asdict(s)
