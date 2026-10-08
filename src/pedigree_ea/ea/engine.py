"""The bookkeeping every search strategy shares (strategies live in strategies/).

`Engine` evaluates genotypes (decode -> Problem.evaluate, plus the
representation's genotype objectives), updates the archive, counts operator
outcomes, logs progress, records optional snapshots and checks stopping rules.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from ..genetics import batch
from .archive import Archive
from .experiments.logger import RunLogger
from .problem import Problem
from ..genetics.canonical import canonical_form
from ..store.writer import RunWriter
from .recording import Recording, events_for_log, records_for_log
from .representations.base import Representation


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
    snapshots: list[dict] = field(default_factory=list)   # filled when Engine(snapshot=True)

    def summary(self) -> dict:
        return {"n_evals": self.n_evals, "seconds": round(self.seconds, 3),
                "stop_reason": self.stop_reason, **self.archive.summary()}


class Engine:
    def __init__(self, problem: Problem, rep: Representation, rng: np.random.Generator,
                 stopping: Stopping, logger: RunLogger | None = None, snapshot: bool = False,
                 recording: Recording | None = None, store: RunWriter | None = None):
        """With `snapshot`, each generation's records (see recording.py) are kept in
        `snapshots` and written to the log; `recording` defaults to best-only.
        With `store`, the run is written to a database (see store/): population
        changes per generation, and each candidate's first appearance (lineage,
        objectives at entry) and when it became a fit."""
        self.problem, self.rep, self.rng = problem, rep, rng
        self.stopping, self.logger = stopping, logger
        self.snapshot = snapshot
        self.recording = recording or (Recording(problem) if snapshot else None)
        self.snapshots: list[dict] = []
        self.archive = Archive(problem)
        self.n_evals = 0
        self.generation = 0
        self.t0 = time.time()
        self.history: list[dict] = []
        self.op_stats = defaultdict(lambda: {"used": 0, "new_phenotypes": 0, "new_fits": 0})
        self.stop_reason = ""
        self.store = store
        self._first: dict[bytes, dict] = {}     # this generation's first evaluations (for the store)
        self._structure: dict[bytes, bytes] = {}  # parent-array key -> canonical form (cache)

    def evaluate(self, genotypes: list, ops: list[str] | None = None,
                 parents: list[tuple] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Objectives (B, M) and fit flags (B,) for genotypes. `ops` names the
        operator that made each genotype and `parents` its parent genotype(s);
        both are only recorded (statistics, lineage)."""
        arrays = batch.normalize(self.rep.decode_batch(genotypes))
        ev = self.problem.evaluate(arrays)
        objs = ev.objectives
        if self.rep.genotype_objective_names:
            extra = np.array([self.rep.genotype_objectives(g) for g in genotypes], dtype=float)
            objs = np.hstack([objs, extra])
        new = self.archive.update(arrays, ev, self.n_evals, genotypes)
        if ops is not None:
            for op, (new_pheno, new_fit, _) in zip(ops, new):
                s = self.op_stats[op]
                s["used"] += 1
                s["new_phenotypes"] += int(new_pheno)
                s["new_fits"] += int(new_fit)
        if self.store is not None:
            self._store_evaluations(parents or [()] * len(genotypes), arrays, objs, ops, new)
        self.n_evals += len(genotypes)
        return objs, ev.fits

    def structure_keys(self, genotypes: list) -> list[bytes]:
        """Canonical form of each genotype's pedigree: equal iff same structure
        (relabelled copies included). Cached by parent array."""
        out = []
        for g in genotypes:
            arr = batch.normalize(self.rep.decode(g))
            k = batch.key(arr)
            c = self._structure.get(k)
            if c is None:
                ped = batch.to_pedigree(arr, self.problem.ids, prune=False)
                c = self._structure[k] = canonical_form(ped, self.problem.ids)
            out.append(c)
        return out

    def _store_evaluations(self, parent_genotypes, arrays, objs, ops, new) -> None:
        store = self.store
        gen = self.generation + 1
        for b, arr in enumerate(arrays):
            k = batch.key(arr)
            first = self._first.get(k)
            if first is None:
                pcs = [store.candidate(batch.normalize(self.rep.decode(g)))
                       for g in parent_genotypes[b]]
                first = self._first[k] = {"eval": self.n_evals + b, "op": ops[b] if ops else None,
                                          "parents": pcs, "obj": objs[b]}
            if new[b, 1] or (store.scope == "all" and new[b, 0]):
                store.appear(store.candidate(arr), first_eval=first["eval"], first_gen=gen,
                             operator=first["op"], parents=first["parents"], objectives=first["obj"],
                             fit_at_eval=self.n_evals + b if new[b, 1] else None)

    def end_generation(self, objs: np.ndarray, genotypes: list | None = None) -> None:
        """Log a generation. With snapshots on, `genotypes` (rows of `objs`) are
        handed to the recorder."""
        self.generation += 1
        rec = {"generation": self.generation, "evals": self.n_evals,
               "seconds": round(time.time() - self.t0, 3), **self.archive.summary(),
               "population": len(objs), "best": objs.min(axis=0).tolist()}
        if self.snapshot and genotypes is not None:
            snap = self._snap(objs, genotypes)
            rec["records"] = records_for_log(snap["records"])
            if snap["events"]:
                rec["removed"] = events_for_log(snap["events"])
        if self.store is not None and genotypes is not None:
            self._store_generation(objs, genotypes)
        self.history.append(rec)
        if self.logger is not None:
            self.logger.log(rec)

    def _store_generation(self, objs: np.ndarray, genotypes: list) -> None:
        store = self.store
        members: dict[int, int] = {}
        for row, g in zip(objs, genotypes):
            arr = batch.normalize(self.rep.decode(g))
            cid = store.candidate(arr)
            members[cid] = members.get(cid, 0) + 1
            first = self._first.get(batch.key(arr), {})
            store.appear(cid, first_eval=first.get("eval"), first_gen=self.generation,
                         operator=first.get("op"), parents=first.get("parents", ()),
                         objectives=row, in_population=True)
        store.generation(self.generation, members, evals=self.n_evals,
                         seconds=round(time.time() - self.t0, 3),
                         n_phenotypes=len(self.archive.phenotypes), n_fits=len(self.archive.fits),
                         n_pareto=len(self.archive.pareto))
        self._first.clear()

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

    def _snap(self, objs: np.ndarray, genotypes: list) -> dict:
        fits = self.archive.fits
        recorded = self.recording.record(objs, self.rep.decode_batch(genotypes), self.generation)
        snap = {
            "generation": self.generation, "evals": self.n_evals, "population": len(objs),
            **recorded,          # records, events (removals now), last_removed
            "n_fits": len(fits), "latest_fit": fits[-1].pedigree if fits else None,
            "latest_fit_at": fits[-1].at_eval if fits else None,
        }
        self.snapshots.append(snap)
        return snap

    def result(self) -> RunResult:
        return RunResult(self.n_evals, time.time() - self.t0, self.stop_reason, self.archive,
                         self.history, {k: dict(v) for k, v in self.op_stats.items()},
                         self.snapshots)


def stopping_dict(s: Stopping) -> dict:
    return asdict(s)
