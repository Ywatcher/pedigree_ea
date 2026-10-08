"""Run configurations, running on a test case, and scoring against its answer key."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from ...genetics.canonical import PedigreeSet, same_structure
from ...data.cases import Case
from ...genetics.ibd import IBDData
from ...genetics.king import KingData
from ...genetics.kinship import KinshipData
from ...genetics.kinship import inbreeding
from ..engine import Engine, RunResult, Stopping
from ...genetics.pedigree import Pedigree
from ...store.writer import RunWriter, set_score
from ..recording import Recording
from .logger import RunLogger
from ..problem import Problem
from ..representations import REPRESENTATIONS
from ..strategies import STRATEGIES, STRATEGY_DEFAULTS, STRATEGY_OBJECTIVES


@dataclass
class RunConfig:
    representation: str = "direct"
    representation_params: dict[str, Any] = field(default_factory=dict)
    strategy: str = "nsga2"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    objectives: tuple[str, ...] = ("ibd_total", "ibd_worst", "n_latent")
    max_latent: int = 2
    tol: float = 0.01
    tol_ibd0: float = 0.15          # KING (kinship + IBS0) targets only
    check_sex: bool = True
    seed: int = 0
    stopping: Stopping = field(default_factory=Stopping)
    snapshot: bool = False          # record per-generation candidates (see ea/recording.py)
    record_closest: int = 1         # candidates kept per reference in each snapshot
    record_rank_by: str = "structure"   # or "relationship"

    def to_dict(self) -> dict:
        return asdict(self)


# Baseline pairing of each implemented representation with a strategy.
BASELINES: dict[str, dict[str, Any]] = {
    "direct": {"strategy": "nsga2", "strategy_params": {"pop_size": 100, "crossover_prob": 0.0}},
    "neat": {"strategy": "nsga2", "strategy_params": {"pop_size": 100, "crossover_prob": 0.5}},
    "cgp": {"strategy": "1+lambda", "strategy_params": {"lam": 4, "restart_window": 2000}},
}


def baseline_config(representation: str, strategy: str | None = None, **overrides) -> RunConfig:
    """Config for a representation with its baseline strategy, or with `strategy`
    (its default parameters and objectives). `overrides` set any RunConfig field;
    `strategy_params` given here are merged onto the defaults."""
    if strategy is None:
        base = BASELINES[representation]
        cfg = RunConfig(representation=representation, strategy=base["strategy"],
                        strategy_params=dict(base["strategy_params"]))
    else:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
        cfg = RunConfig(representation=representation, strategy=strategy,
                        strategy_params=dict(STRATEGY_DEFAULTS[strategy]))
        if strategy in STRATEGY_OBJECTIVES:
            cfg.objectives = STRATEGY_OBJECTIVES[strategy]
    params = overrides.pop("strategy_params", None)
    if params:
        cfg.strategy_params.update(params)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown RunConfig field {k!r}")
        setattr(cfg, k, v)
    return cfg


def run(config: RunConfig, target: IBDData | KinshipData | KingData,
        logger: RunLogger | None = None,
        references: Mapping[str, Pedigree] | None = None,
        case_name: str | None = None) -> RunResult:
    """Run one search. `references` (label -> pedigree) are seen only by the
    recorder (in-memory snapshots, `config.snapshot`) and stored with the run in
    the database, never by the search. With a logger (db on), the run record
    goes to `logger.db_file`."""
    problem = Problem(target, max_latent=config.max_latent, tol=config.tol,
                      tol_ibd0=config.tol_ibd0, objectives=tuple(config.objectives),
                      check_sex=config.check_sex)
    rep = REPRESENTATIONS[config.representation](problem.n_obs, config.max_latent,
                                                 **config.representation_params)
    recording = (Recording(problem, references, config.record_closest, config.record_rank_by)
                 if config.snapshot else None)
    store = None
    if logger is not None and logger.db:
        store = RunWriter(logger.db_file, name=logger.name, grid=logger.grid, stamp=logger.stamp,
                          case_name=case_name, config=config.to_dict(), seed=config.seed,
                          target=target, observed_ids=problem.ids, objectives=config.objectives,
                          references=references, keyframe_every=logger.keyframe_every,
                          scope=logger.scope)
    engine = Engine(problem, rep, np.random.default_rng(config.seed), config.stopping, logger,
                    snapshot=config.snapshot, recording=recording, store=store)
    if logger is not None:
        logger.log({"event": "start", "config": config.to_dict()})
    result = STRATEGIES[config.strategy](engine, **config.strategy_params)
    if store is not None:
        logger.db_run_id = store.finish(result.summary())
    if logger is not None:
        logger.log({"event": "end", **result.summary()})
    return result


def score(result: RunResult, case: Case) -> dict[str, Any]:
    """Compare fitting pedigrees found with the case's answer key.

    recall          fraction of answer-key pedigrees found
    recall_outbred  same, counting only answer-key pedigrees without inbreeding
    extra           fits not in the answer key (should be 0 when bounds/tol match)
    truth_at_eval   evaluation at which the truth pedigree was found (None if
                    not found or the case has no truth)
    """
    found = PedigreeSet()
    for f in result.archive.fits:
        found.add(f.pedigree)
    truth_at = None
    if case.truth is not None:   # real data has no known truth
        truth = case.truth.pruned()
        truth_at = next((f.at_eval for f in result.archive.fits
                         if same_structure(f.pedigree, truth)), None)
    out: dict[str, Any] = {"n_fits": len(result.archive.fits), "truth_at_eval": truth_at}
    if case.solutions is None:
        return out
    key = PedigreeSet()
    for s in case.solutions:
        key.add(s)
    outbred = [s for s in case.solutions if all(v < 1e-12 for v in inbreeding(s).values())]
    hit = [s for s in case.solutions if s in found]
    hit_out = [s for s in outbred if s in found]
    out.update({
        "n_key": len(case.solutions),
        "recall": len(hit) / len(case.solutions) if case.solutions else None,
        "n_key_outbred": len(outbred),
        "recall_outbred": len(hit_out) / len(outbred) if outbred else None,
        "extra": sum(f.pedigree not in key for f in result.archive.fits),
    })
    return out


def case_references(case: Case, extra: Mapping[str, Pedigree] | None = None) -> dict[str, Pedigree]:
    """The case's truth (label "truth") plus any extra references."""
    refs = {"truth": case.truth} if case.truth is not None else {}
    refs.update(extra or {})
    return refs


def run_case(config: RunConfig, case: Case, logger: RunLogger | None = None,
             references: Mapping[str, Pedigree] | None = None) -> tuple[RunResult, dict]:
    """Run on a test case and score it; with a logger, also save the result file.
    With snapshots on, the recorder tracks candidates closest to the case's truth
    and to `references`."""
    result = run(config, case.target, logger, case_references(case, references), case.name)
    sc = score(result, case)
    if logger is not None and logger.db_run_id is not None:
        set_score(logger.db_file, logger.db_run_id, sc)
    if logger is not None:
        logger.save_result({"case": case.name, "config": config.to_dict(),
                            "summary": result.summary(), "score": sc,
                            "operator_stats": result.operator_stats,
                            "fits": [{"at_eval": f.at_eval, "objectives": f.objectives,
                                      "pedigree": str(f.pedigree)} for f in result.archive.fits]},
                           fits=result.archive.fits)
        logger.close()
    return result, sc
