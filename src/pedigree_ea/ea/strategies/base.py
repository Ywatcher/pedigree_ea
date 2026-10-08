"""Search strategy contract.

A strategy is a function `strategy(engine, **params) -> RunResult`. It decides
only what to vary and what survives; everything else goes through the Engine:

    engine.rep               the Representation (random/empty/mutate/crossover/decode)
    engine.rng               the run's random generator
    engine.evaluate(gs, ops, parents)
                             objectives (B, M) and fit flags for genotypes `gs`;
                             `ops` names the operator that made each and `parents`
                             its parent genotype(s) (statistics and lineage only);
                             also updates the archive (the run's results)
    engine.end_generation(objs, genotypes)
                             log a generation; `genotypes` (rows of `objs`) lets
                             snapshots record the best candidate
    engine.done()            True when a stopping rule is met
    engine.result()          the RunResult to return

Each strategy module also defines
    NAME        registry key used in RunConfig.strategy
    DEFAULTS    default keyword parameters
    OBJECTIVES  default objectives, or None to keep RunConfig's default

To add one: write a module here and add it to `_MODULES` in __init__.py.
"""

from __future__ import annotations

from typing import Callable

from ..engine import Engine, RunResult

Strategy = Callable[..., RunResult]

__all__ = ["Strategy", "Engine", "RunResult"]
