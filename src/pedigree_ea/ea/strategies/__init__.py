"""Search strategies. Each module defines a function plus NAME, DEFAULTS and
OBJECTIVES (see base.py); this registry collects them."""

from . import nsga2 as _nsga2
from . import one_plus_lambda as _one_plus_lambda
from . import poss as _poss
from .base import Strategy
from .nsga2 import nsga2
from .one_plus_lambda import one_plus_lambda
from .poss import poss

_MODULES = [(_nsga2, nsga2), (_one_plus_lambda, one_plus_lambda), (_poss, poss)]

STRATEGIES: dict[str, Strategy] = {m.NAME: fn for m, fn in _MODULES}
STRATEGY_DEFAULTS: dict[str, dict] = {m.NAME: dict(m.DEFAULTS) for m, _ in _MODULES}
STRATEGY_OBJECTIVES: dict[str, tuple[str, ...]] = {
    m.NAME: tuple(m.OBJECTIVES) for m, _ in _MODULES if m.OBJECTIVES}

__all__ = ["Strategy", "STRATEGIES", "STRATEGY_DEFAULTS", "STRATEGY_OBJECTIVES",
           "nsga2", "one_plus_lambda", "poss"]
