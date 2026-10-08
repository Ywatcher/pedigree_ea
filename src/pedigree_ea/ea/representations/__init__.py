"""Genotype representations. Each decodes to the shared (N, 2) parent array."""

from .base import Representation
from .cgp import CGP
from .cyclic import Cyclic
from .direct import Direct
from .neat import Neat
from .treegp import TreeGP

REPRESENTATIONS: dict[str, type[Representation]] = {
    "direct": Direct,
    "neat": Neat,
    "cgp": CGP,
    "cyclic": Cyclic,    # not implemented: design open
    "treegp": TreeGP,    # not implemented: design open
}
IMPLEMENTED = ("direct", "neat", "cgp")

__all__ = ["Representation", "REPRESENTATIONS", "IMPLEMENTED",
           "Direct", "Neat", "CGP", "Cyclic", "TreeGP"]
