"""Pairwise KING-robust measurements: kinship and IBS0.

IBS0 is the fraction of variants where a pair are opposite homozygotes (no
shared allele). That can only happen where they share no IBD allele, so
    IBS0 ~ IBD0 * (IBS0 of unrelated pairs),
and IBS0 / unrelated_ibs0 estimates IBD0: ~0 for parent-child, ~0.25 for full
siblings, ~0.5 for second degree, ~1 for unrelated. Unlike the approximate
IBD0/1/2 (ibd.ibd_from_kinship_ibs0), this target only uses measured values.
"""

from __future__ import annotations

import statistics
from typing import Any, NamedTuple

from .ibd import IBDData, ibd_from_kinship_ibs0
from .kinship import KinshipData
from .pairs import PairData

KING_UNRELATED = 2.0 ** -4.5   # 0.0442: KING's kinship cut-off below 3rd degree


class KingPair(NamedTuple):
    kinship: float
    ibs0: float


class KingData(PairData[KingPair]):
    """Kinship and IBS0 per pair, plus the IBS0 expected for unrelated pairs.

    `unrelated_ibs0` defaults to the median IBS0 of pairs with kinship below
    KING's unrelated cut-off (0.0442); pass it explicitly if there are none.
    """

    def __init__(self, values, ids=None, unrelated_ibs0: float | None = None):
        super().__init__(values, ids)
        if unrelated_ibs0 is None:
            unrelated = [v.ibs0 for v in self._values.values() if v.kinship < KING_UNRELATED]
            if not unrelated:
                raise ValueError("no unrelated pairs to estimate the IBS0 baseline; "
                                 "pass unrelated_ibs0")
            unrelated_ibs0 = float(statistics.median(unrelated))
        self.unrelated_ibs0 = unrelated_ibs0

    @staticmethod
    def _convert(v: Any) -> KingPair:
        kin, ibs0 = v
        return KingPair(float(kin), float(ibs0))

    def ibd0(self, a: str, b: str) -> float:
        """IBD0 estimated from IBS0 (not clipped)."""
        return self[a, b].ibs0 / self.unrelated_ibs0

    def kinship(self, clip_negative: bool = True) -> KinshipData:
        return KinshipData({p: max(v.kinship, 0.0) if clip_negative else v.kinship
                            for p, v in self.items()}, self.ids)

    def approx_ibd(self) -> IBDData:
        """Approximate IBD0/1/2 (see ibd.ibd_from_kinship_ibs0)."""
        return IBDData({p: ibd_from_kinship_ibs0(v.kinship, v.ibs0, self.unrelated_ibs0)
                        for p, v in self.items()}, self.ids)
