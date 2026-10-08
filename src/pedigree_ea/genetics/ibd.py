"""Expected IBD sharing (IBD0 / IBD1 / IBD2) between pairs of people.

For a pair (i, j), IBDk is the probability (= expected genome fraction) that
i and j share k alleles identical by descent at a locus. Reference values
(k0, k1, k2), no inbreeding:
    parent-child          (0,    1,   0)
    full siblings         (1/4,  1/2, 1/4)
    half sib / grandparent / avuncular  (1/2, 1/2, 0)
    double first cousins  (9/16, 6/16, 1/16)
    first cousins         (3/4,  1/4, 0)

Computed exactly via Jacquard's 9 condensed identity states, so pedigrees with
inbreeding are handled too. A state is collapsed to "number of IBD alleles
shared" as the largest one-to-one matching between i's two genes and j's two
genes (the usual mapping: states 1,7 -> IBD2; 3,5,8 -> IBD1; 2,4,6,9 -> IBD0).

Method: backward gene dropping. Track the four genes (i's two, j's two).
Repeatedly take the person latest in topological order who carries a tracked
gene, and replace each of their genes by one of their parent's two genes
(probability 1/2 each). A gene from a missing parent is a unique founder gene.
When only founder genes remain, which tracked genes coincide gives the state.

Monotonicity: adding a parent-child edge can only merge founder genes, so IBD0
never increases and IBD2 never decreases (used for pruning in reference/brute_force.py).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any, NamedTuple, Sequence

import numpy as np

from .kinship import _topo_indices
from .pairs import Pair, PairData
from .pedigree import Pedigree, PedigreeError


class IBD(NamedTuple):
    k0: float
    k1: float
    k2: float

    @classmethod
    def from_k1_k2(cls, k1: float, k2: float) -> IBD:
        return cls(1.0 - k1 - k2, k1, k2)

    @property
    def kinship(self) -> float:
        """k1/4 + k2/2; equals the kinship coefficient when neither person is inbred."""
        return self.k1 / 4 + self.k2 / 2

    @property
    def pi_hat(self) -> float:
        """Proportion of the genome IBD (PLINK's PI_HAT) = k1/2 + k2."""
        return self.k1 / 2 + self.k2


class IBDData(PairData[IBD]):
    """IBD0/IBD1/IBD2 for pairs of people. Values may be IBD or (k0, k1, k2)."""

    @staticmethod
    def _convert(v: Any) -> IBD:
        k0, k1, k2 = v
        return IBD(float(k0), float(k1), float(k2))

    def kinship(self):
        """Kinship implied by the IBD values (exact only without inbreeding)."""
        from .kinship import KinshipData
        return KinshipData({pair: v.kinship for pair, v in self.items()}, self.ids)


# ---- Jacquard identity states -----------------------------------------------

def _jacquard_state(a: int, b: int, c: int, d: int) -> int:
    """State index 0..8 (Jacquard's Delta1..Delta9); a, b = i's genes, c, d = j's."""
    if a == b:
        if c == d:
            return 0 if a == c else 1
        return 2 if a in (c, d) else 3
    if c == d:
        return 4 if c in (a, b) else 5
    if (a == c and b == d) or (a == d and b == c):
        return 6
    return 7 if (a in (c, d) or b in (c, d)) else 8


# IBD alleles shared and kinship contribution for each Jacquard state.
_SHARED = np.array([2, 0, 1, 0, 1, 0, 2, 1, 0])
_KINSHIP = np.array([1, 0, 0.5, 0, 0.5, 0, 0.5, 0.25, 0])


def _canon(state: tuple[int, ...]) -> tuple[int, ...]:
    labels: dict[int, int] = {}
    return tuple(labels.setdefault(g, len(labels)) for g in state)


class _GeneDropper:
    """Identity-state distributions for gene tuples in one pedigree (memoized)."""

    def __init__(self, parents: Sequence[Sequence[int]]):
        self.parents = [tuple(ps) for ps in parents]
        self.pos = {x: k for k, x in enumerate(_topo_indices(self.parents))}
        self.memo: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}

    def dist(self, state: tuple[int, ...]) -> dict[tuple[int, ...], float]:
        """Distribution of the coincidence pattern of genes in `state`.

        Gene 2*x + s is person x's gene inherited from parents[x][s]; if that
        parent is missing, it is a founder gene. A pattern labels each position
        of `state` so that equal labels mean identical by descent.
        """
        genes = tuple(sorted(set(state)))
        sub = self._dist_distinct(genes)
        if genes == state:
            return sub
        where = {g: k for k, g in enumerate(genes)}
        out: dict[tuple[int, ...], float] = defaultdict(float)
        for pattern, pr in sub.items():
            out[_canon(tuple(pattern[where[g]] for g in state))] += pr
        return out

    def _dist_distinct(self, genes: tuple[int, ...]) -> dict[tuple[int, ...], float]:
        """`dist` for a sorted tuple of distinct genes (memoized)."""
        hit = self.memo.get(genes)
        if hit is not None:
            return hit
        x, best = None, -1
        for g in genes:
            p, s = divmod(g, 2)
            if s < len(self.parents[p]) and self.pos[p] > best:
                x, best = p, self.pos[p]
        if x is None:
            out = {tuple(range(len(genes))): 1.0}
        else:
            slots = [g & 1 for g in genes if g >> 1 == x and (g & 1) < len(self.parents[x])]
            weight = 0.5 ** len(slots)
            acc: dict[tuple[int, ...], float] = defaultdict(float)
            for choice in product((0, 1), repeat=len(slots)):
                swap = {2 * x + s: 2 * self.parents[x][s] + c for s, c in zip(slots, choice)}
                for pattern, pr in self.dist(tuple(swap.get(g, g) for g in genes)).items():
                    acc[pattern] += pr * weight
            out = dict(acc)
        self.memo[genes] = out
        return out

    def jacquard(self, i: int, j: int) -> np.ndarray:
        delta = np.zeros(9)
        for pattern, pr in self.dist((2 * i, 2 * i + 1, 2 * j, 2 * j + 1)).items():
            delta[_jacquard_state(*pattern)] += pr
        return delta


class _IdentityProb:
    """P(each group of genes is identical by descent), memoized; scalar version
    of `_GeneDropper` for the few events IBD0/1/2 need."""

    def __init__(self, parents: Sequence[Sequence[int]]):
        self.parents = [tuple(ps) for ps in parents]
        self.pos = {x: k for k, x in enumerate(_topo_indices(self.parents))}
        self.memo: dict[tuple[tuple[int, ...], ...], float] = {}

    def _terminal(self, g: int) -> bool:
        return (g & 1) >= len(self.parents[g >> 1])

    @staticmethod
    def _normalize(groups) -> tuple[tuple[int, ...], ...]:
        """Merge groups sharing a gene, drop duplicates and singletons."""
        merged: list[set[int]] = []
        for grp in groups:
            grp = set(grp)
            keep = []
            for m in merged:
                if m & grp:
                    grp |= m
                else:
                    keep.append(m)
            merged = keep + [grp]
        return tuple(sorted(tuple(sorted(m)) for m in merged if len(m) > 1))

    def prob(self, groups) -> float:
        return self._prob(self._normalize(groups))

    def _prob(self, groups: tuple[tuple[int, ...], ...]) -> float:
        if not groups:
            return 1.0
        hit = self.memo.get(groups)
        if hit is not None:
            return hit
        x, best = None, -1
        zero = False
        for grp in groups:
            if sum(self._terminal(g) for g in grp) > 1:  # distinct founder genes
                zero = True
                break
            for g in grp:
                p = g >> 1
                if not self._terminal(g) and self.pos[p] > best:
                    x, best = p, self.pos[p]
        if zero or x is None:
            out = 0.0
        else:
            slots = sorted({g & 1 for grp in groups for g in grp
                            if g >> 1 == x and not self._terminal(g)})
            weight = 0.5 ** len(slots)
            out = 0.0
            for choice in product((0, 1), repeat=len(slots)):
                swap = {2 * x + s: 2 * self.parents[x][s] + c for s, c in zip(slots, choice)}
                new = self._normalize([[swap.get(g, g) for g in grp] for grp in groups])
                out += weight * self._prob(new)
        self.memo[groups] = out
        return out


def ibd_from_jacquard(delta: np.ndarray) -> IBD:
    k = np.bincount(_SHARED, weights=delta, minlength=3)
    return IBD(float(k[0]), float(k[1]), float(k[2]))


def kinship_from_jacquard(delta: np.ndarray) -> float:
    return float(_KINSHIP @ delta)


def jacquard_from_parents(parents: Sequence[Sequence[int]],
                          pairs: Sequence[tuple[int, int]]) -> list[np.ndarray]:
    """Jacquard coefficients (Delta1..Delta9) for index pairs; parents[i] lists i's parents."""
    dropper = _GeneDropper(parents)
    return [dropper.jacquard(i, j) for i, j in pairs]


def ibd_from_parents(parents: Sequence[Sequence[int]], pairs: Sequence[tuple[int, int]],
                     phi: np.ndarray | None = None) -> list[IBD]:
    """IBD0/1/2 for index pairs; parents[i] lists i's parents.

    `phi` is the kinship matrix if already computed. With i's genes a, b and
    j's genes c, d (Delta numbered 1..9 as Jacquard):
        k2 = P(a=c, b=d) + P(a=d, b=c) - Delta1
        Delta3 = P(a=b=c) + P(a=b=d) - 2 Delta1,  Delta5 likewise
        Delta8 = 4 (phi - Delta1 - (Delta3 + Delta5 + Delta7) / 2)
        k1 = Delta3 + Delta5 + Delta8
    Terms involving a=b vanish unless i is inbred (same for j).
    """
    if phi is None:
        from .kinship import kinship_from_parents
        phi = kinship_from_parents(parents)
    ip = _IdentityProb(parents)
    out = []
    for i, j in pairs:
        f = phi[i, j]
        if f <= 0:
            out.append(IBD(1.0, 0.0, 0.0))
            continue
        a, b, c, d = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
        inbred_i, inbred_j = phi[i, i] > 0.5, phi[j, j] > 0.5
        d1 = ip.prob([(a, b, c, d)]) if inbred_i and inbred_j else 0.0
        d7 = ip.prob([(a, c), (b, d)]) + ip.prob([(a, d), (b, c)]) - 2 * d1
        d3 = ip.prob([(a, b, c)]) + ip.prob([(a, b, d)]) - 2 * d1 if inbred_i else 0.0
        d5 = ip.prob([(c, d, a)]) + ip.prob([(c, d, b)]) - 2 * d1 if inbred_j else 0.0
        d8 = 4 * (f - d1 - (d3 + d5 + d7) / 2)
        k2 = d1 + d7
        k1 = d3 + d5 + d8
        out.append(IBD(1.0 - k1 - k2, k1, k2))
    return out


def _parent_indices(ped: Pedigree) -> tuple[dict[str, int], list[list[int]]]:
    idx = {pid: i for i, pid in enumerate(ped.ids)}
    try:
        return idx, [[idx[p] for p in ped[pid].parents] for pid in ped.ids]
    except KeyError as e:
        raise PedigreeError(f"unknown parent {e.args[0]!r}") from None


def jacquard(ped: Pedigree, a: str, b: str) -> np.ndarray:
    """Jacquard's 9 condensed identity coefficients for people `a` and `b`."""
    idx, parents = _parent_indices(ped)
    return jacquard_from_parents(parents, [(idx[a], idx[b])])[0]


def expected_ibd(ped: Pedigree, ids: Sequence[str] | None = None) -> IBDData:
    """Expected IBD0/1/2 for every pair among `ids` (default: observed people)."""
    ids = list(ped.observed_ids if ids is None else ids)
    idx, parents = _parent_indices(ped)
    pairs = [(a, b) for k, a in enumerate(ids) for b in ids[k + 1:]]
    values = ibd_from_parents(parents, [(idx[a], idx[b]) for a, b in pairs])
    return IBDData(dict(zip(pairs, values)), ids)


def ibd_from_kinship_ibs0(kinship: float, ibs0: float, unrelated_ibs0: float) -> IBD:
    """Rough IBD0/1/2 from a KING kinship and IBS0 (no IBD segments needed).

    Opposite homozygotes (IBS0) can only occur where a pair shares no IBD
    allele, so IBS0 ~ IBD0 * (IBS0 of unrelated pairs), giving
    k0 = ibs0 / unrelated_ibs0. With kinship = k1/4 + k2/2 and k0 + k1 + k2 = 1,
    k2 = 4 * kinship - 1 + k0. Values are clipped to [0, 1].

    Approximate: assumes one unrelated baseline for all pairs (allele
    frequencies differ by pair), no inbreeding, and is sensitive to genotyping
    error and LD. Prefer real IBD estimates (KING --ibdseg, PLINK --genome).
    """
    k0 = min(max(ibs0 / unrelated_ibs0, 0.0), 1.0)
    k2 = min(max(4 * kinship - 1 + k0, 0.0), 1.0 - k0)
    return IBD(k0, 1.0 - k0 - k2, k2)


def ibd_errors(expected: IBDData, target: IBDData) -> dict[Pair, np.ndarray]:
    """Signed error (expected - target) as an array [dk0, dk1, dk2] per target pair."""
    return {pair: np.subtract(expected[pair], v) for pair, v in target.items()}
