"""How close two pedigrees over the same observed people are.

structure_distance(a, b)
    Fewest parent->child links to add or remove to turn `a` into `b`. Observed
    people match by id; latent people are matched to minimize the count (a
    latent person may also stay unmatched). Exact, by trying every matching,
    when both have at most `exact_limit` latent people; otherwise a greedy
    matching improved by swaps, which gives an upper bound. Both pedigrees are
    pruned first. Roughly: how many edits `a` is from `b`.

relationship_distance(a, b, ids)
    Largest difference in expected IBD0/1/2 over pairs of observed people.
    0 means pairwise data cannot tell the two apart.
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np

from .ibd import expected_ibd
from .pedigree import Pedigree


def _edges(ped: Pedigree) -> set[tuple[str, str]]:
    return {(p, c) for c in ped for p in ped[c].parents}


def _cost(edges_a, edges_b, mapping: dict[str, str]) -> int:
    mapped = {(mapping.get(p, p), mapping.get(c, c)) for p, c in edges_a}
    return len(mapped ^ edges_b)


def _observed_neighbours(ped: Pedigree, pid: str, children: dict[str, list[str]]) -> set[str]:
    return ({f"p:{p}" for p in ped[pid].parents if ped[p].observed}
            | {f"c:{c}" for c in children[pid] if ped[c].observed})


def structure_distance(a: Pedigree, b: Pedigree, exact_limit: int = 4) -> int:
    a, b = a.pruned(), b.pruned()
    la, lb = a.latent_ids, b.latent_ids
    ea, eb = _edges(a), _edges(b)
    fresh = [f"\0unmatched{i}" for i in range(len(la))]
    targets = lb + fresh
    if len(la) <= exact_limit and len(lb) <= exact_limit:
        return min(_cost(ea, eb, dict(zip(la, perm))) for perm in permutations(targets, len(la)))

    # Greedy start: pair latent people sharing the most observed neighbours.
    ca, cb = a.children_map(), b.children_map()
    na = {x: _observed_neighbours(a, x, ca) for x in la}
    nb = {y: _observed_neighbours(b, y, cb) for y in lb}
    mapping: dict[str, str] = {}
    free = list(targets)
    for x in sorted(la, key=lambda x: -len(na[x])):
        y = max(free, key=lambda y: len(na[x] & nb[y]) if y in nb else -1)
        mapping[x] = y
        free.remove(y)
    best = _cost(ea, eb, mapping)
    improved = True
    while improved:          # swap targets (including unused ones) while it helps
        improved = False
        for i, x in enumerate(la):
            for other in la[i + 1:] + free:
                trial = dict(mapping)
                if other in mapping:
                    trial[x], trial[other] = mapping[other], mapping[x]
                else:
                    trial[x] = other
                c = _cost(ea, eb, trial)
                if c < best:
                    best, mapping, improved = c, trial, True
                    free = [t for t in targets if t not in mapping.values()]
                    break
            if improved:
                break
    return best


def relationship_distance(a: Pedigree, b: Pedigree, ids: Sequence[str]) -> float:
    ia, ib = expected_ibd(a, ids), expected_ibd(b, ids)
    return float(max((np.abs(np.subtract(ia[p], ib[p])).max() for p in ia), default=0.0))
