"""Identify pedigrees that are the same structure.

Observed people keep their ids; latent people are interchangeable, so two
pedigrees are the same if a renaming of latent people maps one onto the other.
Sex is ignored. Pedigrees are pruned first (see `Pedigree.pruned`), so latent
people who cannot affect observed kinship do not make two solutions differ.

`structure_hash` + `same_structure` / `PedigreeSet`: fast in-memory comparison.
`canonical_graph` / `canonical_form`: exact canonical bytes of any directed graph with
labelled observed and interchangeable latent nodes (cycles and weights allowed); the
database identity of a pedigree.
"""

from __future__ import annotations

import warnings
from itertools import permutations, product
from math import factorial
from typing import Iterator, Sequence

import networkx as nx
import numpy as np

from .pedigree import Pedigree

LATENT_LABEL = "?"


def to_graph(ped: Pedigree) -> nx.DiGraph:
    """Directed graph parent -> child; node attribute `label` is the id, or '?' if latent."""
    g = nx.DiGraph()
    for pid in ped:
        g.add_node(pid, label=pid if ped[pid].observed else LATENT_LABEL)
    for pid in ped:
        for p in ped[pid].parents:
            g.add_edge(p, pid)
    return g


def _label_match(a: dict, b: dict) -> bool:
    return a["label"] == b["label"]


def structure_hash(ped: Pedigree) -> str:
    """Hash equal for identical structures. Different structures can rarely
    collide, so confirm with `same_structure`."""
    # networkx >= 3.5 uses both in- and out-edges for directed graphs and warns
    # that hashes differ from older versions; irrelevant for in-memory use.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return nx.weisfeiler_lehman_graph_hash(to_graph(ped), node_attr="label", iterations=4)


def same_structure(a: Pedigree, b: Pedigree) -> bool:
    return nx.is_isomorphic(to_graph(a), to_graph(b), node_match=_label_match)


class PedigreeSet:
    """A collection of distinct pedigree structures (e.g. an archive of solutions)."""

    def __init__(self, prune: bool = True):
        self.prune = prune
        self._buckets: dict[str, list[Pedigree]] = {}
        self._count = 0

    def _prepare(self, ped: Pedigree) -> Pedigree:
        return ped.pruned() if self.prune else ped

    def _find(self, ped: Pedigree) -> tuple[str, Pedigree | None]:
        h = structure_hash(ped)
        for other in self._buckets.get(h, ()):
            if same_structure(ped, other):
                return h, other
        return h, None

    def add(self, ped: Pedigree) -> bool:
        """Add a pedigree; return True if it is a new structure."""
        ped = self._prepare(ped)
        h, existing = self._find(ped)
        if existing is not None:
            return False
        self._buckets.setdefault(h, []).append(ped)
        self._count += 1
        return True

    def __contains__(self, ped: object) -> bool:
        return isinstance(ped, Pedigree) and self._find(self._prepare(ped))[1] is not None

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[Pedigree]:
        for bucket in self._buckets.values():
            yield from bucket


# ---- exact canonical form (database identity) --------------------------------
#
# Works for any directed graph whose observed nodes are labelled and whose
# latent nodes are interchangeable, with optional edge weights: pedigrees (at
# most two incoming edges, no weights) are the special case. Format:
#     byte 0      number of observed nodes
#     byte 1      number of latent nodes
#     byte 2      flags (bit 0: weighted)
#     then        edges as (source slot, target slot) uint8 pairs, sorted
#     then        float32 weights in edge order (if weighted)
# Slots: observed nodes in the given order, then latent nodes in canonical order.

_MAX_ORDERINGS = 40320   # beyond this, ties are ordered without full search (still deterministic)
_WEIGHTED = 1


def canonical_graph(observed: Sequence[str], latent: Sequence[str],
                    edges: Sequence[tuple[str, str]],
                    weights: Sequence[float] | None = None) -> bytes:
    """Canonical bytes of a directed graph: equal iff the graphs are the same up
    to renaming latent nodes. Cycles are allowed."""
    obs, lat = list(observed), list(latent)
    if len(obs) > 255 or len(lat) > 255:
        raise ValueError("canonical_graph supports at most 255 observed and 255 latent nodes")
    w = {e: float(x) for e, x in zip(edges, weights)} if weights is not None else None
    nodes = obs + lat
    ins: dict[str, list[str]] = {x: [] for x in nodes}
    outs: dict[str, list[str]] = {x: [] for x in nodes}
    for a, b in edges:
        outs[a].append(b)
        ins[b].append(a)
    fixed = {x: i for i, x in enumerate(obs)}

    def base(x):
        return (0, fixed[x]) if x in fixed else (1, -1)

    color = {x: base(x) for x in nodes}
    n_classes = len(set(color.values()))
    for _ in range(len(nodes) + 1):
        def wt(a, b):
            return w[(a, b)] if w is not None else 0.0
        sig = {x: (color[x],
                   tuple(sorted((color[q], wt(q, x)) for q in ins[x])),
                   tuple(sorted((color[c], wt(x, c)) for c in outs[x]))) for x in nodes}
        ranks = {s: i for i, s in enumerate(sorted(set(sig.values()), key=repr))}
        color = {x: base(x) if x in fixed else (1, ranks[sig[x]]) for x in nodes}
        if len(set(color.values())) == n_classes:
            break
        n_classes = len(set(color.values()))
    groups: dict[tuple, list[str]] = {}
    for x in sorted(lat, key=lambda x: color[x]):
        groups.setdefault(color[x], []).append(x)
    group_list = [groups[c] for c in sorted(groups)]

    def encode(order: list[str]) -> bytes:
        slot = {x: i for i, x in enumerate(obs + order)}
        coded = sorted((slot[a], slot[b], (a, b)) for a, b in edges)
        head = bytes([len(obs), len(order), _WEIGHTED if w is not None else 0])
        body = bytes(v for s, t, _ in coded for v in (s, t))
        if w is not None:
            body += np.array([w[e] for _, _, e in coded], dtype=np.float32).tobytes()
        return head + body

    n_orderings = 1
    for g in group_list:
        n_orderings *= factorial(len(g))
    if n_orderings > _MAX_ORDERINGS:
        return encode([x for g in group_list for x in g])
    return min(encode([x for perm in perms for x in perm])
               for perms in product(*(permutations(g) for g in group_list)))


def decode_graph(canon: bytes) -> tuple[int, int, list[tuple[int, int]], list[float] | None]:
    """(n_observed, n_latent, edges as slot pairs, weights or None)."""
    n_obs, n_lat, flags = canon[0], canon[1], canon[2]
    rest = canon[3:]
    if flags & _WEIGHTED:
        n_edges = len(rest) // 6
        pairs, weights = rest[:2 * n_edges], np.frombuffer(rest[2 * n_edges:], dtype=np.float32).tolist()
    else:
        pairs, weights = rest, None
    edges = [(pairs[i], pairs[i + 1]) for i in range(0, len(pairs), 2)]
    return n_obs, n_lat, edges, weights


def canonical_form(ped: Pedigree, observed_ids: Sequence[str]) -> bytes:
    """Canonical bytes of a pedigree (pruned first): equal iff same structure.
    Cyclic (invalid) pedigrees are encoded too."""
    p = ped.pruned()
    edges = [(q, x) for x in p for q in p[x].parents]
    return canonical_graph(observed_ids, p.latent_ids, edges)


def canonical_parents(canon: bytes, n_slots: int | None = None) -> np.ndarray:
    """(N, 2) int16 parent array of a pedigree's canonical form (-1 = missing),
    padded with empty slots to `n_slots`."""
    n_obs, n_lat, edges, _ = decode_graph(canon)
    n = max(n_obs + n_lat, n_slots or 0)
    arr = np.full((n, 2), -1, dtype=np.int16)
    fill = np.zeros(n, dtype=int)
    for s, t in edges:
        if fill[t] >= 2:
            raise ValueError("not a pedigree: a node has more than two parents")
        arr[t, fill[t]] = s
        fill[t] += 1
    return arr


def from_canonical(canon: bytes, observed_ids: Sequence[str]) -> Pedigree:
    """Pedigree from canonical bytes; latent people are named L1, L2, ..."""
    n_obs, n_lat, edges, _ = decode_graph(canon)
    if n_obs != len(observed_ids):
        raise ValueError(f"canonical form has {n_obs} observed nodes, got {len(observed_ids)} ids")
    names = list(observed_ids) + [f"L{k + 1}" for k in range(n_lat)]
    parents: dict[int, list[int]] = {}
    for s, t in edges:
        parents.setdefault(t, []).append(s)
    ped = Pedigree()
    for i, pid in enumerate(names):
        ped.add(pid, [names[q] for q in parents.get(i, [])], observed=i < n_obs)
    return ped
