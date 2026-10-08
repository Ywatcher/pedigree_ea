"""Identify pedigrees that are the same structure.

Observed people keep their ids; latent people are interchangeable, so two
pedigrees are the same if a renaming of latent people maps one onto the other.
Sex is ignored. Pedigrees are pruned first (see `Pedigree.pruned`), so latent
people who cannot affect observed kinship do not make two solutions differ.
"""

from __future__ import annotations

import warnings
from typing import Iterator

import networkx as nx

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
