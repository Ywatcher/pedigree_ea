"""What snapshots record each generation. Only the recorder sees the
references; the search never does.

Records per generation (the current population is the pool):
    best                  the best candidate (lexicographic on the objectives)
    closest:<label>       for each reference pedigree, the `closest_k` candidates
                          most similar to it, ranked by structure distance then
                          relationship distance (`rank_by="structure"`), or the
                          other way round (`rank_by="relationship"`)

Each recorded candidate: pedigree (pruned), objectives (invalid candidates
carry the penalty), worst per-pair error against the target, whether it fits,
and its distances to every reference (see genetics/similarity.py).

Removal events: the closest candidate to each reference is tracked from one
generation to the next (ties keep the one tracked last time). When it is no
longer in the population, an event records the removed candidate (as last
recorded), the candidate now closest ("replaced_by"), and the distance before
and after. `last_removed[label]` keeps the most recent event per reference.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..genetics import batch
from ..genetics.canonical import same_structure, structure_hash
from ..genetics.ibd import expected_ibd
from ..genetics.pedigree import Pedigree
from ..genetics.similarity import structure_distance
from .problem import Problem


class Recording:
    def __init__(self, problem: Problem, references: Mapping[str, Pedigree] | None = None,
                 closest_k: int = 1, rank_by: str = "structure", exact_limit: int = 4):
        if rank_by not in ("structure", "relationship"):
            raise ValueError(f"rank_by must be 'structure' or 'relationship', got {rank_by!r}")
        self.problem = problem
        self.references = {label: ref.pruned() for label, ref in (references or {}).items()}
        self.closest_k = closest_k
        self.rank_by = rank_by
        self.exact_limit = exact_limit
        # Relationship distance = worst per-pair IBD difference to the reference,
        # computed in batch by a Problem whose target is the reference's expected IBD.
        self._rel = {label: Problem(expected_ibd(ref, problem.ids), max_latent=problem.max_latent,
                                    tol=0.0, objectives=("ibd_worst",), check_sex=False)
                     for label, ref in self.references.items()}
        self._struct: dict[tuple[str, bytes], int] = {}
        self._peds: dict[bytes, Pedigree] = {}
        self._hashes: dict[bytes, str] = {}
        self._tracked: dict[str, dict] = {}          # label -> closest item last generation
        self.last_removed: dict[str, dict | None] = {label: None for label in self.references}

    def _hash(self, key: bytes, ped: Pedigree) -> str:
        h = self._hashes.get(key)
        if h is None:
            h = self._hashes[key] = structure_hash(ped)
        return h

    def _pedigree(self, key: bytes, parents: np.ndarray) -> Pedigree:
        ped = self._peds.get(key)
        if ped is None:
            ped = self._peds[key] = batch.to_pedigree(parents, self.problem.ids)
        return ped

    def record(self, objs: np.ndarray, parents: np.ndarray, generation: int = 0) -> dict:
        """`parents` (B, N, 2) are the decoded rows of `objs`. Returns
        {"records": {name: [item]}, "events": [removal events this generation],
         "last_removed": {label: latest event or None}}."""
        parents = batch.normalize(parents)
        keys = [batch.key(p) for p in parents]
        # One row per distinct structure (arrays that differ only by latent labels collapse).
        first: dict[bytes, int] = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        uniq, seen = [], {}
        for k, i in first.items():
            ped = self._pedigree(k, parents[i])
            h = self._hash(k, ped)
            if not any(same_structure(ped, other) for other in seen.get(h, [])):
                seen.setdefault(h, []).append(ped)
                uniq.append(i)
        structure = {i: (self._peds[keys[i]], self._hashes[keys[i]]) for i in uniq}
        ev = self.problem.evaluate(parents[uniq])
        info = {keys[i]: (float(ev.pair_error[j].max(initial=0.0)), bool(ev.fits[j]))
                for j, i in enumerate(uniq)}
        dist: dict[bytes, dict[str, dict]] = {keys[i]: {} for i in uniq}
        for label, ref in self.references.items():
            rel = self._rel[label].evaluate(parents[uniq]).objectives[:, 0]
            for j, i in enumerate(uniq):
                k = keys[i]
                s = self._struct.get((label, k))
                if s is None:
                    s = self._struct[(label, k)] = structure_distance(
                        self._pedigree(k, parents[i]), ref, self.exact_limit)
                dist[k][label] = {"structure": int(s), "relationship": round(float(rel[j]), 6)}

        def item(i: int) -> dict:
            k = keys[i]
            worst, fits = info[k]
            return {"pedigree": self._pedigree(k, parents[i]), "objectives": objs[i].tolist(),
                    "worst_error": worst, "fits": fits, "distances": dist[k]}

        records = {"best": [item(int(np.lexsort(objs.T[::-1])[0]))]}
        events = []
        for label in self.references:
            tracked = self._tracked.get(label)
            still_there = None
            if tracked is not None:
                th = structure_hash(tracked["pedigree"])
                still_there = next((i for i in uniq if structure[i][1] == th
                                    and same_structure(structure[i][0], tracked["pedigree"])), None)

            def rank(i, label=label):
                d = dist[keys[i]][label]
                pair = (d["structure"], d["relationship"])
                pair = pair if self.rank_by == "structure" else pair[::-1]
                return (*pair, i != still_there)        # ties keep the tracked candidate
            closest = [item(i) for i in sorted(uniq, key=rank)[:self.closest_k]]
            records[f"closest:{label}"] = closest
            if tracked is not None and still_there is None:
                event = {"reference": label, "generation": generation, "lost": tracked,
                         "replaced_by": closest[0],
                         "distance_before": tracked["distances"][label]["structure"],
                         "distance_after": closest[0]["distances"][label]["structure"]}
                events.append(event)
                self.last_removed[label] = event
            self._tracked[label] = closest[0]
        return {"records": records, "events": events, "last_removed": dict(self.last_removed)}


def _item_for_log(r: dict) -> dict:
    return {**r, "pedigree": str(r["pedigree"])}


def records_for_log(records: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """JSON-friendly copy (pedigree as text)."""
    return {name: [_item_for_log(r) for r in items] for name, items in records.items()}


def events_for_log(events: list[dict]) -> list[dict]:
    return [{**e, "lost": _item_for_log(e["lost"]), "replaced_by": _item_for_log(e["replaced_by"])}
            for e in events]
