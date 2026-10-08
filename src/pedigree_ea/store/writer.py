"""Writing one run into a database file (see schema.py).

The Engine calls `candidate` for pedigrees it wants to refer to, `appear`
for a candidate's first appearance in this run, and `generation` once per
generation with the population; `finish` closes the run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..genetics import batch
from ..genetics.canonical import canonical_form, decode_graph
from ..genetics.pedigree import Pedigree
from .codec import (code_version, config_digest, config_label, digest, pack_floats, pack_members,
                    search_config, target_payload)
from .schema import connect


def _get_or_insert(conn, table: str, key: dict, extra: dict | None = None) -> int:
    where = " AND ".join(f"{k} IS ?" for k in key)
    row = conn.execute(f"SELECT id FROM {table} WHERE {where}", tuple(key.values())).fetchone()
    if row:
        return row[0]
    cols = {**key, **(extra or {})}
    cur = conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                       tuple(cols.values()))
    return cur.lastrowid


class RunWriter:
    def __init__(self, path: str | Path, *, name: str, config: dict, seed: int,
                 target, observed_ids: Sequence[str], objectives: Sequence[str],
                 references: Mapping[str, Pedigree] | None = None, grid: str | None = None,
                 stamp: str | None = None, case_name: str | None = None,
                 keyframe_every: int = 50, scope: str = "population"):
        if scope not in ("population", "all"):
            raise ValueError("scope must be 'population' (population members and fits) or 'all'")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = connect(self.path)
        self.ids = list(observed_ids)
        self.keyframe_every = keyframe_every
        self.scope = scope
        c = self.conn
        self.people_set_id = _get_or_insert(c, "people_sets", {"ids": json.dumps(self.ids)})
        kind, payload = target_payload(target)
        target_id = _get_or_insert(c, "targets", {"digest": digest([kind, payload])},
                                   {"people_set_id": self.people_set_id, "kind": kind,
                                    "payload": json.dumps(payload)})
        sc = search_config(config)
        config_id = _get_or_insert(c, "configs", {"digest": config_digest(config)},
                                   {"label": config_label(config), "config": json.dumps(sc)})
        commit, dirty = code_version()
        code_id = _get_or_insert(c, "code_versions", {"commit_id": commit, "dirty": dirty})
        cur = c.execute(
            "INSERT INTO runs (name, grid, stamp, case_name, people_set_id, target_id, config_id, seed,"
            " code_version_id, objectives, keyframe_every, scope, started) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (name, grid, stamp, case_name, self.people_set_id, target_id, config_id, seed, code_id,
             json.dumps(list(objectives)), keyframe_every, scope, time.time()))
        self.run_id = cur.lastrowid
        self._cids: dict[bytes, int] = {}
        self._appeared: dict[int, tuple[bool, bool]] = {}   # cid -> (in_population, fit recorded)
        self._members: dict[int, int] = {}
        for label, ped in (references or {}).items():
            cid = self.candidate_pedigree(ped)
            ref_id = _get_or_insert(c, "refs", {"people_set_id": self.people_set_id, "label": label,
                                                "candidate_id": cid})
            c.execute("INSERT OR IGNORE INTO run_refs VALUES (?, ?)", (self.run_id, ref_id))
        c.commit()

    # ---- candidates -----------------------------------------------------------
    def _insert_canon(self, canon: bytes, valid: bool) -> int:
        n_lat = decode_graph(canon)[1]
        return _get_or_insert(self.conn, "candidates",
                              {"people_set_id": self.people_set_id, "canon": canon},
                              {"n_latent": n_lat, "valid": int(valid)})

    def candidate(self, parents: np.ndarray) -> int:
        """Candidate id of a (normalized) parent array; cached by the array."""
        key = batch.key(parents)
        cid = self._cids.get(key)
        if cid is None:
            ped = batch.to_pedigree(parents, self.ids, prune=False)
            valid = not bool(batch.has_cycle(parents[None])[0])
            cid = self._cids[key] = self._insert_canon(canonical_form(ped, self.ids), valid)
        return cid

    def candidate_pedigree(self, ped: Pedigree) -> int:
        return self._insert_canon(canonical_form(ped, self.ids), ped.is_valid(check_sex=False))

    # ---- per run ------------------------------------------------------------
    def appear(self, cid: int, *, first_eval: int, first_gen: int, operator: str | None,
               parents: Sequence[int] = (), objectives=None, in_population: bool = False,
               fit_at_eval: int | None = None) -> None:
        """Record a candidate's first appearance; later calls only add
        'entered the population' or 'became a fit'."""
        seen = self._appeared.get(cid)
        if seen is None:
            self.conn.execute(
                "INSERT INTO appearances VALUES (?,?,?,?,?,?,?,?,?)",
                (self.run_id, cid, first_eval, first_gen, operator, json.dumps(list(parents)),
                 None if objectives is None else pack_floats(objectives), int(in_population),
                 fit_at_eval))
            self._appeared[cid] = (in_population, fit_at_eval is not None)
            return
        in_pop, fit = seen
        if in_population and not in_pop:
            self.conn.execute("UPDATE appearances SET in_population = 1 WHERE run_id = ? AND candidate_id = ?",
                              (self.run_id, cid))
        if fit_at_eval is not None and not fit:
            self.conn.execute("UPDATE appearances SET fit_at_eval = ? WHERE run_id = ? AND candidate_id = ?",
                              (fit_at_eval, self.run_id, cid))
        self._appeared[cid] = (in_pop or in_population, fit or fit_at_eval is not None)

    def has_appeared(self, cid: int) -> bool:
        return cid in self._appeared

    def generation(self, gen: int, members: dict[int, int], *, evals: int, seconds: float,
                   n_phenotypes: int, n_fits: int, n_pareto: int) -> None:
        c = self.conn
        c.execute("INSERT INTO generations VALUES (?,?,?,?,?,?,?,?)",
                  (self.run_id, gen, evals, seconds, sum(members.values()), n_phenotypes, n_fits, n_pareto))
        deltas = []
        for cid in set(members) | set(self._members):
            d = members.get(cid, 0) - self._members.get(cid, 0)
            if d:
                deltas.append((self.run_id, gen, cid, d))
        c.executemany("INSERT INTO membership VALUES (?,?,?,?)", deltas)
        if gen == 1 or gen % self.keyframe_every == 0:
            c.execute("INSERT INTO keyframes VALUES (?,?,?)", (self.run_id, gen, pack_members(members)))
        self._members = dict(members)
        if gen % 50 == 0:
            c.commit()

    def finish(self, summary: dict | None = None, score: dict | None = None) -> int:
        self.conn.execute("UPDATE runs SET finished = ?, summary = ?, score = ? WHERE id = ?",
                          (time.time(), json.dumps(summary, default=str) if summary else None,
                           json.dumps(score, default=str) if score else None, self.run_id))
        self.conn.commit()
        self.conn.close()
        return self.run_id


def set_score(path: str | Path, run_id: int, score: dict) -> None:
    conn = connect(path, create=False)
    conn.execute("UPDATE runs SET score = ? WHERE id = ?", (json.dumps(score, default=str), run_id))
    conn.commit()
    conn.close()
