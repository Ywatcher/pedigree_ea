"""Merging databases: shared rows are matched by content key and ids remapped.

Used to combine per-run shards (written by parallel workers) into one database
per grid, and works the same way for combining any databases of this schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .codec import pack_members, unpack_members
from .schema import connect
from .writer import _get_or_insert


def merge(sources: Iterable[str | Path], dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = connect(dest)
    for src in sources:
        _merge_one(connect(src, create=False), out)
        out.commit()
    out.close()
    return dest


def _merge_one(src, out) -> None:
    def rows(sql, *args):
        return src.execute(sql, args).fetchall()

    people = {i: _get_or_insert(out, "people_sets", {"ids": ids})
              for i, ids in rows("SELECT id, ids FROM people_sets")}
    cands = {i: _get_or_insert(out, "candidates", {"people_set_id": people[ps], "canon": canon},
                               {"n_latent": nl, "valid": v})
             for i, ps, canon, nl, v in rows("SELECT id, people_set_id, canon, n_latent, valid FROM candidates")}
    targets = {i: _get_or_insert(out, "targets", {"digest": dg},
                                 {"people_set_id": people[ps], "kind": kind, "payload": payload,
                                  "source": source})
               for i, ps, kind, payload, dg, source in
               rows("SELECT id, people_set_id, kind, payload, digest, source FROM targets")}
    configs = {i: _get_or_insert(out, "configs", {"digest": dg}, {"label": label, "config": cfg})
               for i, dg, label, cfg in rows("SELECT id, digest, label, config FROM configs")}
    codes = {i: _get_or_insert(out, "code_versions", {"commit_id": c, "dirty": d})
             for i, c, d in rows("SELECT id, commit_id, dirty FROM code_versions")}
    refs = {i: _get_or_insert(out, "refs", {"people_set_id": people[ps], "label": label,
                                            "candidate_id": cands[cid]})
            for i, ps, label, cid in rows("SELECT id, people_set_id, label, candidate_id FROM refs")}

    cols = ("name, grid, stamp, case_name, people_set_id, target_id, config_id, seed, code_version_id,"
            " objectives, keyframe_every, scope, started, finished, summary, score")
    for r in rows(f"SELECT id, {cols} FROM runs"):
        old_id, vals = r[0], list(r[1:])
        vals[4], vals[5], vals[6], vals[8] = people[vals[4]], targets[vals[5]], configs[vals[6]], codes[vals[8]]
        new_id = out.execute(f"INSERT INTO runs ({cols}) VALUES ({','.join('?' * len(vals))})",
                             vals).lastrowid
        out.executemany("INSERT OR IGNORE INTO run_refs VALUES (?, ?)",
                        [(new_id, refs[ref]) for (ref,) in rows("SELECT ref_id FROM run_refs WHERE run_id = ?", old_id)])
        out.executemany("INSERT INTO generations VALUES (?,?,?,?,?,?,?,?)",
                        [(new_id, *g) for g in rows(
                            "SELECT gen, evals, seconds, population, n_phenotypes, n_fits, n_pareto"
                            " FROM generations WHERE run_id = ?", old_id)])
        out.executemany("INSERT INTO membership VALUES (?,?,?,?)",
                        [(new_id, gen, cands[c], d) for gen, c, d in rows(
                            "SELECT gen, candidate_id, delta FROM membership WHERE run_id = ?", old_id)])
        out.executemany("INSERT INTO keyframes VALUES (?,?,?)",
                        [(new_id, gen, pack_members({cands[c]: n for c, n in unpack_members(blob).items()}))
                         for gen, blob in rows("SELECT gen, members FROM keyframes WHERE run_id = ?", old_id)])
        out.executemany("INSERT INTO appearances VALUES (?,?,?,?,?,?,?,?,?)",
                        [(new_id, cands[c], fe, fg, op, json.dumps([cands[p] for p in json.loads(par)]),
                          obj, inpop, fit) for c, fe, fg, op, par, obj, inpop, fit in rows(
                            "SELECT candidate_id, first_eval, first_gen, operator, parents,"
                            " entry_objectives, in_population, fit_at_eval FROM appearances WHERE run_id = ?",
                            old_id)])
    src.close()
