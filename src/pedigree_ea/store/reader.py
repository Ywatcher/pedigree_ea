"""Reading run records back and deriving everything that is not stored.

    db = Database("results/db/grid_x_<stamp>.sqlite")
    db.runs()                          # list of runs with config, seed, scores
    view = db.run(run_id, extra_refs)  # RunView
    view.members_at(gen)               # {candidate id: count} in the population
    view.card(cid, refs)               # one candidate: objectives, fit, distances, lineage
    view.frame(gen, refs, k)           # best / Pareto / members / closest / removed / latest fit
    view.series(refs)                  # per-generation curves and events (for charts and jumps)
    view.per_pair(cid)                 # observed vs predicted values per pair
    view.snapshots(refs, k)            # same structure as Engine snapshots (for viz.write_gif)

Objectives are recomputed with the current code under the run's settings
(`objectives`); the values a candidate had when it entered the run are kept
as `entry_objectives`. Closest candidates and removal events are replayed
from the stored population, so references can be added after the run.
"""

from __future__ import annotations

import bisect
import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..ea.pareto import nondominated_ranks
from ..ea.problem import Problem
from ..genetics.canonical import canonical_form, canonical_parents, from_canonical
from ..genetics.ibd import IBDData, expected_ibd
from ..genetics.king import KingData
from ..genetics.kinship import KinshipData, expected_kinship, inbreeding
from ..genetics.pedigree import Pedigree
from ..genetics.similarity import structure_distance
from .codec import target_from_payload, unpack_floats, unpack_members
from .schema import connect


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = connect(self.path, create=False)
        self._views: dict[tuple, RunView] = {}

    def runs(self) -> list[dict]:
        out = []
        for (rid, name, grid, stamp, case, seed, objectives, summary, score, label, cfg, kind,
             n_gen) in self.conn.execute(
                "SELECT r.id, r.name, r.grid, r.stamp, r.case_name, r.seed, r.objectives, r.summary,"
                " r.score, c.label, c.config, t.kind,"
                " (SELECT MAX(gen) FROM generations g WHERE g.run_id = r.id)"
                " FROM runs r JOIN configs c ON c.id = r.config_id JOIN targets t ON t.id = r.target_id"
                " ORDER BY r.grid, r.case_name, c.label, r.seed, r.id"):
            out.append({"id": rid, "name": name, "grid": grid, "stamp": stamp, "case": case,
                        "seed": seed, "label": label, "config": json.loads(cfg), "kind": kind,
                        "objectives": json.loads(objectives), "generations": n_gen or 0,
                        "summary": json.loads(summary) if summary else None,
                        "score": json.loads(score) if score else None})
        return out

    def run_references(self, run_id: int, ids: Sequence[str]) -> dict[str, Pedigree]:
        """References stored with this run (e.g. its case's truth)."""
        return {label: from_canonical(canon, ids) for label, canon in self.conn.execute(
            "SELECT r.label, c.canon FROM run_refs rr JOIN refs r ON r.id = rr.ref_id"
            " JOIN candidates c ON c.id = r.candidate_id WHERE rr.run_id = ?", (run_id,))}

    def run(self, run_id: int, extra_refs: Mapping[str, Pedigree] | None = None) -> RunView:
        key = (run_id, tuple(sorted((extra_refs or {}).keys())))
        if key not in self._views:
            self._views[key] = RunView(self, run_id, extra_refs or {})
        return self._views[key]


class RunView:
    def __init__(self, db: Database, run_id: int, extra_refs: Mapping[str, Pedigree]):
        c = db.conn
        self.db, self.run_id = db, run_id
        row = c.execute("SELECT people_set_id, target_id, config_id, seed, objectives, summary, score,"
                        " name, case_name, grid FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no run {run_id}")
        ps, target_id, config_id, self.seed, objectives, summary, score, self.name, self.case, self.grid = row
        self.ids: list[str] = json.loads(c.execute("SELECT ids FROM people_sets WHERE id = ?",
                                                   (ps,)).fetchone()[0])
        kind, payload = c.execute("SELECT kind, payload FROM targets WHERE id = ?", (target_id,)).fetchone()
        self.kind = kind
        self.target = target_from_payload(kind, json.loads(payload))
        label, cfg = c.execute("SELECT label, config FROM configs WHERE id = ?", (config_id,)).fetchone()
        self.label, self.config = label, json.loads(cfg)
        self.objective_names: list[str] = json.loads(objectives)
        self.summary = json.loads(summary) if summary else None
        self.score = json.loads(score) if score else None
        self.problem = Problem(self.target, max_latent=self.config["max_latent"], tol=self.config["tol"],
                               tol_ibd0=self.config.get("tol_ibd0", 0.15),
                               objectives=tuple(self.config["objectives"]),
                               check_sex=self.config.get("check_sex", True))
        self.refs: dict[str, Pedigree] = {**db.run_references(run_id, self.ids), **extra_refs}
        self._ref_problems = {label: Problem(expected_ibd(ref, self.ids), max_latent=self.problem.max_latent,
                                             tol=0.0, objectives=("ibd_worst",), check_sex=False)
                              for label, ref in self.refs.items()}
        gens = c.execute("SELECT gen, evals, seconds, population, n_phenotypes, n_fits, n_pareto"
                         " FROM generations WHERE run_id = ? ORDER BY gen", (run_id,)).fetchall()
        self.generations = [dict(zip(("gen", "evals", "seconds", "population", "n_phenotypes",
                                      "n_fits", "n_pareto"), g)) for g in gens]
        self.n_generations = len(gens)
        self._keyframes = {g: unpack_members(b) for g, b in c.execute(
            "SELECT gen, members FROM keyframes WHERE run_id = ? ORDER BY gen", (run_id,))}
        self._kf_gens = sorted(self._keyframes)
        self._deltas: dict[int, list[tuple[int, int]]] = {}
        for g, cid, d in c.execute("SELECT gen, candidate_id, delta FROM membership WHERE run_id = ?",
                                   (run_id,)):
            self._deltas.setdefault(g, []).append((cid, d))
        self._appear = {}
        for cid, fe, fg, op, par, obj, inpop, fit in c.execute(
                "SELECT candidate_id, first_eval, first_gen, operator, parents, entry_objectives,"
                " in_population, fit_at_eval FROM appearances WHERE run_id = ?", (run_id,)):
            self._appear[cid] = {"first_eval": fe, "first_gen": fg, "operator": op,
                                 "parents": json.loads(par or "[]"),
                                 "entry_objectives": unpack_floats(obj), "in_population": bool(inpop),
                                 "fit_at_eval": fit}
        fits = sorted((a["fit_at_eval"], cid) for cid, a in self._appear.items()
                      if a["fit_at_eval"] is not None)
        self._fit_evals = [e for e, _ in fits]
        self._fit_cids = [c_ for _, c_ in fits]
        self._canon: dict[int, bytes] = {}
        self._eval_cache: dict[int, dict] = {}
        self._dist: dict[tuple[int, str], dict] = {}
        self._members_cache: tuple[int, dict] | None = None
        self._replays: dict[tuple, dict] = {}

    # ---- candidates -------------------------------------------------------------
    def canon(self, cid: int) -> bytes:
        if cid not in self._canon:
            self._canon[cid] = self.db.conn.execute("SELECT canon FROM candidates WHERE id = ?",
                                                    (cid,)).fetchone()[0]
        return self._canon[cid]

    def pedigree(self, cid: int) -> Pedigree:
        return _pedigree(self.canon(cid), tuple(self.ids))

    def _arrays(self, cids: Sequence[int]) -> np.ndarray:
        return np.stack([canonical_parents(self.canon(c), self.problem.n) for c in cids])

    def evaluate(self, cids: Sequence[int]) -> dict[int, dict]:
        """Recomputed objectives, worst per-pair error, fit flag (cached)."""
        todo = [c for c in dict.fromkeys(cids) if c not in self._eval_cache]
        if todo:
            ev = self.problem.evaluate(self._arrays(todo))
            for j, c in enumerate(todo):
                ped = self.pedigree(c)
                self._eval_cache[c] = {
                    "objectives": [float(x) for x in ev.objectives[j]],
                    "worst_error": float(ev.pair_error[j].max(initial=0.0)),
                    "fits": bool(ev.fits[j]), "valid": bool(ev.valid[j]),
                    "n_latent": len(ped.latent_ids),
                    "inbred": ped.is_valid(check_sex=False)
                              and any(f > 1e-12 for f in inbreeding(ped).values())}
        return {c: self._eval_cache[c] for c in cids}

    def distance(self, cid: int, ref: str) -> dict:
        key = (cid, ref)
        if key not in self._dist:
            s = structure_distance(self.pedigree(cid), self.refs[ref])
            r = float(self._ref_problems[ref].evaluate(self._arrays([cid])).objectives[0, 0])
            self._dist[key] = {"structure": int(s), "relationship": round(r, 6)}
        return self._dist[key]

    def appearance(self, cid: int) -> dict | None:
        return self._appear.get(cid)

    def card(self, cid: int, refs: Sequence[str] = (), count: int | None = None) -> dict:
        from .. import viz
        out = {"cid": cid, "layout": viz.layout(self.pedigree(cid), self.ids), "count": count,
               **self.evaluate([cid])[cid],
               "distances": {r: self.distance(cid, r) for r in refs}}
        out.update(self._appear.get(cid) or {})
        return out

    def per_pair(self, cid: int) -> list[dict]:
        """Observed vs predicted values for every target pair."""
        ped = self.pedigree(cid)
        rows = []
        valid = ped.is_valid(check_sex=False)
        ibd = expected_ibd(ped, self.ids) if valid else None
        kin = expected_kinship(ped, self.ids) if valid else None
        for (a, b), v in self.target.items():
            row = {"pair": f"{a}-{b}"}
            if isinstance(self.target, KingData):
                row.update(obs_kinship=v.kinship, obs_ibs0=v.ibs0, obs_ibd0=self.target.ibd0(a, b))
            elif isinstance(self.target, KinshipData):
                row.update(obs_kinship=v)
            else:
                row.update(obs_ibd0=v.k0, obs_ibd1=v.k1, obs_ibd2=v.k2)
            if valid:
                p = ibd[a, b]
                row.update(pred_ibd0=p.k0, pred_ibd1=p.k1, pred_ibd2=p.k2, pred_kinship=kin[a, b])
                if isinstance(self.target, KingData):
                    row["pred_ibs0"] = p.k0 * self.target.unrelated_ibs0
            rows.append(row)
        return rows

    # ---- populations over time --------------------------------------------------
    def members_at(self, gen: int) -> dict[int, int]:
        gen = max(1, min(gen, self.n_generations))
        cache = self._members_cache
        if cache and cache[0] <= gen and gen - cache[0] < 50:
            g0, members = cache[0], dict(cache[1])
        else:
            g0 = self._kf_gens[bisect.bisect_right(self._kf_gens, gen) - 1]
            members = dict(self._keyframes[g0])
        for g in range(g0 + 1, gen + 1):
            for cid, d in self._deltas.get(g, ()):
                n = members.get(cid, 0) + d
                if n:
                    members[cid] = n
                else:
                    members.pop(cid, None)
        self._members_cache = (gen, dict(members))
        return members

    def latest_fit(self, gen: int) -> tuple[int, int] | None:
        """(candidate id, evaluation) of the latest fit found by the end of `gen`."""
        evals = self.generations[max(1, min(gen, self.n_generations)) - 1]["evals"]
        i = bisect.bisect_right(self._fit_evals, evals - 1) - 1
        return (self._fit_cids[i], self._fit_evals[i]) if i >= 0 else None

    def _replay(self, gen: int, refs: tuple[str, ...], k: int) -> dict:
        """Closest candidates per reference and removal events, replayed up to `gen`
        (ties keep the candidate tracked last generation)."""
        st = self._replays.setdefault((refs, k), {"gen": 0, "tracked": {r: None for r in refs},
                                                  "last": {r: None for r in refs}, "frames": {}})
        for g in range(st["gen"] + 1, gen + 1):
            members = self.members_at(g)
            frame = {"closest": {}, "events": [], "last_removed": {}}
            for r in refs:
                tracked = st["tracked"][r]
                still = tracked if tracked in members else None
                order = sorted(members, key=lambda c: (self.distance(c, r)["structure"],
                                                       self.distance(c, r)["relationship"], c != still))
                frame["closest"][r] = order[:k]
                if tracked is not None and still is None:
                    ev = {"generation": g, "reference": r, "lost": tracked, "replaced_by": order[0],
                          "distance_before": self.distance(tracked, r)["structure"],
                          "distance_after": self.distance(order[0], r)["structure"]}
                    frame["events"].append(ev)
                    st["last"][r] = ev
                st["tracked"][r] = order[0]
                frame["last_removed"][r] = st["last"][r]
            st["frames"][g] = frame
        st["gen"] = max(st["gen"], gen)
        return st["frames"][gen]

    def frame(self, gen: int, refs: Sequence[str] = (), k: int = 1) -> dict:
        gen = max(1, min(gen, self.n_generations))
        refs = tuple(r for r in refs if r in self.refs)
        members = self.members_at(gen)
        cids = sorted(members)
        ev = self.evaluate(cids)
        objs = np.array([ev[c]["objectives"] for c in cids])
        best = cids[int(np.lexsort(objs.T[::-1])[0])]
        ranks = nondominated_ranks(objs)
        pareto = [c for c, r in zip(cids, ranks) if r == 0]
        rep = self._replay(gen, refs, k) if refs else {"closest": {}, "events": [], "last_removed": {}}
        lf = self.latest_fit(gen)

        def card(c):
            return self.card(c, refs, members.get(c))

        removed = {}
        for r, e in rep["last_removed"].items():
            if e is not None:
                removed[r] = {**e, "card": card(e["lost"]), "new": e["generation"] == gen}
        g = self.generations[gen - 1]
        return {"gen": gen, **g, "distinct": len(cids),
                "best": card(best),
                "pareto": [card(c) for c in pareto],
                "members": [card(c) for c in sorted(cids, key=lambda c: (ranks[cids.index(c)], c))],
                "closest": {r: [card(c) for c in cs] for r, cs in rep["closest"].items()},
                "removed": removed,
                "latest_fit": {**card(lf[0]), "found_at_eval": lf[1]} if lf else None}

    def series(self, refs: Sequence[str] = ()) -> dict:
        refs = tuple(r for r in refs if r in self.refs)
        out = {key: [g[key] for g in self.generations]
               for key in ("gen", "evals", "population", "n_phenotypes", "n_fits", "n_pareto")}
        out["distinct"] = []
        best = {name: [] for name in self.objective_names}
        closest = {r: [] for r in refs}
        events = []
        for g in range(1, self.n_generations + 1):
            members = self.members_at(g)
            out["distinct"].append(len(members))
            ev = self.evaluate(list(members))
            for j, name in enumerate(self.objective_names):
                best[name].append(min(e["objectives"][j] for e in ev.values()))
            if refs:
                fr = self._replay(g, refs, 1)
                for r in refs:
                    closest[r].append(self.distance(fr["closest"][r][0], r)["structure"])
                events += [{k_: v for k_, v in e.items() if k_ not in ("lost", "replaced_by")}
                           for e in fr["events"]]
        out["best"] = best
        out["closest"] = closest
        out["events"] = events
        first_fit = next((g["gen"] for g in self.generations if g["n_fits"] > 0), None)
        out["first_fit_gen"] = first_fit
        return out

    def snapshots(self, refs: Sequence[str] = (), k: int = 1, every: int = 1) -> list[dict]:
        """Snapshots in the Engine's format, for viz.write_gif / write_html."""
        refs = tuple(r for r in refs if r in self.refs)
        out = []
        gens = list(range(1, self.n_generations + 1, every))
        if gens and gens[-1] != self.n_generations:
            gens.append(self.n_generations)
        for g in range(1, self.n_generations + 1):
            rep = self._replay(g, refs, k) if refs else {"closest": {}, "events": [], "last_removed": {}}
            if g not in gens and not rep["events"]:
                continue
            members = self.members_at(g)
            cids = sorted(members)
            ev = self.evaluate(cids)
            objs = np.array([ev[c]["objectives"] for c in cids])

            def item(c):
                e = ev.get(c) or self.evaluate([c])[c]
                return {"pedigree": self.pedigree(c), "objectives": e["objectives"],
                        "worst_error": e["worst_error"], "fits": e["fits"],
                        "distances": {r: self.distance(c, r) for r in refs}}
            records = {"best": [item(cids[int(np.lexsort(objs.T[::-1])[0])])]}
            for r in refs:
                records[f"closest:{r}"] = [item(c) for c in rep["closest"][r]]
            ev_items = [{**e, "lost": item(e["lost"]), "replaced_by": item(e["replaced_by"])}
                        for e in rep["events"]]
            last = {r: (None if e is None else {**e, "lost": item(e["lost"]),
                                                "replaced_by": item(e["replaced_by"])})
                    for r, e in rep["last_removed"].items()}
            lf = self.latest_fit(g)
            gi = self.generations[g - 1]
            out.append({"generation": g, "evals": gi["evals"], "population": gi["population"],
                        "records": records, "events": ev_items, "last_removed": last,
                        "n_fits": gi["n_fits"], "latest_fit": self.pedigree(lf[0]) if lf else None,
                        "latest_fit_at": lf[1] if lf else None})
        return out

    def find(self, ped: Pedigree) -> int | None:
        """Candidate id of a pedigree in this database, if present."""
        row = self.db.conn.execute("SELECT id FROM candidates WHERE canon = ?",
                                   (canonical_form(ped, self.ids),)).fetchone()
        return row[0] if row else None


@lru_cache(maxsize=20000)
def _pedigree(canon: bytes, ids: tuple[str, ...]) -> Pedigree:
    return from_canonical(canon, list(ids))
