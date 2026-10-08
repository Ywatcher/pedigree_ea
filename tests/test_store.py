import json
import sqlite3

import numpy as np
import pytest

from pedigree_ea import batch, expected_ibd, same_structure, synth
from pedigree_ea.data.cases import load_case
from pedigree_ea.ea import RunLogger, Stopping, baseline_config, run, run_case
from pedigree_ea.genetics.canonical import (canonical_form, canonical_graph, canonical_parents,
                                            from_canonical)
from pedigree_ea.ea.experiments.serialize import to_jsonable
from pedigree_ea.store import merge
from pedigree_ea.store.reader import Database
from pedigree_ea.webviz.server import App

from test_grid import TINY

CASES = TINY["case_dir"]


# ---- canonical form ------------------------------------------------------------

@pytest.mark.parametrize("name", list(synth.STANDARD))
def test_canonical_form_ignores_latent_names_and_round_trips(name):
    ped = synth.STANDARD[name]()
    ids = ped.observed_ids
    lat = ped.latent_ids
    renamed = ped.relabel({x: f"Z{i}" for i, x in enumerate(reversed(lat))})
    c = canonical_form(ped, ids)
    assert canonical_form(renamed, ids) == c
    assert same_structure(from_canonical(c, ids), ped.pruned())
    assert canonical_parents(c).shape == (len(ids) + len(ped.pruned().latent_ids), 2)


def test_canonical_form_equals_iff_same_structure():
    peds = [synth.random_pedigree(n_observed=4, seed=s) for s in range(40)]
    for a in peds:
        for b in peds[:15]:
            ids = sorted(a.observed_ids)
            if ids != sorted(b.observed_ids):
                continue
            assert (canonical_form(a, ids) == canonical_form(b, ids)) == same_structure(a.pruned(), b.pruned())


def test_canonical_graph_handles_cycles_and_weights():
    e1 = [("A", "x"), ("x", "y"), ("y", "x"), ("y", "B")]
    e2 = [("A", "q"), ("q", "p"), ("p", "q"), ("p", "B")]
    assert canonical_graph(["A", "B"], ["x", "y"], e1, [1, 2, 3, 4]) == \
        canonical_graph(["A", "B"], ["q", "p"], e2, [1, 2, 3, 4])
    assert canonical_graph(["A", "B"], ["x", "y"], e1, [1, 2, 3, 4]) != \
        canonical_graph(["A", "B"], ["x", "y"], e1, [1, 2, 3, 5])


# ---- writer -> reader ------------------------------------------------------------

def _run_logged(tmp_path, strategy=None, name="r", seed=0, case="avuncular", max_evals=3000):
    lg = RunLogger(name, tmp_path / "logs", tmp_path / "results", stamp="S",
                   db_path=tmp_path / "db" / f"{name}.sqlite")
    c = load_case(f"{CASES}/{case}")
    ak = c.info["answer_key"]
    cfg = baseline_config("direct", strategy, max_latent=ak["max_latent"], tol=ak["tol"], seed=seed,
                          snapshot=True, stopping=Stopping(max_evals=max_evals))
    res, sc = run_case(cfg, c, lg)
    return lg, res, sc


@pytest.mark.parametrize("strategy", [None, "poss"])
def test_population_replay_matches_run(tmp_path, strategy):
    lg, res, sc = _run_logged(tmp_path, strategy)
    view = Database(lg.db_file).run(lg.db_run_id)
    assert view.n_generations == len(res.history)
    ids = view.ids
    for snap in res.snapshots[::7]:      # in-memory best vs stored population
        members = view.members_at(snap["generation"])
        assert sum(members.values()) == snap["population"]
        best = snap["records"]["best"][0]["pedigree"]
        assert any(same_structure(view.pedigree(c), best) for c in members)
    fits = {c for c, a in view._appear.items() if a["fit_at_eval"] is not None}
    assert len(fits) == len(res.archive.fits)
    stored = [view.pedigree(c) for c in fits]
    assert all(any(same_structure(p, f.pedigree) for p in stored) for f in res.archive.fits)
    for a in view._appear.values():      # lineage points at stored candidates
        assert all(view.canon(p) for p in a["parents"])


def test_reader_events_match_in_memory_recording(tmp_path):
    lg, res, _ = _run_logged(tmp_path, max_evals=4000)
    view = Database(lg.db_file).run(lg.db_run_id)
    mem = [(e["generation"], e["distance_before"], e["distance_after"])
           for s in res.snapshots for e in s["events"]]
    stored = [(e["generation"], e["distance_before"], e["distance_after"])
              for e in view.series(["truth"])["events"]]
    assert stored == mem


def test_frame_and_per_pair(tmp_path):
    lg, res, _ = _run_logged(tmp_path)
    view = Database(lg.db_file).run(lg.db_run_id)
    f = view.frame(5, ["truth"], 2)
    assert f["distinct"] == len(f["members"]) and f["best"]["layout"]["nodes"]
    assert len(f["closest"]["truth"]) <= 2 and f["pareto"]
    rows = view.per_pair(f["best"]["cid"])
    assert {"obs_ibd0", "pred_ibd0", "pred_kinship"} <= set(rows[0])
    json.dumps(to_jsonable(f))   # what the web server sends


def test_extra_reference_after_the_run(tmp_path):
    lg, res, _ = _run_logged(tmp_path)
    view = Database(lg.db_file).run(lg.db_run_id, {"halfsibs": synth.half_sibs()})
    assert set(view.refs) == {"truth", "halfsibs"}
    assert view.series(["halfsibs"])["closest"]["halfsibs"]


# ---- merge and grid --------------------------------------------------------------

def test_merge_deduplicates_candidates(tmp_path):
    a, _, sa = _run_logged(tmp_path, name="a", seed=0)
    b, _, sb = _run_logged(tmp_path, name="b", seed=1)
    out = merge([a.db_file, b.db_file], tmp_path / "m.sqlite")
    con = sqlite3.connect(out)
    n = lambda path, t: sqlite3.connect(path).execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    assert n(out, "runs") == 2
    assert n(out, "candidates") < n(a.db_file, "candidates") + n(b.db_file, "candidates")
    assert n(out, "membership") == n(a.db_file, "membership") + n(b.db_file, "membership")
    scores = sorted(json.loads(s)["recall"] for (s,) in con.execute("SELECT score FROM runs"))
    assert scores == sorted([sa["recall"], sb["recall"]])
    view = Database(out).run(2)
    assert view.series(["truth"])["gen"]


def test_grid_writes_one_database_per_grid(tmp_path):
    from pedigree_ea.ea import run_grids
    (r,) = run_grids([TINY], workers=2, log_dir=tmp_path / "logs", results_dir=tmp_path / "results",
                     stamp="S", verbose=False)
    db = Database(r.paths["db"])
    assert len(db.runs()) == 4
    assert not (tmp_path / "results" / "db" / "grid_tiny_S_shards").exists()


def test_web_app_endpoints(tmp_path):
    lg, _, _ = _run_logged(tmp_path)
    app = App(lg.db_file, {"halfsibs": synth.half_sibs()})
    runs = app.handle("/api/runs", {})["runs"]
    rid = runs[0]["id"]
    meta = app.handle(f"/api/run/{rid}", {})
    assert {r["label"] for r in meta["references"]} == {"truth", "halfsibs"}
    series = app.handle(f"/api/run/{rid}/series", {"refs": ["truth"]})
    assert len(series["gen"]) == meta["generations"]
    frame = app.handle(f"/api/run/{rid}/frame", {"gen": ["3"], "refs": ["truth"], "k": ["2"]})
    cand = app.handle(f"/api/run/{rid}/candidate/{frame['best']['cid']}", {"refs": ["truth"]})
    assert cand["per_pair"]
    with pytest.raises(KeyError):
        app.handle("/api/nope", {})
