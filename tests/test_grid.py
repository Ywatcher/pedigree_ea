import json
from pathlib import Path

import numpy as np
import pytest

from pedigree_ea.ea import GridSpec, Problem, run_grid, run_grids
from pedigree_ea.ea.experiments.grid import expand, make_config, setting_label
from pedigree_ea import expected_ibd, synth, batch
from pedigree_ea.ea import Stopping, baseline_config, run

TINY = {
    "name": "tiny",
    "cases": ["parent_child", "half_sibs"],
    "case_dir": str(Path(__file__).resolve().parents[1] / "test_cases"),
    "seeds": 1,
    "stopping": {"max_evals": 400},
    "grid": {"strategy": ["nsga2", "poss"]},
}


def test_expand_and_labels():
    grid = {"strategy": ["nsga2", "poss"], "objectives": [None, ["excess_total", "n_latent"]],
            "representation": ["direct"]}
    settings = expand(grid)
    assert len(settings) == 4
    assert setting_label(settings[3], grid) == "strategy=poss objectives=excess_total+n_latent"
    assert expand({}) == [{}]
    with pytest.raises(ValueError):
        expand({"strategy": []})


def test_make_config_defaults_and_overrides():
    cfg = make_config({"strategy": "poss", "objectives": None}, {"representation": "neat"},
                      {"max_evals": 100}, {"max_latent": 2, "tol": 1e-6}, seed=4)
    assert (cfg.representation, cfg.strategy, cfg.seed) == ("neat", "poss", 4)
    assert cfg.objectives == ("excess_total", "n_latent")
    assert cfg.stopping.max_evals == 100 and cfg.max_latent == 2
    cfg = make_config({"strategy_params": {"lam": 3}}, {"strategy": "poss"}, {}, {}, seed=0)
    assert cfg.strategy_params["lam"] == 3 and "max_archive" in cfg.strategy_params
    with pytest.raises(ValueError, match="seed"):
        make_config({"seed": 1}, {}, {}, {}, seed=0)


def test_run_grid_writes_files(tmp_path):
    res = run_grid(TINY, log=True, log_dir=tmp_path / "logs", results_dir=tmp_path / "results",
                   stamp="S", verbose=False)
    assert len(res.rows) == 4 and len(res.summary) == 4
    assert all(r["recall"] == 1.0 for r in res.rows)
    saved = json.loads((tmp_path / "results" / "grid_tiny_S.json").read_text())
    assert saved["stamp"] == "S" and len(saved["rows"]) == 4
    assert (tmp_path / "results" / "grid_tiny_S_summary.csv").exists()
    assert len((tmp_path / "logs" / "grid_tiny_S.jsonl").read_text().splitlines()) == 4
    assert list((tmp_path / "logs").glob("tiny_parent_child_g000_s0_S.jsonl"))


def test_run_grid_parallel_matches_serial(tmp_path):
    a = run_grid(TINY, log=False, verbose=False)
    b = run_grid(GridSpec.from_dict(TINY), workers=2, log=False, verbose=False)
    key = lambda r: (r["target"], r["setting"], r["n_fits"], r["truth_at_eval"])
    assert sorted(map(key, a.rows)) == sorted(map(key, b.rows))


def test_poss_archive_finds_both_parent_child_orientations():
    target = expected_ibd(synth.parent_child())
    cfg = baseline_config("direct", "poss", max_latent=0, tol=1e-6,
                          stopping=Stopping(max_evals=1000))
    assert len(run(cfg, target).archive.fits) == 2


def test_excess_objectives_zero_within_tolerance():
    truth = synth.half_sibs()
    target = expected_ibd(truth)
    prob = Problem(target, max_latent=1, tol=0.1,
                   objectives=("excess_total", "excess_worst", "ibd_total"))
    grand = batch.from_pedigree(synth.grandparent(), prob.ids, 1)
    ev = prob.evaluate(grand[None])   # same IBD as half-sibs
    assert ev.objectives[0].tolist() == [0.0, 0.0, 0.0]
    unrelated = np.full((1, prob.n, 2), -1, dtype=np.int16)
    ev = prob.evaluate(unrelated)   # predicted (1, 0, 0) vs (0.5, 0.5, 0): error 0.5
    assert ev.objectives[0, 0] == pytest.approx(0.4)
    assert ev.objectives[0, 1] == pytest.approx(0.4)


def test_run_grids_shares_pool_and_writes_each_grid(tmp_path):
    other = {**TINY, "name": "tiny2", "grid": {"strategy": ["1+lambda"]}, "fixed": {"representation": "cgp"}}
    results = run_grids([TINY, other], workers=2, log=True, log_dir=tmp_path / "logs",
                        results_dir=tmp_path / "results", stamp="S", verbose=False)
    assert [len(r.rows) for r in results] == [4, 2]
    assert {r["grid"] for r in results[1].rows} == {"tiny2"}
    for name in ("tiny", "tiny2"):
        assert (tmp_path / "results" / f"grid_{name}_S_summary.csv").exists()
        assert (tmp_path / "logs" / f"grid_{name}_S.jsonl").exists()
    single = run_grid(TINY, log=False, verbose=False)
    key = lambda r: (r["target"], r["setting"], r["n_fits"], r["truth_at_eval"])
    assert sorted(map(key, results[0].rows)) == sorted(map(key, single.rows))


def test_run_grids_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        run_grids([TINY, TINY], log=False, verbose=False)


def test_answer_keys_computed_once_across_grids(tmp_path):
    from pedigree_ea import io
    from pedigree_ea.ea.experiments.grid import key_requests
    path = tmp_path / "hs.csv"
    io.write_ibd(expected_ibd(synth.half_sibs()), path)
    f = {"path": str(path), "target": "ibd", "max_latent": 1, "tol": 1e-6, "key": True}
    a = {"name": "a", "files": [f], "seeds": 1, "stopping": {"max_evals": 200}}
    b = {**a, "name": "b"}
    reqs = {r for s in (a, b) for r in key_requests(GridSpec.from_dict(s))}
    assert len(reqs) == 1
    ra, rb = run_grids([a, b], workers=2, log=False, verbose=False)
    assert ra.rows[0]["n_key"] == rb.rows[0]["n_key"] == 3      # half-sibs key: 3 pedigrees
