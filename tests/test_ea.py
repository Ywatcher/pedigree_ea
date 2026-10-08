import json

import numpy as np
import pytest

from pedigree_ea import batch, expected_ibd, synth
from pedigree_ea.ea import (IMPLEMENTED, REPRESENTATIONS, Problem, RunLogger, Stopping,
                            baseline_config, run, run_case)
from pedigree_ea.ea.selection import crowding, dominance, nondominated_ranks, nsga2_select
from pedigree_ea.cases import Case


# ---- representation contract (run for every implemented representation) -------

@pytest.fixture(params=IMPLEMENTED)
def rep(request):
    return REPRESENTATIONS[request.param](n_obs=4, max_latent=3)


def test_decode_is_valid_and_reproducible(rep):
    for seed in range(20):
        g = rep.random(np.random.default_rng(seed))
        p = rep.decode(g)
        assert p.shape == (rep.n, 2)
        assert np.array_equal(p, batch.normalize(p))
        assert not batch.has_cycle(p[None])[0]
        assert np.array_equal(p, rep.decode(rep.random(np.random.default_rng(seed))))


def test_operators_do_not_modify_input(rep):
    rng = np.random.default_rng(0)
    g = rep.random(rng)
    before = rep.decode(g).copy()
    for _ in range(200):
        new, op = rep.mutate(g, rng)
        assert np.array_equal(rep.decode(g), before), op
        assert not batch.has_cycle(rep.decode(new)[None])[0], op
        g, before = new, rep.decode(new).copy()


def test_every_operator_can_apply(rep):
    rng = np.random.default_rng(1)
    applied = set()
    g = rep.random(rng)
    for _ in range(500):
        for name, op in rep.operators().items():
            new = op(g, rng)
            if new is not None:
                applied.add(name)
                g = new
    assert applied == set(rep.operators())


def test_crossover_valid(rep):
    if not rep.has_crossover:
        pytest.skip("no crossover")
    rng = np.random.default_rng(2)
    for _ in range(50):
        a, b = rep.random(rng), rep.random(rng)
        child = rep.crossover(a, b, rng)
        assert not batch.has_cycle(rep.decode(child)[None])[0]


@pytest.mark.parametrize("name", ["cyclic", "treegp"])
def test_open_designs_are_marked(name):
    with pytest.raises(NotImplementedError):
        REPRESENTATIONS[name](4, 2)


# ---- selection ----------------------------------------------------------------

def test_nondominated_ranks():
    objs = np.array([[1, 1], [2, 2], [1, 3], [3, 1], [3, 3]], dtype=float)
    assert nondominated_ranks(objs).tolist() == [0, 1, 1, 1, 2]
    assert dominance(objs)[0, 4]


def test_nsga2_select_prefers_front_and_spread():
    objs = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0], [5, 5]], dtype=float)
    keep = set(nsga2_select(objs, 3).tolist())
    assert 5 not in keep and {0, 4} <= keep
    assert np.isinf(crowding(objs, nondominated_ranks(objs))[[0, 4]]).all()


def test_nsga2_select_penalizes_duplicates():
    objs = np.array([[0, 0], [0, 0], [1, 1]], dtype=float)
    keep = nsga2_select(objs, 2, penalize=np.array([False, True, False]))
    assert sorted(keep.tolist()) == [0, 2]


# ---- problem and runs ---------------------------------------------------------

def test_problem_truth_fits():
    truth = synth.avuncular()
    prob = Problem(expected_ibd(truth), max_latent=3, tol=1e-6)
    arr = batch.from_pedigree(truth, prob.ids, 3)
    ev = prob.evaluate(arr[None])
    assert ev.fits[0] and ev.objectives[0, 0] == pytest.approx(0)
    assert ev.objectives[0, 2] == 3   # n_latent


def test_problem_rejects_cycles():
    prob = Problem(expected_ibd(synth.parent_child()), max_latent=0)
    ev = prob.evaluate(np.array([[[1, -1], [0, -1]]]))
    assert not ev.valid[0] and not ev.fits[0]


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_baseline_finds_parent_child(name):
    target = expected_ibd(synth.parent_child())
    cfg = baseline_config(name, max_latent=0, tol=1e-6, stopping=Stopping(max_evals=3000))
    if name == "cgp":
        # Reversing parent/child needs an order swap plus an input change, which
        # Pareto acceptance blocks; only restarts get there.
        cfg.strategy_params = {**cfg.strategy_params, "restart_window": 200}
    res = run(cfg, target)
    assert len(res.archive.fits) == 2   # A parent of B, or B parent of A


def test_logger_writes_files_with_shared_stamp(tmp_path):
    truth = synth.half_sibs()
    case = Case("half_sibs", "", truth, expected_ibd(truth), {}, None)
    logger = RunLogger("test", log_dir=tmp_path / "logs", results_dir=tmp_path / "results",
                       stamp="STAMP")
    cfg = baseline_config("direct", max_latent=1, tol=1e-6, stopping=Stopping(max_evals=500))
    run_case(cfg, case, logger)
    log = tmp_path / "logs" / "test_STAMP.jsonl"
    res = tmp_path / "results" / "test_STAMP.json"
    assert log.exists() and res.exists()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["event"] == "start" and records[-1]["event"] == "end"
    assert any("generation" in r for r in records)
    saved = json.loads(res.read_text())
    assert saved["stamp"] == "STAMP" and saved["summary"]["n_fits"] >= 1
    assert list((tmp_path / "results" / "test_STAMP_solutions").glob("*.tsv"))
