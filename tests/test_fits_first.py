"""Fit status keeps candidates (fits_first), without being an objective."""

import numpy as np

from pedigree_ea import expected_ibd, synth
from pedigree_ea.data.cases import load_case
from pedigree_ea.ea import Stopping, baseline_config, run, run_case
from pedigree_ea.ea.pareto import fit_first_ranks, select_by_ranks

from test_grid import TINY


def test_fit_first_ranks():
    # truth-like fit (0, 0, 3), simpler fit (0, 0, 1), unrelated non-fit (0.5, 0.5, 0)
    objs = np.array([[0, 0, 3], [0, 0, 1], [0.5, 0.5, 0], [0.6, 0.6, 1]], dtype=float)
    fits = np.array([True, True, False, False])
    ranks = fit_first_ranks(objs, fits)
    assert ranks.tolist()[:2] == [0, 0]            # fits do not compete
    assert ranks[2] == 1 and ranks[3] == 2         # non-fits keep Pareto order, after fits
    keep = set(select_by_ranks(objs, ranks, 2).tolist())
    assert keep == {0, 1}                          # both fits survive, the 3-latent one included


def test_duplicates_rank_last():
    objs = np.array([[0, 0, 1], [0, 0, 1], [0.5, 0.5, 0]], dtype=float)
    ranks = fit_first_ranks(objs, np.array([True, True, False]))
    keep = select_by_ranks(objs, ranks, 2, penalize=np.array([False, True, False]))
    assert sorted(keep.tolist()) == [0, 2]


def test_nsga2_holds_the_truth_once_found():
    case = load_case(f"{TINY['case_dir']}/avuncular")
    cfg = baseline_config("direct", max_latent=3, tol=1e-6, seed=0, snapshot=True,
                          stopping=Stopping(max_evals=10000))
    res, sc = run_case(cfg, case)
    edits = [s["records"]["closest:truth"][0]["distances"]["truth"]["structure"] for s in res.snapshots]
    assert sc["truth_at_eval"] is not None
    first = edits.index(0)
    assert all(e == 0 for e in edits[first:])      # held to the end


def test_nsga2_fits_first_can_be_turned_off():
    cfg = baseline_config("direct", max_latent=1, tol=1e-6, stopping=Stopping(max_evals=600),
                          strategy_params={"fits_first": False})
    assert cfg.strategy_params["fits_first"] is False
    assert len(run(cfg, expected_ibd(synth.half_sibs())).archive.fits) >= 1


def test_poss_keeps_every_fit_found():
    target = expected_ibd(synth.half_sibs())
    cfg = baseline_config("direct", "poss", max_latent=1, tol=1e-6, snapshot=True,
                          stopping=Stopping(max_evals=2000))
    res = run(cfg, target)
    in_archive = len(res.archive.fits)
    held = res.history[-1]["population"]
    assert in_archive == 3 and held >= in_archive   # all three fits are kept as members


def test_one_plus_lambda_never_leaves_a_fit():
    target = expected_ibd(synth.half_sibs())
    cfg = baseline_config("cgp", "1+lambda", max_latent=1, tol=1e-6, snapshot=True,
                          stopping=Stopping(max_evals=3000), strategy_params={"restart_window": None})
    res = run(cfg, target)
    fit_seen = False
    # Rows of each generation are the parent and its children; a fit has zero
    # error, so once the parent fits the best row of every later generation fits.
    for s in res.snapshots:
        best = s["records"]["best"][0]
        if fit_seen:
            assert best["fits"]
        fit_seen = fit_seen or best["fits"]
