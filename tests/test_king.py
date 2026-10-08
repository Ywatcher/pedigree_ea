import numpy as np
import pytest

from pedigree_ea import (KingData, Pedigree, PedigreeSet, batch, enumerate_pedigrees,
                         expected_ibd, expected_kinship, io, synth)
from pedigree_ea.ea import Problem, Stopping, baseline_config, run

from test_io import KIN0

BASE = 0.075


def king_from(ped: Pedigree) -> KingData:
    """Exact KING-like data for a pedigree: kinship, and IBS0 = IBD0 * BASE."""
    kin, ibd = expected_kinship(ped), expected_ibd(ped)
    return KingData({p: (kin[p], ibd[p].k0 * BASE) for p in kin}, kin.ids, unrelated_ibs0=BASE)


def task3_candidate(sibs: bool) -> Pedigree:
    """Report candidate 4 (100, 200, 300 full sibs) or 5 (200 parent of 100, 300)."""
    p = Pedigree()
    for pid in ("A", "B"):
        p.add(pid, observed=False)
    if sibs:
        for pid in ("100", "200", "300"):
            p.add(pid, ["A", "B"])
        p.add("400", ["200"])
    else:
        p.add("200")
        p.add("100", ["200", "A"])
        p.add("300", ["200", "A"])
        p.add("400", ["200", "B"])
    p.add("500")
    return p


def test_king_data_baseline_and_ibd0():
    d = KingData({("A", "B"): (0.25, 0.019), ("A", "C"): (-0.05, 0.076), ("B", "C"): (-0.07, 0.074)})
    assert d.unrelated_ibs0 == pytest.approx(0.075)
    assert d.ibd0("A", "B") == pytest.approx(0.019 / 0.075)
    assert d.kinship()["A", "C"] == 0
    with pytest.raises(ValueError, match="baseline"):
        KingData({("A", "B"): (0.25, 0.0)})


def test_read_king(tmp_path):
    path = tmp_path / "x.kin0"
    path.write_text(KIN0)
    d = io.read_king(path)
    assert d["100", "200"].ibs0 == pytest.approx(0.0198108)
    assert d.ibd0("200", "300") == pytest.approx(0.22, abs=0.01)


def test_problem_separates_siblings_from_parent_child(tmp_path):
    path = tmp_path / "x.kin0"
    path.write_text(KIN0)
    target = io.read_king(path)
    prob = Problem(target, max_latent=2, tol=0.05, tol_ibd0=0.15)
    arrs = np.stack([batch.from_pedigree(task3_candidate(s), prob.ids, 2) for s in (True, False)])
    ev = prob.evaluate(arrs)
    assert ev.fits.tolist() == [True, False]
    # Kinship alone cannot tell them apart.
    kin_prob = Problem(target.kinship(), max_latent=2, tol=0.05)
    assert kin_prob.evaluate(arrs).fits.tolist() == [True, True]


@pytest.mark.parametrize("name", ["full_sibs", "parent_child", "half_sibs", "avuncular"])
def test_brute_force_with_king_target(name):
    truth = synth.STANDARD[name]()
    target = king_from(truth)
    results = enumerate_pedigrees(target, max_latent=len(truth.latent_ids), tol=1e-6, tol_ibd0=1e-6)
    found = PedigreeSet()
    for p in results:
        found.add(p)
        k, ibd = expected_kinship(p, target.ids), expected_ibd(p, target.ids)
        for pair, v in target.items():
            assert k[pair] == pytest.approx(v.kinship, abs=1e-6)
            assert ibd[pair].k0 == pytest.approx(v.ibs0 / BASE, abs=1e-6)
    assert truth in found


def test_ea_and_brute_force_agree_on_king_target():
    truth = synth.full_sibs()
    target = king_from(truth)
    key = PedigreeSet()
    for p in enumerate_pedigrees(target, max_latent=2, tol=1e-6, tol_ibd0=1e-6):
        key.add(p)
    cfg = baseline_config("direct", max_latent=2, tol=1e-6, tol_ibd0=1e-6,
                          stopping=Stopping(max_evals=3000))
    res = run(cfg, target)
    assert res.archive.fits and all(f.pedigree in key for f in res.archive.fits)
    assert truth in key
