import numpy as np
import pytest

from pedigree_ea import (IBDData, KinshipData, PedigreeSet, enumerate_pedigrees, expected_ibd,
                         expected_kinship, synth)


def as_set(peds):
    s = PedigreeSet()
    for p in peds:
        s.add(p)
    return s


def test_parent_child_without_latent():
    target = KinshipData({("A", "B"): 0.25})
    results = enumerate_pedigrees(target, max_latent=0)
    s = as_set(results)
    assert len(results) == 2  # A parent of B, or B parent of A
    assert synth.parent_child() in s


def test_first_degree_with_latent_includes_full_sibs():
    target = KinshipData({("A", "B"): 0.25})
    s = as_set(enumerate_pedigrees(target, max_latent=2))
    assert synth.full_sibs() in s
    assert synth.parent_child() in s


def test_second_degree_alternatives():
    target = KinshipData({("A", "B"): 0.125})
    s = as_set(enumerate_pedigrees(target, max_latent=2, allow_inbreeding=False))
    assert synth.half_sibs() in s
    assert synth.grandparent() in s
    assert synth.grandparent().relabel({"A": "B", "B": "A"}) in s
    # Avuncular needs 3 latent people, so it must not appear yet.
    assert synth.avuncular() not in s


def test_all_results_fit():
    target = KinshipData({("A", "B"): 0.25, ("A", "C"): 0.125, ("B", "C"): 0.0625})
    for ped in enumerate_pedigrees(target, max_latent=2, tol=0.01):
        ped.validate()
        k = expected_kinship(ped, target.ids)
        assert all(abs(k[p] - v) <= 0.01 for p, v in target.items())


def test_unrelated_pairs_force_disconnection():
    target = KinshipData({("A", "B"): 0.0})
    results = enumerate_pedigrees(target, max_latent=1)
    assert len(results) == 1
    assert results[0].ids == ["A", "B"]


@pytest.mark.parametrize("name", ["half_sibs", "grandparent", "full_sibs"])
def test_truth_is_recovered(name):
    truth = synth.STANDARD[name]()
    target = expected_kinship(truth)
    s = as_set(enumerate_pedigrees(target, max_latent=len(truth.latent_ids)))
    assert truth in s


def test_max_results():
    target = KinshipData({("A", "B"): 0.25})
    assert len(enumerate_pedigrees(target, max_latent=2, max_results=1)) == 1


# ---- IBD targets ----------------------------------------------------------

def test_ibd_separates_parent_child_from_full_sibs():
    pc = as_set(enumerate_pedigrees(IBDData({("A", "B"): (0, 1, 0)}), max_latent=2))
    fs = as_set(enumerate_pedigrees(IBDData({("A", "B"): (0.25, 0.5, 0.25)}), max_latent=2))
    assert synth.parent_child() in pc and synth.full_sibs() not in pc
    assert synth.full_sibs() in fs and synth.parent_child() not in fs


def test_ibd_second_degree():
    s = as_set(enumerate_pedigrees(IBDData({("A", "B"): (0.5, 0.5, 0)}), max_latent=2))
    assert synth.half_sibs() in s
    assert synth.grandparent() in s


@pytest.mark.parametrize("name", ["full_sibs", "sibs_with_children", "two_families"])
def test_ibd_truth_is_recovered(name):
    truth = synth.STANDARD[name]()
    target = expected_ibd(truth)
    results = enumerate_pedigrees(target, max_latent=len(truth.latent_ids))
    assert truth in as_set(results)
    for ped in results:
        got = expected_ibd(ped, target.ids)
        for pair, v in target.items():
            assert np.asarray(got[pair]) == pytest.approx(v, abs=0.01)
