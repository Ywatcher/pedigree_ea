import math

import pytest

from pedigree_ea import (KinshipData, Pedigree, degree_class, expected_kinship,
                         inbreeding, kinship_matrix, kinship_to_degree, pair_errors)
from pedigree_ea import synth


@pytest.mark.parametrize("name, expected", [
    ("parent_child", 0.25),
    ("full_sibs", 0.25),
    ("half_sibs", 0.125),
    ("grandparent", 0.125),
    ("avuncular", 0.125),
    ("first_cousins", 0.0625),
    ("double_first_cousins", 0.125),
])
def test_standard_pair_kinship(name, expected):
    k = expected_kinship(synth.STANDARD[name]())
    assert k["A", "B"] == pytest.approx(expected)


def test_self_kinship_is_half_without_inbreeding():
    ids, phi = kinship_matrix(synth.first_cousins())
    assert all(phi[i, i] == pytest.approx(0.5) for i in range(len(ids)))


def test_child_of_full_sibs():
    ped = synth.full_sibs()
    ped.add("C", ["A", "B"])
    f = inbreeding(ped)
    assert f["C"] == pytest.approx(0.25)
    assert f["A"] == pytest.approx(0.0)
    k = expected_kinship(ped, ["A", "C"])
    # 1/2 * (phi(A,A) + phi(A,B)) = 1/2 * (0.5 + 0.25)
    assert k["A", "C"] == pytest.approx(0.375)


def test_related_parents():
    ped = synth.related_parents()
    # B's parents are first cousins: F = phi(A, L5) = 1/16
    assert inbreeding(ped)["B"] == pytest.approx(1 / 16)
    # phi(A, B) = 1/2 * (phi(A,A) + phi(A,L5)) = 1/2 * (1/2 + 1/16)
    assert expected_kinship(ped)["A", "B"] == pytest.approx(9 / 32)


def test_sibs_with_children():
    k = expected_kinship(synth.sibs_with_children())
    assert k["A", "B"] == pytest.approx(0.25)
    assert k["A", "D"] == pytest.approx(0.25)    # parent
    assert k["B", "D"] == pytest.approx(0.125)   # aunt
    assert k["D", "E"] == pytest.approx(0.0625)  # first cousins


def test_two_families_unrelated():
    k = expected_kinship(synth.two_families())
    assert k["A", "C"] == 0
    assert k["C", "D"] == pytest.approx(0.25)


def test_missing_parent_counts_as_unrelated():
    ped = Pedigree()
    ped.add("A")
    ped.add("B", ["A"])
    ped.add("C", ["B"])
    k = expected_kinship(ped)
    assert k["A", "C"] == pytest.approx(0.125)


def test_kinship_data_is_unordered():
    d = KinshipData({("B", "A"): 0.25})
    assert d["A", "B"] == d["B", "A"] == 0.25
    assert ("A", "B") in d
    assert d.ids == ["A", "B"]
    ids, m = d.matrix()
    assert m[0, 1] == 0.25 and math.isnan(m[0, 0])


def test_pair_errors():
    expected = expected_kinship(synth.half_sibs())
    target = KinshipData({("A", "B"): 0.1})
    assert pair_errors(expected, target)[("A", "B")] == pytest.approx(0.025)


def test_degrees():
    assert kinship_to_degree(0.25) == pytest.approx(1)
    assert kinship_to_degree(0.0625) == pytest.approx(3)
    assert kinship_to_degree(0) == math.inf
    assert degree_class(0.5) == 0
    assert degree_class(0.2357) == 1
    assert degree_class(0.0997) == 2
    assert degree_class(0.05) == 3
    assert degree_class(-0.08) is None
