import pytest

from pedigree_ea import FEMALE, MALE, Pedigree, PedigreeError
from pedigree_ea.data.synth import full_sibs, grandparent


def test_topological_order_parents_first():
    ped = grandparent()
    order = ped.topological_order()
    assert order.index("A") < order.index("L1") < order.index("B")


def test_cycle_detected():
    ped = Pedigree()
    ped.add("A", ["B"])
    ped.add("B", ["A"])
    with pytest.raises(PedigreeError, match="cycle"):
        ped.topological_order()
    assert not ped.is_valid()


def test_unknown_parent_and_bad_parent_lists():
    ped = Pedigree()
    ped.add("A", ["X"])
    ped.add("B", ["B"])
    ped.add("C", ["A", "A"])
    problems = " ".join(ped.problems())
    assert "unknown parent" in problems
    assert "own parent" in problems
    assert "same parent twice" in problems


def test_sex_conflict_triangle_of_mates():
    # A-B, B-C and A-C all have children together: impossible with two sexes.
    ped = Pedigree()
    for pid in "ABC":
        ped.add(pid)
    ped.add("x", ["A", "B"])
    ped.add("y", ["B", "C"])
    ped.add("z", ["A", "C"])
    assert ped.sex_conflicts()
    assert ped.is_valid(check_sex=False)


def test_sex_conflict_with_known_sexes():
    ped = Pedigree()
    ped.add("A", sex=MALE)
    ped.add("B", sex=MALE)
    ped.add("x", ["A", "B"])
    assert ped.sex_conflicts()
    ped2 = Pedigree()
    ped2.add("A", sex=MALE)
    ped2.add("B", sex=FEMALE)
    ped2.add("x", ["A", "B"])
    assert not ped2.sex_conflicts()


def test_remove_drops_parent_links():
    ped = full_sibs()
    ped.remove("L1")
    assert ped.parents("A") == ("L2",)


def test_copy_is_independent():
    ped = full_sibs()
    other = ped.copy()
    other.set_parents("A", [])
    assert ped.parents("A") == ("L1", "L2")


def test_pruned_removes_irrelevant_latent_people():
    ped = Pedigree()
    ped.add("L1", observed=False)            # founder with one child -> removable
    ped.add("A", ["L1"])
    ped.add("L2", ["A"], observed=False)     # no observed descendants -> removable
    ped.add("L3", observed=False)            # founder with two children -> kept
    ped.add("B", ["L3"])
    ped.add("C", ["L3"])
    pruned = ped.pruned()
    assert set(pruned) == {"A", "L3", "B", "C"}
    assert pruned.parents("A") == ()


def test_pruned_keeps_latent_link_between_generations():
    pruned = grandparent().pruned()
    assert set(pruned) == {"A", "L1", "B"}


def test_relabel():
    ped = full_sibs().relabel({"L1": "X"})
    assert ped.parents("A") == ("X", "L2")
