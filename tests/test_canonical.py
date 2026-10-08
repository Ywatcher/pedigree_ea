from pedigree_ea import PedigreeSet, same_structure, structure_hash
from pedigree_ea import synth


def test_latent_renaming_is_same_structure():
    a = synth.first_cousins()
    b = a.relabel({"L1": "X", "L3": "Y", "L4": "L3"})
    assert same_structure(a, b)
    assert structure_hash(a) == structure_hash(b)


def test_observed_renaming_is_different():
    a = synth.grandparent()
    b = a.relabel({"A": "B", "B": "A"})  # now B is the grandparent
    assert not same_structure(a, b)


def test_different_relationships_differ():
    assert not same_structure(synth.half_sibs(), synth.full_sibs())


def test_pedigree_set_dedupes():
    s = PedigreeSet()
    assert s.add(synth.full_sibs())
    assert not s.add(synth.full_sibs().relabel({"L1": "Q"}))
    assert s.add(synth.half_sibs())
    assert len(s) == 2
    assert synth.half_sibs() in s


def test_pedigree_set_prunes_before_comparing():
    ped = synth.parent_child()
    extra = ped.copy()
    extra.add("L9", observed=False)
    extra.set_parents("A", ["L9"])  # unknown parent made explicit
    s = PedigreeSet()
    s.add(ped)
    assert not s.add(extra)
