import numpy as np
import pytest

from pedigree_ea import (IBD, IBDData, Pedigree, expected_ibd, ibd_errors, jacquard,
                         kinship_matrix, synth)
from pedigree_ea.genetics.ibd import kinship_from_jacquard


@pytest.mark.parametrize("name, expected", [
    ("parent_child", (0, 1, 0)),
    ("full_sibs", (0.25, 0.5, 0.25)),
    ("half_sibs", (0.5, 0.5, 0)),
    ("grandparent", (0.5, 0.5, 0)),
    ("avuncular", (0.5, 0.5, 0)),
    ("first_cousins", (0.75, 0.25, 0)),
    ("double_first_cousins", (9 / 16, 6 / 16, 1 / 16)),
])
def test_standard_ibd(name, expected):
    assert tuple(expected_ibd(synth.STANDARD[name]())["A", "B"]) == pytest.approx(expected)


def test_unrelated_and_missing_pairs():
    ibd = expected_ibd(synth.two_families())
    assert tuple(ibd["A", "C"]) == (1, 0, 0)
    assert tuple(ibd["C", "D"]) == pytest.approx((0, 1, 0))


def inbred_pedigrees():
    # Child of full siblings, observed with a parent and an aunt/uncle.
    sib_mating = synth.full_sibs(3)
    sib_mating.add("D", ["A", "B"])
    # Parent-child mating.
    pc = Pedigree()
    pc.add("A")
    pc.add("B", ["A"])
    pc.add("C", ["A", "B"])
    pc.add("D", ["A", "B"])
    return [sib_mating, pc, synth.related_parents()]


def _simulate(ped: Pedigree, reps: int, rng) -> dict[tuple[str, str], np.ndarray]:
    """Forward gene dropping: independent check of the exact computation."""
    order = ped.topological_order()
    ids = ped.observed_ids
    pairs = [(a, b) for k, a in enumerate(ids) for b in ids[k + 1:]]
    counts = {p: np.zeros(3) for p in pairs}
    for _ in range(reps):
        genes, next_label = {}, 0
        for pid in order:
            g = []
            for s in range(2):
                ps = ped[pid].parents
                if s < len(ps):
                    g.append(genes[ps[s]][rng.integers(2)])
                else:
                    g.append(next_label)
                    next_label += 1
            genes[pid] = g
        for a, b in pairs:
            (x1, x2), (y1, y2) = genes[a], genes[b]
            shared = max((x1 == y1) + (x2 == y2), (x1 == y2) + (x2 == y1))
            counts[(a, b)][shared] += 1
    return {p: c / reps for p, c in counts.items()}


@pytest.mark.parametrize("ped", inbred_pedigrees())
def test_exact_matches_simulation_with_inbreeding(ped):
    rng = np.random.default_rng(0)
    sim = _simulate(ped, 20000, rng)
    exact = expected_ibd(ped)
    for pair, freq in sim.items():
        assert np.asarray(exact[pair]) == pytest.approx(freq, abs=0.02), pair


def test_jacquard_consistent_with_kinship():
    peds = inbred_pedigrees() + [synth.random_pedigree(seed=s, allow_inbreeding=True,
                                                      p_marry_within=0.6, n_generations=4)
                                for s in range(10)]
    for ped in peds:
        ids, phi = kinship_matrix(ped)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                delta = jacquard(ped, ids[a], ids[b])
                assert delta.sum() == pytest.approx(1)
                assert kinship_from_jacquard(delta) == pytest.approx(phi[a, b])


def test_fast_path_matches_full_jacquard():
    from pedigree_ea.genetics.ibd import ibd_from_jacquard
    peds = inbred_pedigrees() + [synth.random_pedigree(seed=s, allow_inbreeding=True,
                                                      p_marry_within=0.6, n_generations=4)
                                for s in range(10)]
    for ped in peds:
        ids = ped.ids
        fast = expected_ibd(ped, ids)
        for k, a in enumerate(ids):
            for b in ids[k + 1:]:
                slow = ibd_from_jacquard(jacquard(ped, a, b))
                assert np.asarray(fast[a, b]) == pytest.approx(slow, abs=1e-12), (a, b)


def test_child_of_sibs_with_parent():
    ped = inbred_pedigrees()[0]
    # Parent A and inbred child D. D's gene from B is IBD with the gene D got
    # from A (prob 1/4, Delta5), with A's other gene (1/4, Delta7), or neither
    # (1/2, Delta8).
    delta = jacquard(ped, "A", "D")
    assert delta == pytest.approx([0, 0, 0, 0, 0.25, 0, 0.25, 0.5, 0])


@pytest.mark.parametrize("seed", range(10))
def test_adding_a_parent_never_decreases_sharing(seed):
    rng = np.random.default_rng(seed)
    ped = synth.random_pedigree(n_observed=5, seed=seed)
    before = expected_ibd(ped)
    order = ped.topological_order()
    for _ in range(20):
        child = rng.choice(order[1:])
        earlier = order[:order.index(child)]
        parent = rng.choice(earlier)
        if len(ped[child].parents) < 2 and parent not in ped[child].parents:
            ped.set_parents(child, (*ped[child].parents, str(parent)))
            break
    after = expected_ibd(ped)
    for pair, v in before.items():
        assert after[pair].k0 <= v.k0 + 1e-12
        assert after[pair].k2 >= v.k2 - 1e-12


def test_ibd_helpers():
    v = IBD(0.25, 0.5, 0.25)
    assert v.kinship == pytest.approx(0.25)
    assert v.pi_hat == pytest.approx(0.5)
    assert IBD.from_k1_k2(1, 0) == (0, 1, 0)
    data = IBDData({("B", "A"): (0.5, 0.5, 0)})
    assert data["A", "B"].k1 == 0.5
    assert data.kinship()["A", "B"] == pytest.approx(0.125)
    err = ibd_errors(expected_ibd(synth.half_sibs()), IBDData({("A", "B"): (0.4, 0.6, 0)}))
    assert err[("A", "B")] == pytest.approx([0.1, -0.1, 0])
