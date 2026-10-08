import pytest

from pedigree_ea import expected_kinship, inbreeding, synth


@pytest.mark.parametrize("name", list(synth.STANDARD))
def test_standard_pedigrees_are_valid(name):
    synth.STANDARD[name]().validate()


@pytest.mark.parametrize("seed", range(20))
def test_random_pedigree_is_valid(seed):
    ped = synth.random_pedigree(n_observed=5, seed=seed)
    ped.validate()
    assert len(ped.observed_ids) == 5
    assert ped.pruned().ids == ped.ids  # already pruned


def test_random_pedigree_is_reproducible():
    a = synth.random_pedigree(seed=7)
    b = synth.random_pedigree(seed=7)
    assert str(a) == str(b)


@pytest.mark.parametrize("seed", range(20))
def test_random_pedigree_without_inbreeding(seed):
    ped = synth.random_pedigree(n_observed=6, n_generations=4, p_marry_within=0.5, seed=seed)
    assert all(f == 0 for f in inbreeding(ped).values())


def test_observe_without_noise_is_exact():
    ped = synth.first_cousins()
    assert dict(synth.observe(ped).items()) == dict(expected_kinship(ped).items())


def test_observe_with_noise():
    ped = synth.random_pedigree(seed=1)
    exact = expected_kinship(ped)
    noisy = synth.observe(ped, noise_sd=0.01, seed=1)
    diffs = [noisy[p] - v for p, v in exact.items()]
    assert any(d != 0 for d in diffs)
    assert all(abs(d) < 0.1 for d in diffs)
