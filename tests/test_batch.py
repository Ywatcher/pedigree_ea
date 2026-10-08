import numpy as np
import pytest

from pedigree_ea import batch, expected_ibd, kinship_matrix, synth
from pedigree_ea.genetics.canonical import same_structure
from pedigree_ea.genetics.ibd import ibd_from_parents
from pedigree_ea.genetics.kinship import kinship_from_parents


def random_dag(rng, n, p_edge):
    order = rng.permutation(n)
    par = np.full((n, 2), -1, dtype=np.int16)
    for t in range(1, n):
        k = 0
        for p in rng.permutation(order[:t]):
            if k < 2 and rng.random() < p_edge:
                par[order[t], k] = p
                k += 1
    return batch.normalize(par)


def test_normalize():
    p = np.array([[3, 1], [-1, 2], [4, 4], [-1, -1]])
    assert batch.normalize(p).tolist() == [[1, 3], [2, -1], [4, -1], [-1, -1]]


@pytest.mark.parametrize("seed", range(5))
def test_kinship_and_ibd_match_exact(seed):
    rng = np.random.default_rng(seed)
    n = 9
    par = np.stack([random_dag(rng, n, rng.uniform(0.1, 0.7)) for _ in range(16)])
    anc = batch.ancestor_matrix(par)
    phi, f = batch.kinship(par)
    pi, pj = np.triu_indices(n, 1)
    got = batch.ibd(par, pi, pj, phi, f, anc)
    for b in range(len(par)):
        plist = [[int(x) for x in row if x >= 0] for row in par[b]]
        ref_phi = kinship_from_parents(plist)
        assert phi[b] == pytest.approx(ref_phi, abs=1e-12)
        ref = ibd_from_parents(plist, list(zip(pi.tolist(), pj.tolist())), ref_phi)
        assert got[b] == pytest.approx(np.asarray(ref), abs=1e-12)


def test_cycle_detection():
    par = np.array([[[1, -1], [2, -1], [0, -1]], [[-1, -1], [0, -1], [1, -1]]])
    assert batch.has_cycle(par).tolist() == [True, False]


def test_round_trip_and_standard_values():
    for name, make in synth.STANDARD.items():
        ped = make()
        obs = ped.observed_ids
        arr = batch.from_pedigree(ped, obs, len(ped.latent_ids) + 1)
        assert same_structure(batch.to_pedigree(arr, obs), ped.pruned()), name
        par = arr[None]
        phi, f = batch.kinship(par)
        ids, ref = kinship_matrix(ped)
        pi, pj = np.triu_indices(len(obs), 1)
        got = batch.ibd(par, pi, pj, phi, f, batch.ancestor_matrix(par))[0]
        exact = expected_ibd(ped)
        for k, (i, j) in enumerate(zip(pi, pj)):
            assert got[k] == pytest.approx(np.asarray(exact[obs[i], obs[j]]), abs=1e-12)


def test_relevant_latent_matches_pruned():
    rng = np.random.default_rng(3)
    n_obs = 4
    for _ in range(50):
        par = random_dag(rng, 8, 0.4)
        ped = batch.to_pedigree(par, [f"P{i}" for i in range(n_obs)], prune=False)
        kept = len(ped.pruned().latent_ids)
        assert batch.relevant_latent(par[None], n_obs)[0].sum() == kept


def test_sex_conflict():
    triangle = np.array([[[-1, -1], [-1, -1], [-1, -1], [0, 1], [1, 2], [0, 2]]])
    ok = np.array([[[-1, -1], [-1, -1], [0, 1], [0, 1]]])
    assert batch.sex_conflict(triangle).tolist() == [True]
    assert batch.sex_conflict(ok).tolist() == [False]
