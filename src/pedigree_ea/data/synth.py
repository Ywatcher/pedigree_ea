"""Synthetic test pedigrees and observations.

- `STANDARD`: small named pedigrees with known relationships.
- `random_pedigree`: random multi-generation families with hidden ancestors.
- `observe_ibd` / `observe`: turn a pedigree into pairwise IBD or kinship
  data, optionally noisy.

Observed people are named P1, P2, ... (or A, B, ... in standard pedigrees);
latent people L1, L2, ...
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..genetics.ibd import IBDData, expected_ibd
from ..genetics.kinship import KinshipData, expected_kinship
from ..genetics.pedigree import FEMALE, MALE, Pedigree


def _build(rows: list[tuple]) -> Pedigree:
    """rows: (id, parents, sex). Ids starting with 'L' are latent."""
    ped = Pedigree()
    for pid, parents, sex in rows:
        ped.add(pid, parents, sex, observed=not pid.startswith("L"))
    return ped


# ---- standard pedigrees ---------------------------------------------------

def parent_child() -> Pedigree:
    return _build([("A", (), MALE), ("B", ("A",), FEMALE)])


def trio() -> Pedigree:
    """Both parents (A, B) and their child C observed."""
    return _build([("A", (), MALE), ("B", (), FEMALE), ("C", ("A", "B"), None)])


def three_generations() -> Pedigree:
    """Grandparent A, parent B and grandchild C, all observed."""
    return _build([("A", (), MALE), ("B", ("A",), FEMALE), ("C", ("B",), None)])


def full_sibs(n: int = 2) -> Pedigree:
    rows = [("L1", (), MALE), ("L2", (), FEMALE)]
    rows += [(chr(ord("A") + k), ("L1", "L2"), None) for k in range(n)]
    return _build(rows)


def half_sibs() -> Pedigree:
    return _build([("L1", (), MALE), ("A", ("L1",), None), ("B", ("L1",), None)])


def grandparent() -> Pedigree:
    return _build([("A", (), MALE), ("L1", ("A",), FEMALE), ("B", ("L1",), None)])


def avuncular() -> Pedigree:
    """A is the aunt/uncle of B."""
    return _build([
        ("L1", (), MALE), ("L2", (), FEMALE),
        ("A", ("L1", "L2"), None), ("L3", ("L1", "L2"), MALE),
        ("B", ("L3",), None),
    ])


def first_cousins() -> Pedigree:
    return _build([
        ("L1", (), MALE), ("L2", (), FEMALE),
        ("L3", ("L1", "L2"), MALE), ("L4", ("L1", "L2"), FEMALE),
        ("A", ("L3",), None), ("B", ("L4",), None),
    ])


def double_first_cousins() -> Pedigree:
    """Two brothers married two sisters."""
    return _build([
        ("L1", (), MALE), ("L2", (), FEMALE), ("L3", (), MALE), ("L4", (), FEMALE),
        ("L5", ("L1", "L2"), MALE), ("L6", ("L1", "L2"), MALE),
        ("L7", ("L3", "L4"), FEMALE), ("L8", ("L3", "L4"), FEMALE),
        ("A", ("L5", "L7"), None), ("B", ("L6", "L8"), None),
    ])


def sibs_with_children() -> Pedigree:
    """Three observed siblings; two have an observed child each."""
    return _build([
        ("L1", (), MALE), ("L2", (), FEMALE),
        ("A", ("L1", "L2"), MALE), ("B", ("L1", "L2"), FEMALE), ("C", ("L1", "L2"), MALE),
        ("D", ("A",), None), ("E", ("B",), None),
    ])


def two_families() -> Pedigree:
    """Full siblings A, B plus an unrelated parent-child pair C, D."""
    return _build([
        ("L1", (), MALE), ("L2", (), FEMALE),
        ("A", ("L1", "L2"), None), ("B", ("L1", "L2"), None),
        ("C", (), MALE), ("D", ("C",), None),
    ])


def related_parents() -> Pedigree:
    """B is the child of first cousins A and L5 (B's parents are related)."""
    return _build([
        ("L1", (), MALE), ("L2", (), FEMALE),
        ("L3", ("L1", "L2"), MALE), ("L4", ("L1", "L2"), FEMALE),
        ("A", ("L3",), MALE), ("L5", ("L4",), FEMALE),
        ("B", ("A", "L5"), None),
    ])


STANDARD: dict[str, Callable[[], Pedigree]] = {
    "parent_child": parent_child,
    "trio": trio,
    "three_generations": three_generations,
    "full_sibs": full_sibs,
    "half_sibs": half_sibs,
    "grandparent": grandparent,
    "avuncular": avuncular,
    "first_cousins": first_cousins,
    "double_first_cousins": double_first_cousins,
    "sibs_with_children": sibs_with_children,
    "two_families": two_families,
    "related_parents": related_parents,
}


# ---- random pedigrees -----------------------------------------------------

def random_pedigree(
    n_observed: int = 5,
    n_generations: int = 3,
    n_founder_couples: int = 2,
    mean_children: float = 2.0,
    p_marry_in: float = 0.6,
    p_marry_within: float = 0.2,
    allow_inbreeding: bool = False,
    observed_generations: int = 2,
    seed: int | np.random.Generator | None = None,
    max_tries: int = 100,
) -> Pedigree:
    """Simulate a multi-generation population and hide most of it.

    Generation 0 has `n_founder_couples` couples. In each later generation every
    couple has Poisson(`mean_children`) children. Each child then, with
    probability `p_marry_within`, pairs with another person of the same
    generation (unrelated to them unless `allow_inbreeding`), or otherwise, with
    probability `p_marry_in`, with a new founder spouse.

    `n_observed` people are drawn from the last `observed_generations`
    generations; everyone else becomes latent, and latent people irrelevant to
    the observed ones are pruned. Retries if too few candidates are born.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        ped = _simulate_population(rng, n_generations, n_founder_couples, mean_children,
                                   p_marry_in, p_marry_within, allow_inbreeding)
        if ped is None:
            continue
        gens = {pid: int(pid[1:].split("_")[0]) for pid in ped}
        candidates = [pid for pid in ped if gens[pid] >= n_generations - observed_generations]
        if len(candidates) < n_observed:
            continue
        chosen = set(rng.choice(candidates, size=n_observed, replace=False).tolist())
        for pid in ped:
            ped[pid].observed = pid in chosen
        ped = ped.pruned()
        obs = sorted(ped.observed_ids, key=lambda p: (gens[p], p))
        lat = sorted(ped.latent_ids, key=lambda p: (gens[p], p))
        mapping = {pid: f"P{k + 1}" for k, pid in enumerate(obs)}
        mapping.update({pid: f"L{k + 1}" for k, pid in enumerate(lat)})
        return ped.relabel(mapping)
    raise RuntimeError("could not generate enough candidates; increase mean_children "
                       "or n_founder_couples, or decrease n_observed")


def _simulate_population(rng, n_generations, n_founder_couples, mean_children,
                         p_marry_in, p_marry_within, allow_inbreeding) -> Pedigree | None:
    ped = Pedigree()
    counter = [0]

    def new(gen: int, parents=(), sex=None) -> str:
        counter[0] += 1
        pid = f"g{gen}_{counter[0]}"
        ped.add(pid, parents, sex or rng.choice([MALE, FEMALE]), observed=False)
        return pid

    couples = [(new(0, sex=MALE), new(0, sex=FEMALE)) for _ in range(n_founder_couples)]
    for gen in range(1, n_generations):
        children = [new(gen, couple) for couple in couples
                    for _ in range(rng.poisson(mean_children))]
        if not children:
            return None
        if gen == n_generations - 1:
            break
        couples = []
        order = [children[i] for i in rng.permutation(len(children))]
        paired: set[str] = set()
        for k, x in enumerate(order):
            if x in paired:
                continue
            mate = None
            if rng.random() < p_marry_within:
                for y in order[k + 1:]:
                    if y in paired or ped[y].sex == ped[x].sex:
                        continue
                    if allow_inbreeding or not (ped.ancestors(x) & ped.ancestors(y)):
                        mate = y
                        break
            if mate is None and rng.random() < p_marry_in:
                mate = new(gen, sex=FEMALE if ped[x].sex == MALE else MALE)
            if mate is not None:
                paired.update((x, mate))
                couples.append((x, mate))
        if not couples:
            return None
    return ped


# ---- observations ---------------------------------------------------------

def observe_ibd(ped: Pedigree, noise_sd: float = 0.0,
                seed: int | np.random.Generator | None = None) -> IBDData:
    """Expected IBD0/1/2 among observed people, plus optional Gaussian noise.

    Noisy values are clipped at 0 and rescaled to sum to 1. Gaussian noise is a
    crude stand-in; use Ped-sim for realistic variation.
    """
    exact = expected_ibd(ped)
    if noise_sd == 0:
        return exact
    rng = np.random.default_rng(seed)
    values = {}
    for pair, v in exact.items():
        k = np.clip(np.asarray(v) + rng.normal(0.0, noise_sd, 3), 0.0, None)
        values[pair] = k / k.sum()
    return IBDData(values, exact.ids)


def observe(ped: Pedigree, noise_sd: float = 0.0,
            seed: int | np.random.Generator | None = None) -> KinshipData:
    """Expected kinship among observed people, plus optional Gaussian noise.

    Gaussian noise is a crude stand-in; use Ped-sim for realistic variation.
    """
    exact = expected_kinship(ped)
    if noise_sd == 0:
        return exact
    rng = np.random.default_rng(seed)
    return KinshipData({pair: v + rng.normal(0.0, noise_sd) for pair, v in exact.items()},
                       exact.ids)
