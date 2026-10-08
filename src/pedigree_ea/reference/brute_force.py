"""Exhaustive search for all pedigrees that fit an IBD or kinship target.

Only feasible for small cases (roughly up to 6-7 people including latent
ones). Its purpose is ground truth: the complete solution set an EA should
find, so EA results can be scored by recall.

Search: assign each person's parents (none, one or two of the other people)
in turn, depth first. Branches are cut when
- the new parents create a cycle,
- couples can no longer be given opposite sexes (if `check_sex`),
- a pair can no longer fit, because adding parents only moves its values one
  way: kinship and IBD2 never decrease, IBD0 never increases. So a branch is
  cut once kinship > target + tol, IBD2 > target + tol or IBD0 < target - tol.
Latent people are interchangeable, so they are introduced in order (L1 before
L2, ...), which avoids enumerating relabelled copies.
"""

from __future__ import annotations

from itertools import combinations

from ..genetics.canonical import PedigreeSet
from ..genetics.ibd import IBDData, ibd_from_parents
from ..genetics.king import KingData
from ..genetics.kinship import KinshipData, inbreeding, kinship_from_parents
from ..genetics.pedigree import Pedigree


def _kinship_checks(target: KinshipData, tol: float):
    def measure(parents, pairs):
        phi = kinship_from_parents(parents)
        return [phi[i, j] for i, j in pairs]

    def can_fit(values):
        return all(v <= t + tol for v, t in zip(values, target_values))

    def fits(values):
        return all(abs(v - t) <= tol for v, t in zip(values, target_values))

    target_values = [v for _, v in target.items()]
    return measure, can_fit, fits


def _ibd_checks(target: IBDData, tol: float):
    # Cheap test first: kinship <= k2 + k1/2 in every identity state, and
    # kinship never decreases, so kinship above this bound can never fit.
    target_values = [v for _, v in target.items()]
    max_kinship = [t.k2 + t.k1 / 2 + 1.5 * tol for t in target_values]

    def measure(parents, pairs):
        phi = kinship_from_parents(parents)
        if any(phi[i, j] > m for (i, j), m in zip(pairs, max_kinship)):
            return None
        return ibd_from_parents(parents, pairs, phi)

    def can_fit(values):
        return values is not None and all(v.k2 <= t.k2 + tol and v.k0 >= t.k0 - tol
                                          for v, t in zip(values, target_values))

    def fits(values):
        return all(abs(a - b) <= tol for v, t in zip(values, target_values)
                   for a, b in zip(v, t))

    return measure, can_fit, fits


def _king_checks(target: KingData, tol: float, tol_ibd0: float):
    """Kinship within tol and IBD0 (= IBS0 / unrelated IBS0) within tol_ibd0.
    Prunes on kinship > target + tol and IBD0 < target - tol_ibd0 (both monotone)."""
    t_kin = [max(v.kinship, 0.0) for _, v in target.items()]
    t_k0 = [target.ibd0(*p) for p in target]

    def measure(parents, pairs):
        phi = kinship_from_parents(parents)
        kin = [phi[i, j] for i, j in pairs]
        if any(k > t + tol for k, t in zip(kin, t_kin)):
            return None
        return kin, [v.k0 for v in ibd_from_parents(parents, pairs, phi)]

    def can_fit(values):
        return values is not None and all(k0 >= t - tol_ibd0 for k0, t in zip(values[1], t_k0))

    def fits(values):
        kin, k0 = values
        return (all(abs(a - b) <= tol for a, b in zip(kin, t_kin))
                and all(abs(a - b) <= tol_ibd0 for a, b in zip(k0, t_k0)))

    return measure, can_fit, fits


def enumerate_pedigrees(
    target: IBDData | KinshipData | KingData,
    max_latent: int = 1,
    tol: float = 0.01,
    check_sex: bool = True,
    allow_inbreeding: bool = True,
    max_results: int | None = None,
    tol_ibd0: float = 0.15,
) -> list[Pedigree]:
    """All distinct pedigrees with at most `max_latent` latent people whose
    expected values are within `tol` of every pair in `target`. For an IBD
    target, each of IBD0, IBD1 and IBD2 must be within `tol`; for a KING target
    (kinship + IBS0), kinship within `tol` and IBD0 within `tol_ibd0`.

    Results are pruned (see `Pedigree.pruned`), so some have fewer latent people.
    """
    obs = list(target.ids)
    n_obs = len(obs)
    names = obs + [f"L{k + 1}" for k in range(max_latent)]
    n = len(names)
    idx = {pid: i for i, pid in enumerate(names)}
    pairs = [(idx[a], idx[b]) for a, b in target]
    if isinstance(target, KingData):
        measure, can_fit, fits = _king_checks(target, tol, tol_ibd0)
    elif isinstance(target, IBDData):
        measure, can_fit, fits = _ibd_checks(target, tol)
    else:
        measure, can_fit, fits = _kinship_checks(target, tol)

    options = []
    for k in range(n):
        others = [i for i in range(n) if i != k]
        options.append([()] + [(p,) for p in others] + list(combinations(others, 2)))

    parents: list[tuple[int, ...]] = [()] * n
    found = PedigreeSet(prune=True)

    def is_ancestor_or_self(a: int, b: int) -> bool:
        """Is `a` equal to or an ancestor of `b` under the current assignment?"""
        stack, seen = [b], set()
        while stack:
            x = stack.pop()
            if x == a:
                return True
            if x not in seen:
                seen.add(x)
                stack.extend(parents[x])
        return False

    def mates_bipartite() -> bool:
        adj: dict[int, list[int]] = {}
        for ps in parents:
            if len(ps) == 2:
                adj.setdefault(ps[0], []).append(ps[1])
                adj.setdefault(ps[1], []).append(ps[0])
        color: dict[int, int] = {}
        for s in adj:
            if s in color:
                continue
            color[s] = 0
            stack = [s]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in color:
                        color[y] = 1 - color[x]
                        stack.append(y)
                    elif color[y] == color[x]:
                        return False
        return True

    def leaf(values) -> None:
        if not fits(values):
            return
        ped = Pedigree()
        for i, pid in enumerate(names):
            ped.add(pid, [names[p] for p in parents[i]], observed=i < n_obs)
        if not allow_inbreeding and any(f > 1e-12 for f in inbreeding(ped.pruned()).values()):
            return
        found.add(ped)

    def dfs(k: int, n_used_latent: int, values) -> None:
        if max_results is not None and len(found) >= max_results:
            return
        # Latent people k-n_obs and later are not anyone's parent yet, and only
        # later (also unused) latent people remain, so nothing can use them.
        if k == n or k - n_obs >= n_used_latent:
            leaf(values)
            return
        for ps in options[k]:
            new_latent = sorted(p - n_obs for p in ps if p - n_obs >= n_used_latent)
            if new_latent != list(range(n_used_latent, n_used_latent + len(new_latent))):
                continue
            if any(is_ancestor_or_self(k, p) for p in ps):
                continue
            parents[k] = ps
            if not check_sex or mates_bipartite():
                new_values = measure(parents, pairs)
                if can_fit(new_values):
                    dfs(k + 1, n_used_latent + len(new_latent), new_values)
            parents[k] = ()

    dfs(0, 0, measure(parents, pairs))
    return list(found)
