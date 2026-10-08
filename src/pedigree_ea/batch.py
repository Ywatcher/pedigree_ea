"""Pedigrees as fixed-size arrays, and batched evaluation with numpy.

A pedigree with `n_obs` observed people and up to `max_latent` latent people
is an int array `parents` of shape (N, 2), N = n_obs + max_latent:
    slots 0 .. n_obs-1   observed people, in the order of the target's ids
    slots n_obs .. N-1   latent people
    parents[i] = (p, q)  i's parents; -1 = missing
A batch has shape (B, N, 2). Rows are kept normalized: valid parents first in
increasing order, no duplicate parent (see `normalize`).

Main entry points:
    ancestor_matrix   transitive closure, cycle detection
    kinship           exact kinship (Henderson's A = L D L^T), batched
    ibd               exact IBD0/1/2 for given pairs; closed form for outbred
                      pairs, exact gene dropping (ibd.py) for inbred ones
    relevant_latent   latent people that survive `Pedigree.pruned`
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .ibd import ibd_from_parents
from .pedigree import Pedigree

MISSING = -1
_EPS = 1e-12


# ---- conversion -------------------------------------------------------------

def normalize(parents: np.ndarray) -> np.ndarray:
    """Sort each parent pair (missing last) and drop a duplicated parent. Returns a copy."""
    p = np.array(parents, dtype=np.int16, copy=True)
    a, b = p[..., 0].copy(), p[..., 1].copy()
    b = np.where(a == b, MISSING, b)
    lo = np.where((a < 0) | ((b >= 0) & (b < a)), b, a)
    hi = np.where((a < 0) | ((b >= 0) & (b < a)), a, b)
    p[..., 0], p[..., 1] = lo, hi
    return p


def from_pedigree(ped: Pedigree, observed_ids: Sequence[str], max_latent: int) -> np.ndarray:
    """Pedigree -> (N, 2) array. Raises ValueError if it has too many latent people."""
    latent = [pid for pid in ped.ids if pid not in set(observed_ids)]
    if len(latent) > max_latent:
        raise ValueError(f"{len(latent)} latent people > max_latent={max_latent}")
    idx = {pid: i for i, pid in enumerate(list(observed_ids) + latent)}
    out = np.full((len(observed_ids) + max_latent, 2), MISSING, dtype=np.int16)
    for pid in ped.ids:
        for k, par in enumerate(ped[pid].parents):
            out[idx[pid], k] = idx[par]
    return normalize(out)


def to_pedigree(parents: np.ndarray, observed_ids: Sequence[str], prune: bool = True) -> Pedigree:
    """(N, 2) array -> Pedigree; unused latent slots are dropped, latent named L1, L2, ..."""
    n_obs = len(observed_ids)
    n = parents.shape[0]
    used = {int(x) for x in parents[parents >= 0]} | {i for i in range(n_obs, n)
                                                      if (parents[i] >= 0).any()}
    latent = [i for i in range(n_obs, n) if i in used]
    names = {i: pid for i, pid in enumerate(observed_ids)}
    names.update({i: f"L{k + 1}" for k, i in enumerate(latent)})
    ped = Pedigree()
    for i in list(range(n_obs)) + latent:
        ped.add(names[i], [names[int(p)] for p in parents[i] if p >= 0], observed=i < n_obs)
    return ped.pruned() if prune else ped


def key(parents: np.ndarray) -> bytes:
    """Hashable key of one normalized (N, 2) array (not invariant to latent relabelling)."""
    return np.ascontiguousarray(parents, dtype=np.int16).tobytes()


# ---- structure ----------------------------------------------------------------

def adjacency(parents: np.ndarray) -> np.ndarray:
    """adj[b, p, c] = True if p is a parent of c."""
    b_dim, n = parents.shape[0], parents.shape[1]
    adj = np.zeros((b_dim, n, n), dtype=bool)
    bi, ci, ki = np.nonzero(parents >= 0)
    adj[bi, parents[bi, ci, ki], ci] = True
    return adj


def ancestor_matrix(parents: np.ndarray) -> np.ndarray:
    """anc[b, i, j] = True if i is a (strict) ancestor of j. Cycles show on the diagonal."""
    reach = adjacency(parents)
    n = reach.shape[1]
    steps = max(1, int(np.ceil(np.log2(max(n, 2)))) + 1)
    r = reach.astype(np.int32)
    for _ in range(steps):
        new = (r | ((r @ r) > 0)).astype(np.int32)
        if np.array_equal(new, r):
            break
        r = new
    return r.astype(bool)


def has_cycle(parents: np.ndarray, anc: np.ndarray | None = None) -> np.ndarray:
    anc = ancestor_matrix(parents) if anc is None else anc
    return np.diagonal(anc, axis1=1, axis2=2).any(axis=1)


def relevant_latent(parents: np.ndarray, n_obs: int, anc: np.ndarray | None = None) -> np.ndarray:
    """Mask (B, N) of latent people kept by `Pedigree.pruned`: latent people with an
    observed descendant, after repeatedly removing latent founders with one child."""
    anc = ancestor_matrix(parents) if anc is None else anc
    b_dim, n = parents.shape[:2]
    latent = np.zeros((b_dim, n), dtype=bool)
    latent[:, n_obs:] = True
    alive = ~latent | anc[:, :, :n_obs].any(axis=2)
    adj = adjacency(parents)
    for _ in range(n):
        live_parent = adj & alive[:, :, None]          # p alive and parent of c
        n_children = (adj & alive[:, None, :]).sum(axis=2)
        is_founder = ~live_parent.any(axis=1)
        drop = latent & alive & is_founder & (n_children == 1)
        if not drop.any():
            break
        alive &= ~drop
    return latent & alive


def sex_conflict(parents: np.ndarray) -> np.ndarray:
    """(B,) True if couples cannot be given opposite sexes (mate graph not bipartite)."""
    out = np.zeros(parents.shape[0], dtype=bool)
    for b in range(parents.shape[0]):
        rows = parents[b][(parents[b] >= 0).all(axis=1)]
        if len(rows) < 3:
            continue
        adj: dict[int, set[int]] = {}
        for p, q in rows.tolist():
            adj.setdefault(p, set()).add(q)
            adj.setdefault(q, set()).add(p)
        color: dict[int, int] = {}
        for s in adj:
            if s in color:
                continue
            color[s] = 0
            stack = [s]
            while stack and not out[b]:
                x = stack.pop()
                for y in adj[x]:
                    if y not in color:
                        color[y] = 1 - color[x]
                        stack.append(y)
                    elif color[y] == color[x]:
                        out[b] = True
                        break
    return out


# ---- kinship and IBD ------------------------------------------------------------

def kinship(parents: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Kinship (B, N, N) and inbreeding F (B, N) for acyclic pedigrees.

    Uses the additive relationship matrix A = 2 phi = L D L^T with
    L = (I - T)^-1, T[i, p] = 1/2 for each parent p of i, and Mendelian sampling
    variances D_ii = 1/2 - (F_p + F_q)/4 (two parents), 3/4 - F_p/4 (one), 1 (none).
    D depends on the parents' F, so this iterates until F stops changing.
    Rows where `valid` is False (e.g. cyclic) are returned as NaN.
    """
    b_dim, n = parents.shape[:2]
    par = parents.copy()
    if valid is not None:
        par[~valid] = MISSING
    t = np.zeros((b_dim, n, n))
    bi, ci, ki = np.nonzero(par >= 0)
    np.add.at(t, (bi, ci, par[bi, ci, ki]), 0.5)
    lmat = np.linalg.inv(np.eye(n) - t)
    has = par >= 0
    n_par = has.sum(axis=2)
    f = np.zeros((b_dim, n))
    for _ in range(n + 1):
        fp = np.where(has, np.take_along_axis(f[:, :, None].repeat(2, 2),
                                              np.where(has, par, 0).astype(np.intp), axis=1), 0.0)
        d = np.where(n_par == 2, 0.5 - 0.25 * fp.sum(axis=2),
                     np.where(n_par == 1, 0.75 - 0.25 * fp.sum(axis=2), 1.0))
        a = lmat @ (d[:, :, None] * np.swapaxes(lmat, 1, 2))
        f_new = np.diagonal(a, axis1=1, axis2=2) - 1.0
        if np.allclose(f_new, f, atol=1e-14):
            f = f_new
            break
        f = f_new
    phi = a / 2
    if valid is not None:
        phi[~valid] = np.nan
        f[~valid] = np.nan
    return phi, np.where(np.abs(f) < _EPS, 0.0, f)


def ibd(parents: np.ndarray, pi: np.ndarray, pj: np.ndarray, phi: np.ndarray, f: np.ndarray,
        anc: np.ndarray) -> np.ndarray:
    """Exact (k0, k1, k2) for pairs (pi[k], pj[k]); returns (B, P, 3).

    Outbred pairs (neither person inbred) use the closed form
        k2 = phi(a1,b1) phi(a2,b2) + phi(a1,b2) phi(a2,b1)
    over the parents a1, a2 of i and b1, b2 of j (0 if one is an ancestor of
    the other), k1 = 4 phi(i,j) - 2 k2. Pairs with an inbred person fall back
    to exact gene dropping (ibd.ibd_from_parents) for that pedigree.
    """
    b_dim, n = parents.shape[:2]
    phip = np.zeros((b_dim, n + 1, n + 1))   # extra row/col = missing parent
    phip[:, :n, :n] = np.nan_to_num(phi)
    par = np.where(parents >= 0, parents, n).astype(np.intp)
    bidx = np.arange(b_dim)[:, None]
    a1, a2 = par[:, pi, 0], par[:, pi, 1]
    b1, b2 = par[:, pj, 0], par[:, pj, 1]
    k2 = phip[bidx, a1, b1] * phip[bidx, a2, b2] + phip[bidx, a1, b2] * phip[bidx, a2, b1]
    lineal = anc[:, pi, pj] | anc[:, pj, pi]
    k2 = np.where(lineal, 0.0, k2)
    fij = phip[bidx, pi, pj]
    k1 = 4 * fij - 2 * k2
    out = np.stack([1.0 - k1 - k2, k1, k2], axis=2)
    inbred = (f[:, pi] > _EPS) | (f[:, pj] > _EPS)
    for b in np.nonzero(inbred.any(axis=1))[0]:
        cols = np.nonzero(inbred[b])[0]
        plist = [[int(x) for x in row if x >= 0] for row in parents[b]]
        vals = ibd_from_parents(plist, list(zip(pi[cols].tolist(), pj[cols].tolist())), phi[b])
        out[b, cols] = np.asarray(vals)
    return out
