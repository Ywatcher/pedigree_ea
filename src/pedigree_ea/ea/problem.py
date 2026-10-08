"""The search problem: a pairwise target, bounds, and objectives on decoded pedigrees.

The target is one of
    IBDData      IBD0/1/2; per-pair error = max |dIBD0|, |dIBD1|, |dIBD2|
    KinshipData  kinship;  per-pair error = |dkinship|
    KingData     kinship + IBS0 (-> IBD0 = IBS0 / unrelated IBS0); per-pair error
                 = max(|dkinship|, |dIBD0| * tol / tol_ibd0), so a pair is within
                 tolerance iff |dkinship| <= tol and |dIBD0| <= tol_ibd0.
                 Negative KING kinship is treated as 0.
The same rules as the brute-force search. All objectives are minimized.

Available phenotype objectives:
    ibd_total    sum of per-pair errors
    ibd_worst    largest per-pair error
    n_bad_pairs  number of pairs with error > tol
    excess_total sum over pairs of max(0, error - tol): every pedigree within
                 tolerance scores 0, so fits are not ranked by how well they fit noise
    excess_worst max(0, largest per-pair error - tol)
    n_latent     latent people that matter (kept by Pedigree.pruned)
    inbreeding   sum of inbreeding coefficients over observed and relevant latent people

A pedigree *fits* if it is valid (acyclic; sex-consistent if `check_sex`)
and every pair's error is <= tol. Invalid pedigrees get PENALTY on every objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..genetics import batch
from ..genetics.ibd import IBDData
from ..genetics.king import KingData
from ..genetics.kinship import KinshipData

PENALTY = 1e6
OBJECTIVES = ("ibd_total", "ibd_worst", "n_bad_pairs", "excess_total", "excess_worst",
              "n_latent", "inbreeding")


@dataclass
class Evaluation:
    objectives: np.ndarray   # (B, M)
    pair_error: np.ndarray   # (B, P)
    valid: np.ndarray        # (B,)
    fits: np.ndarray         # (B,)


@dataclass
class Problem:
    target: IBDData | KinshipData | KingData
    max_latent: int
    tol: float = 0.01
    tol_ibd0: float = 0.15          # KingData targets only
    objectives: tuple[str, ...] = ("ibd_total", "ibd_worst", "n_latent")
    check_sex: bool = True
    cache: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        unknown = set(self.objectives) - set(OBJECTIVES)
        if unknown:
            raise ValueError(f"unknown objectives {sorted(unknown)}; choose from {OBJECTIVES}")
        self.ids = list(self.target.ids)
        self.n_obs = len(self.ids)
        self.n = self.n_obs + self.max_latent
        idx = {pid: i for i, pid in enumerate(self.ids)}
        pairs = list(self.target)
        self.pi = np.array([idx[a] for a, _ in pairs], dtype=np.intp)
        self.pj = np.array([idx[b] for _, b in pairs], dtype=np.intp)
        if isinstance(self.target, KingData):
            self.kind = "king"
            values = [(max(self.target[p].kinship, 0.0), self.target.ibd0(*p)) for p in pairs]
        elif isinstance(self.target, KinshipData):
            self.kind = "kinship"
            values = [(self.target[p],) for p in pairs]
        else:
            self.kind = "ibd"
            values = [tuple(self.target[p]) for p in pairs]
        self.target_values = np.array(values, dtype=float).reshape(len(pairs), -1)
        # Per-column weights so that one `tol` applies to every column.
        self.weights = np.array([1.0, self.tol / self.tol_ibd0] if self.kind == "king"
                                else [1.0] * self.target_values.shape[1])

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    def evaluate(self, parents: np.ndarray) -> Evaluation:
        """Evaluate a batch of (normalized) parent arrays, shape (B, N, 2)."""
        parents = batch.normalize(parents)
        if not self.cache:
            return self._evaluate(parents)
        keys = [batch.key(p) for p in parents]
        todo = [b for b, k in enumerate(keys) if k not in self._cache]
        if todo:
            fresh = self._evaluate(parents[todo])
            for k, b in enumerate(todo):
                self._cache[keys[b]] = (fresh.objectives[k], fresh.pair_error[k],
                                        fresh.valid[k], fresh.fits[k])
        rows = [self._cache[k] for k in keys]
        return Evaluation(*(np.array([r[c] for r in rows]) for c in range(4)))

    def _evaluate(self, parents: np.ndarray) -> Evaluation:
        b_dim = parents.shape[0]
        anc = batch.ancestor_matrix(parents)
        valid = ~batch.has_cycle(parents, anc)
        if self.check_sex:
            valid &= ~batch.sex_conflict(parents)
        phi, f = batch.kinship(parents, valid)
        kin = phi[:, self.pi, self.pj][:, :, None]
        if self.kind == "kinship":
            predicted = kin
        else:
            ibd = batch.ibd(parents, self.pi, self.pj, phi, f, anc)
            predicted = ibd if self.kind == "ibd" else np.concatenate([kin, ibd[:, :, :1]], axis=2)
        diff = np.abs(predicted - self.target_values[None]) * self.weights
        err = np.nan_to_num(diff.max(axis=2), nan=PENALTY)
        relevant = batch.relevant_latent(parents, self.n_obs, anc)
        values = {
            "ibd_total": err.sum(axis=1),
            "ibd_worst": err.max(axis=1, initial=0.0),
            "n_bad_pairs": (err > self.tol).sum(axis=1).astype(float),
            "excess_total": np.maximum(err - self.tol, 0.0).sum(axis=1),
            "excess_worst": np.maximum(err.max(axis=1, initial=0.0) - self.tol, 0.0),
            "n_latent": relevant.sum(axis=1).astype(float),
        }
        if "inbreeding" in self.objectives:
            keep = relevant.copy()
            keep[:, :self.n_obs] = True
            values["inbreeding"] = np.where(keep, np.nan_to_num(f), 0.0).sum(axis=1)
        obj = np.stack([values[name] for name in self.objectives], axis=1)
        obj[~valid] = PENALTY
        err = np.where(valid[:, None], err, PENALTY)
        fits = valid & (err.max(axis=1, initial=0.0) <= self.tol)
        return Evaluation(obj, err, valid, fits)
